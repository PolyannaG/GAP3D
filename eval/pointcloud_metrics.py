import os
import sys
import json
import math
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from PIL import Image
import utils3d
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "point-e"))
from point_e.evals.feature_extractor import PointNetClassifier
from point_e.evals.fid_is import compute_statistics, compute_inception_score
from point_e.evals.npz_stream import NpzStreamer
from point_e.util.point_cloud import PointCloud

sys.path.insert(0, str(Path(__file__).parent.parent / "TRELLIS"))
from trellis.utils.render_utils import (
    yaw_pitch_r_fov_to_extrinsics_intrinsics,
    render_frames_eval as render_frames,
)

# ========================= Utility: normalization =========================

def normalize_point_cloud(pc: PointCloud) -> PointCloud:
    """
    Normalizes a PointCloud to fit within a unit sphere (radius 1.0).
    """
    if pc.coords.shape[0] == 0:
        print("Warning: Cannot normalize an empty point cloud.")
        return pc

    coords_centered = pc.coords - pc.coords.mean(axis=0)
    max_l2_norm = np.max(np.linalg.norm(coords_centered, axis=1))

    if max_l2_norm < 1e-6:
        return PointCloud(coords=coords_centered, channels=pc.channels)

    coords_normalized = coords_centered / max_l2_norm
    return PointCloud(coords=coords_normalized, channels=pc.channels)


# ========================= TRELLIS backprojection =========================

def trellis_to_camera_params(
    extr_world2cam: torch.Tensor,
    intr_norm: torch.Tensor,
    H: int,
    W: int,
) -> Dict[str, np.ndarray]:
    """
    Convert TRELLIS intrinsics/extrinsics to Blender-like camera params:
    """
    intr = intr_norm.detach().cpu().numpy().astype(np.float32)
    fx_norm, fy_norm = intr[0, 0], intr[1, 1]
    cx_norm, cy_norm = intr[0, 2], intr[1, 2]

    fx = fx_norm * W
    fy = fy_norm * H
    cx = cx_norm * W
    cy = cy_norm * H

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    E = extr_world2cam.detach().cpu().numpy().astype(np.float32)
    T_cam2world = np.linalg.inv(E).astype(np.float32)

    return {"intrinsics": K, "extrinsics": T_cam2world}



def _extract_pointcloud_from_trellis_view(
    depth_img: np.ndarray,          # from MeshRenderer ("depth")
    color_img: np.ndarray | None,   # from MeshRenderer ("color")
    mask_img: np.ndarray,           # from MeshRenderer ("mask")
    extr_world2cam: torch.Tensor,   # [4,4] from yaw_pitch_r_fov_to_extrinsics_intrinsics
    intr_norm: torch.Tensor,        # [3,3] from intrinsics_from_fov_xy
    include_color: bool = False,
    rgb_img: np.ndarray | None = None,
    near_clip: float = 1e-4,
) -> Optional[PointCloud]:
    """
    Back-project a single TRELLIS mesh view into world space.
    depth_img is Z_cam (not percent / ray distance).
    intr_norm is the normalized intrinsics returned by intrinsics_from_fov_xy.
    uv are in [0,1] as expected by utils3d.torch.unproject_cv.
    """

    # ----- 1. Prep depth, mask -----
    depth = np.asarray(depth_img, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]

    H, W = depth.shape

    mask = np.asarray(mask_img, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[..., 0]

    depth_flat = depth.reshape(-1)
    mask_flat = mask.reshape(-1)

    valid = (
        (mask_flat > 0.999) &
        np.isfinite(depth_flat) &
        (depth_flat > near_clip)
    )
    if not np.any(valid):
        print("No valid pixels after depth+mask filter -> skipping view")
        return None

    depth_valid = depth_flat[valid]   # Z_cam

    # ----- 2. Build uv in [0,1] -----
    u, v = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32),
    )  # v: rows, u: cols

    u = u.reshape(-1)[valid]
    v = v.reshape(-1)[valid]

    # pixel -> uv in [0,1], with +0.5 for pixel centers (same as utils3d.pixel_to_uv)
    u_c = (u + 0.5) / float(W)
    v_c = (v + 0.5) / float(H)
    uv = np.stack([u_c, v_c], axis=-1)  # [N,2]

    # ----- 3. Unproject using utils3d (this handles intrinsics + extrinsics) -----
    device = extr_world2cam.device
    uv_t = torch.from_numpy(uv).to(device)[None, ...]           # [1,N,2]
    depth_t = torch.from_numpy(depth_valid).to(device)[None, ...]  # [1,N]

    pts_world = utils3d.torch.unproject_cv(
        uv_coord=uv_t,
        depth=depth_t,
        extrinsics=extr_world2cam[None, ...],  # world->cam
        intrinsics=intr_norm[None, ...],       # normalized intrinsics
    )[0]  # [N,3]

    coords_world = pts_world.detach().cpu().numpy()

    # ----- 4. Optional color -----
    channels = {}
    if include_color and rgb_img is not None:
        rgb = np.asarray(rgb_img, dtype=np.uint8)
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        elif rgb.shape[2] == 4:
            rgb = rgb[..., :3]

        rgb_flat = rgb.reshape(-1, 3)[valid] / 255.0
        channels["R"] = rgb_flat[:, 0]
        channels["G"] = rgb_flat[:, 1]
        channels["B"] = rgb_flat[:, 2]

    return PointCloud(coords=coords_world, channels=channels)



def extract_pointcloud_from_trellis_renders(
    rgb_images: List[np.ndarray],
    depth_images: List[np.ndarray],
    mask_images: List[np.ndarray],
    extrinsics: List[torch.Tensor],
    intrinsics: List[torch.Tensor],
    n_points: int = 4096,
    include_color: bool = False,
) -> Optional[PointCloud]:
    if not rgb_images or not depth_images:
        print("No RGB/depth images provided to extract_pointcloud_from_trellis_renders.")
        return None

    all_coords: List[np.ndarray] = []
    all_r: List[np.ndarray] = []
    all_g: List[np.ndarray] = []
    all_b: List[np.ndarray] = []

    for view_idx, (rgb_img, depth_img, mask_img, extr, intr) in enumerate(
        zip(rgb_images, depth_images, mask_images, extrinsics, intrinsics)
    ):
        if depth_img is None:
            continue

        depth_np = np.asarray(depth_img, dtype=np.float32)
        if depth_np.ndim == 3:
            depth_np = depth_np[..., 0]

        invalid = (~np.isfinite(depth_np)) | (depth_np <= 1e-6)
        depth_np = depth_np.copy()
        depth_np[invalid] = 0.0

        # print(f"\n=== TRELLIS DEPTH STATS (view {view_idx}) ===")
        # print("min:", np.min(depth_np), "max:", np.max(depth_np),
        #       "mean:", np.mean(depth_np))
        # print("num zero:", np.sum(depth_np == 0), "num finite:", np.sum(np.isfinite(depth_np)))

        pc_view = _extract_pointcloud_from_trellis_view(
            depth_np,
            rgb_img,
            mask_img,
            extr_world2cam=extr,
            intr_norm=intr,
            include_color=include_color,
        )

        if pc_view is not None and pc_view.coords.shape[0] > 0:
            coords = pc_view.coords           
            # save_point_cloud_ply(
            #     pc_view,
            #     f"trellis_single_view_{view_idx}.ply",
            # )
            # print(f"Saved single-view point cloud trellis_single_view_{view_idx}.ply")

            all_coords.append(coords)
            if include_color and all(c in pc_view.channels for c in ("R", "G", "B")):
                all_r.append(pc_view.channels["R"])
                all_g.append(pc_view.channels["G"])
                all_b.append(pc_view.channels["B"])

    if not all_coords:
        print("No valid points extracted from TRELLIS renders.")
        return None

    merged_coords = np.concatenate(all_coords, axis=0)

    merged_channels: Dict[str, np.ndarray] = {}

    if include_color and all_r:
        merged_channels["R"] = np.concatenate(all_r, axis=0)
        merged_channels["G"] = np.concatenate(all_g, axis=0)
        merged_channels["B"] = np.concatenate(all_b, axis=0)

    merged_pc = PointCloud(coords=merged_coords, channels=merged_channels)

    # # Save merged *before* FPS / normalize so you see raw geometry
    # save_point_cloud_ply(merged_pc, "trellis_merged_raw.ply")
    # print("Saved merged raw TRELLIS cloud to trellis_merged_raw.ply")

    merged_pc = merged_pc.farthest_point_sample(n_points)
    merged_pc = normalize_point_cloud(merged_pc)

    return merged_pc


def extract_pointcloud_from_generated(
    gen_obj_gaussian,
    gen_obj_mesh,
    n_views: int = 20,
    n_points: int = 4096,
) -> Optional[PointCloud]:
    """
    Extract a point cloud from a generated 3D object using TRELLIS:
      - render Gaussian (RGB) and Mesh (depth) from multiple views,
      - back-project using TRELLIS intrinsics + extrinsics,
      - FPS + normalize to unit sphere.
    """
    yaws = [i * (2.0 * math.pi / n_views) for i in range(n_views)]
    pitchs = [math.radians(30.0)] * n_views
    radius = 2.0
    fovs = 40.0  # degrees

    extr, intr = yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws, pitchs, rs=radius, fovs=fovs
    )

    try:
        # Gaussian for RGB
        rets1 = render_frames(
            gen_obj_gaussian,
            extr,
            intr,
            {
                "resolution": 512,
                "bg_color": [0, 0, 0],
            },
        )
        # Mesh for depth
        rets2 = render_frames(
            gen_obj_mesh,
            extr,
            intr,
            {
                "resolution": 512,
                "bg_color": [0, 0, 0],
            },
        )

        rgb_images = rets1.get("color")
        depth_images = rets2.get("depth")
        mask_images = rets2.get("mask")

        if not rgb_images or not depth_images or not mask_images:
            print("Rendering failed - no RGB or depth or mask images.")
            return None

        pc = extract_pointcloud_from_trellis_renders(
            rgb_images,
            depth_images,
            extrinsics=extr,
            intrinsics=intr,
            n_points=n_points,
            include_color=False,
            mask_images=mask_images,
        )
        return pc

    except Exception as e:
        print(f"Error extracting point cloud from rendered views: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_pointcloud_from_rgbad(
    depth_img: np.ndarray,
    rgb_img: np.ndarray,
    camera_params: Dict[str, np.ndarray],
    include_color: bool = False,
) -> PointCloud:
    """
    Extract point cloud from RGBD image (Blender GT path),
    assuming:
      - depth_img: distance from camera along view ray (after undoing MapRange),
      - intrinsics in pixel units,
      - extrinsics camera->world, Blender convention (forward = -Z, up = +Y).
    """
    H, W = depth_img.shape

    K = np.asarray(camera_params["intrinsics"], dtype=np.float32)  # [3,3]
    T = np.asarray(camera_params["extrinsics"], dtype=np.float32)  # [4,4], cam->world

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.flatten()
    v = v.flatten()
    depth = depth_img.flatten()

    # Valid depth
    valid_mask = (depth > 0) & np.isfinite(depth)
    if not np.any(valid_mask):
        return PointCloud(coords=np.zeros((0, 3), dtype=np.float32), channels={})

    u = u[valid_mask]
    v = v[valid_mask]
    depth = depth[valid_mask]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Normalized image coords
    x_n = (u - cx) / fx
    y_n = (v - cy) / fy

    # Blender camera: forward = -Z, up = +Y, but image v increases downward.
    # Ray direction in camera space (unnormalized):
    #   dir_cam ~ (x_n, -y_n, -1)
    den = np.sqrt(x_n**2 + y_n**2 + 1.0)
    t = depth / den  # distance along ray

    X = x_n * t
    Y = -y_n * t
    Z = -1.0 * t

    points_cam = np.stack([X, Y, Z], axis=1).astype(np.float32)

    # Camera -> world
    ones = np.ones((points_cam.shape[0], 1), dtype=np.float32)
    points_cam_h = np.concatenate([points_cam, ones], axis=1)
    coords = (T @ points_cam_h.T).T[:, :3]

    channels: Dict[str, np.ndarray] = {}
    if include_color:
        rgb = rgb_img.reshape(-1, 3)[valid_mask] / 255.0
        channels["R"] = rgb[:, 0]
        channels["G"] = rgb[:, 1]
        channels["B"] = rgb[:, 2]

    return PointCloud(coords=coords, channels=channels)



def load_pointcloud_from_blender_renders(
    render_dir: Path,
    n_points: int = 4096,
    include_color: bool = False,
) -> Optional[PointCloud]:
    """
    Load and merge point clouds from multiple Blender RGBD renders.

    Assumes:
      - transforms.json with camera_angle_x and frames list
      - each frame has file_path (RGBA PNG) and a *_depth.png
      - transform_matrix is camera->world
    """
    transforms_file = render_dir / "transforms.json"
    if not transforms_file.exists():
        return None

    with open(transforms_file) as f:
        data = json.load(f)

    all_coords: List[np.ndarray] = []
    all_r: List[np.ndarray] = []
    all_g: List[np.ndarray] = []
    all_b: List[np.ndarray] = []

    frames = data.get("frames", [])
    for frame in frames:
        file_path = frame.get("file_path")
        if not file_path:
            print("Warning: No file_path found in frame, skipping.")
            continue

        rgb_path = render_dir / Path(file_path).name
        depth_path = render_dir / (Path(file_path).stem + "_depth.png")

        rgba = np.array(Image.open(rgb_path).convert("RGBA"))
        rgb_img = rgba[..., :3]
        alpha = rgba[..., 3] / 255.0

        depth_img = np.array(Image.open(depth_path))
        if depth_img.ndim == 3:
            print("Warning: Depth image has 3 channels, taking the first channel only.")
            depth_img = depth_img[:, :, 0]

        depth_img = depth_img.astype(np.float32)

        depth_norm = depth_img / 65535.0

        # Get per-view near/far from transforms.json
        if "depth" in frame:
            near = frame["depth"]["min"]
            far  = frame["depth"]["max"]
            depth_metric = near + depth_norm * (far - near)
        else:
            print("Warning: No depth range info in frame, using normalized depth as-is.")
            depth_metric = depth_norm


        # Drop background using alpha
        background_mask = (alpha <= 0.999)
        depth_metric[background_mask] = 0.0

        transform_matrix = np.array(frame.get("transform_matrix"))
        if transform_matrix is None:
            print("Warning: No transform_matrix in frame, skipping.")
            continue

        fov = frame.get("camera_angle_x")
        if fov is None:
            print("Warning: No camera_angle_x in frame")
        H, W = rgb_img.shape[0], rgb_img.shape[1]
        fx = fy = W / (2.0 * math.tan(fov / 2.0))
        cx, cy = W / 2.0, H / 2.0
        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]])

        camera_params = {
            "intrinsics": K,
            "extrinsics": transform_matrix,  # camera->world
        }

        pc_view = extract_pointcloud_from_rgbad(
            depth_metric, rgb_img, camera_params, include_color
        )
        if pc_view is not None and len(pc_view.coords) > 0:
            all_coords.append(pc_view.coords)
            if include_color and all(c in pc_view.channels for c in ("R", "G", "B")):
                all_r.append(pc_view.channels["R"])
                all_g.append(pc_view.channels["G"])
                all_b.append(pc_view.channels["B"])

    if not all_coords:
        return None

    merged_coords = np.concatenate(all_coords, axis=0)
    merged_channels: Dict[str, np.ndarray] = {}

    if all_r and include_color:
        merged_channels["R"] = np.concatenate(all_r, axis=0)
        merged_channels["G"] = np.concatenate(all_g, axis=0)
        merged_channels["B"] = np.concatenate(all_b, axis=0)

    merged_pc = PointCloud(coords=merged_coords, channels=merged_channels)
    merged_pc = merged_pc.farthest_point_sample(n_points)
    merged_pc = normalize_point_cloud(merged_pc)
    return merged_pc


# ========================= Metrics (unchanged) =========================

class PointCloudStreamer(NpzStreamer):
    """Adapter to stream point clouds in the format expected by Point-E evaluators."""

    def __init__(self, point_clouds: List[PointCloud]):
        self.point_clouds = point_clouds
        self.index = 0

    def stream(self, batch_size: int, keys: List[str]):
        while self.index < len(self.point_clouds):
            batch_end = min(self.index + batch_size, len(self.point_clouds))
            batch = self.point_clouds[self.index: batch_end]
            batch_arr = np.stack([pc.coords for pc in batch], axis=0)
            yield {"arr_0": batch_arr}
            self.index = batch_end


def compute_p_fid_p_is(
    ref_point_clouds: List[PointCloud],
    gen_point_clouds: List[PointCloud],
    device: str = "cuda",
    batch_size: int = 64,
) -> Dict[str, float]:
    if not ref_point_clouds or not gen_point_clouds:
        return {"error": "Empty point cloud list"}

    devices = [device] if isinstance(device, str) else device
    extractor = PointNetClassifier(
        devices=devices,
        device_batch_size=batch_size,
        cache_dir=None,
    )

    print(f"Extracting features from {len(ref_point_clouds)} reference point clouds...")
    ref_streamer = PointCloudStreamer(ref_point_clouds)
    ref_features, ref_preds = extractor.features_and_preds(ref_streamer)

    print(f"Extracting features from {len(gen_point_clouds)} generated point clouds...")
    gen_streamer = PointCloudStreamer(gen_point_clouds)
    gen_features, gen_preds = extractor.features_and_preds(gen_streamer)

    ref_stats = compute_statistics(ref_features)
    gen_stats = compute_statistics(gen_features)

    p_fid = ref_stats.frechet_distance(gen_stats)
    p_is = compute_inception_score(gen_preds, split_size=min(5000, len(gen_preds)))

    return {
        "p_fid": float(p_fid),
        "p_is": float(p_is),
        "num_ref": len(ref_point_clouds),
        "num_gen": len(gen_point_clouds),
        "feature_dim": extractor.feature_dim,
        "num_classes": extractor.num_classes,
    }


def save_point_cloud_ply(point_cloud: PointCloud, output_path: str) -> None:
    with open(output_path, "wb") as f:
        point_cloud.write_ply(f)


def visualize_point_cloud_stats(point_clouds: List[PointCloud]) -> Dict:
    if not point_clouds:
        return {}

    n_points = [len(pc.coords) for pc in point_clouds]
    has_color = [all(c in pc.channels for c in ("R", "G", "B")) for pc in point_clouds]

    return {
        "num_clouds": len(point_clouds),
        "points_per_cloud_mean": float(np.mean(n_points)),
        "points_per_cloud_std": float(np.std(n_points)),
        "points_per_cloud_min": int(np.min(n_points)),
        "points_per_cloud_max": int(np.max(n_points)),
        "clouds_with_color": int(np.sum(has_color)),
        "clouds_without_color": int(len(has_color) - np.sum(has_color)),
    }
