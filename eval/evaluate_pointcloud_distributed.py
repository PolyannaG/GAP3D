import os
import sys
import json
import math
import argparse
import traceback
import random
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
from tqdm import tqdm
from PIL import Image
import torch

from pointcloud_metrics import (
    load_pointcloud_from_blender_renders,
    compute_p_fid_p_is,
    visualize_point_cloud_stats,
    save_point_cloud_ply,
    extract_pointcloud_from_generated,
    PointCloud,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "TRELLIS"))
from trellis.pipelines import (
    TrellisImageTo3DPipeline,
    TrellisTextTo3DPipeline,
    BlipTrellisImageTo3DPipeline,
)
from trellis.utils import render_utils
import imageio

from model_helpers import BlipTextEmbedder, ModelArguments, DataArguments


from utils import (
    select_even_indices,
    list_asset_ids,
    load_captions_by_sha,
)

from dist_helpers import (
    dist_setup,
    get_rank,
    get_world_size,
    barrier,
    shard_list,
    gather_objects_to_rank0,
)


# ========================= Utility: asset listing =========================

# def load_gt_pointclouds(
#     renders_root: str,
#     asset_ids: List[str],
#     n_points: int = 4096,
# ) -> Dict[str, PointCloud]:
#     """
#     Load ground-truth point clouds for each asset by unprojecting multiple views.
#     Uses your pointcloud_metrics helpers: load_pointcloud_from_blender_renders().

#     Returns:
#         { asset_id: PointCloud }
#     """
#     gt_clouds: Dict[str, PointCloud] = {}

#     for aid in asset_ids:
#         asset_dir = Path(renders_root) / aid

#         try:
#             pc = load_pointcloud_from_blender_renders(
#                 asset_dir,
#                 n_points=n_points,
#                 include_color=False,
#             )

#             if pc is None:
#                 print(f"[GT] Skipping {aid}: could not extract point cloud")
#                 continue

#             gt_clouds[aid] = pc

#         except Exception as e:
#             print(f"[GT] Error loading GT for {aid}: {e}")
#             continue

#     return gt_clouds

def load_gt_pointclouds(
    renders_root: str,
    asset_ids: List[str],
    n_points: int = 4096,
) -> Dict[str, PointCloud]:
    """
    Load ground-truth point clouds for each asset by unprojecting multiple views.
    Uses your pointcloud_metrics helpers: load_pointcloud_from_blender_renders().

    Returns:
        { asset_id: PointCloud }
    """
    gt_clouds: Dict[str, PointCloud] = {}

    for aid in asset_ids:
        asset_dir = Path(renders_root) / aid

        try:
            pc = load_pointcloud_from_blender_renders(
                asset_dir,
                n_points=n_points,
                include_color=False,
            )

            # Skip if we couldn't get a point cloud at all
            if pc is None:
                print(f"[GT] Skipping {aid}: could not extract point cloud (None)")
                continue

            # Skip if the cloud is empty or has unexpected shape
            if pc.coords is None or pc.coords.ndim != 2 or pc.coords.shape[0] == 0:
                print(
                    f"[GT] Skipping {aid}: invalid GT point cloud shape {None if pc.coords is None else pc.coords.shape}"
                )
                continue

            # (Optional but extra-safe) enforce consistent point count
            if pc.coords.shape[0] != n_points:
                # try to resample to n_points; if still wrong, skip
                try:
                    pc = pc.farthest_point_sample(n_points)
                except Exception as e:
                    print(f"[GT] Skipping {aid}: FPS resample failed: {e}")
                    continue

                if pc.coords.shape[0] != n_points:
                    print(
                        f"[GT] Skipping {aid}: expected {n_points} points, got {pc.coords.shape[0]}"
                    )
                    continue

            gt_clouds[aid] = pc

        except Exception as e:
            print(f"[GT] Error loading GT for {aid}: {e}")
            continue

    return gt_clouds


# ========================= Generation + Extraction =========================

def run_generation_and_extract(
    asset_ids: List[str],
    gt_clouds: Dict[str, PointCloud],
    cond_images: Dict[str, Image.Image],
    captions_by_sha: Dict[str, List[str]],
    pipelines: Dict[str, Any],
    n_views: int = 20,
    n_points: int = 4096,
    device: str = "cuda",
    seed_base: int = 42,
    run_image: bool = True,
    run_text: bool = True,
    run_joint: bool = True,
) -> Dict[str, Any]:
    """
    Run 3D generation and extract point clouds for all modalities on this rank's shard.

    Returns:
        Payload dict with:
            - "gen_clouds": { modality -> [(asset_id, PointCloud), ...] }
            - "used_captions_text": [caption_used_for_text2_3d, ...]
            - "used_captions_joint": [caption_used_for_blip_joint, ...]
            - "num_assets": len(asset_ids)
    """
    results: Dict[str, List[Tuple[str, PointCloud]]] = {
        "image_to_3d": [],
        "text_to_3d": [],
        "joint": [],
    }

    used_captions_text: List[str] = []
    used_captions_joint: List[str] = []

    img_pipe = pipelines.get("image")
    txt_pipe = pipelines.get("text")
    joint_pipe = pipelines.get("joint")
    blip = pipelines.get("blip")

    print(f"\n[rank shard] Generating and extracting point clouds for {len(asset_ids)} assets...")

    for idx, asset_id in enumerate(tqdm(asset_ids, desc="Generating shard")):
        if asset_id not in gt_clouds:
            print(f"[{asset_id}] Skipping: no GT point cloud loaded")
            continue

        cond_img = cond_images.get(asset_id)
        caps = captions_by_sha.get(asset_id)
        if not caps:
            print(f"[{asset_id}] No captions found in captions_by_sha")
            continue

        # --- Match seed logic from the other script (per-shard index) ---
        img_seed = (seed_base * 1009 + idx) & 0x7FFFFFFF
        txt_seed = (seed_base * 2003 + idx) & 0x7FFFFFFF

        # Text caption index: identical logic
        cap_rng = random.Random(txt_seed)
        cap_idx = cap_rng.randint(0, len(caps) - 1)
        chosen_caption = caps[cap_idx]

        # ============ Image→3D (single deterministic cond image per asset) ============
                # ============ Image→3D (single deterministic cond image per asset) ============
        if img_pipe is not None and cond_img is not None and run_image:
            try:
                this_seed = img_seed
                print(f"[{asset_id}] Running Image→3D with seed={this_seed}")
                out_img = img_pipe.run(
                    image=cond_img,
                    num_samples=1,
                    seed=this_seed,
                    formats=["gaussian", "mesh"],
                    preprocess_image=True,
                )
                gen_obj_gaussian = out_img["gaussian"][0]
                gen_obj_mesh = out_img["mesh"][0]


                # --- Extract point cloud ---
                pc = extract_pointcloud_from_generated(
                    gen_obj_gaussian, gen_obj_mesh, n_views, n_points
                )
                if pc is None:
                    print(
                        f"[{asset_id}] WARNING: Image→3D point cloud extraction returned None"
                    )
                else:
                    results["image_to_3d"].append((asset_id, pc))
            except Exception as e:
                print(f"[{asset_id}] ERROR in Image→3D: {e}")
                traceback.print_exc()


        # ============ Text→3D (single deterministic caption per asset) ============
        if txt_pipe is not None and run_text:
            try:
                print(f"[{asset_id}] Running Text→3D with caption[{cap_idx}]: {chosen_caption!r}")
                out_txt = txt_pipe.run(
                    prompt=chosen_caption,
                    num_samples=1,
                    seed=txt_seed,
                    formats=["gaussian", "mesh"],
                )
                print(f"[{asset_id}] Text→3D out keys: {list(out_txt.keys())}")
                gen_obj_gaussian = out_txt["gaussian"][0]
                
                

                gen_obj_mesh = out_txt["mesh"][0]
                pc = extract_pointcloud_from_generated(
                    gen_obj_gaussian, gen_obj_mesh, n_views, n_points
                )
                if pc is None:
                    print(
                        f"[{asset_id}] WARNING: Text→3D point cloud extraction returned None"
                    )
                else:
                    results["text_to_3d"].append((asset_id, pc))
                    used_captions_text.append(chosen_caption)
            except Exception as e:
                print(f"[{asset_id}] ERROR in Text→3D: {e}")
                traceback.print_exc()

        # ============ BLIP Joint (single deterministic caption per asset) ============
        if joint_pipe is not None and blip is not None and run_joint:
            try:
                # Match pattern from other script: this_seed depends on img_seed and caption index
                this_seed = (img_seed + cap_idx) & 0x7FFFFFFF
                image_embeds = blip.get_image_embeds(chosen_caption, steps=50)

                print(f"[{asset_id}] Running BLIP Joint with same caption[{cap_idx}]")
                out_joint = joint_pipe.run(
                    seed=this_seed,
                    image_embeds=image_embeds,
                    formats=["gaussian", "mesh"],
                )
                print(f"[{asset_id}] Joint out keys: {list(out_joint.keys())}")
                gen_obj_gaussian = out_joint["gaussian"][0]
                gen_obj_mesh = out_joint["mesh"][0]
                pc = extract_pointcloud_from_generated(
                    gen_obj_gaussian, gen_obj_mesh, n_views, n_points
                )
                if pc is None:
                    print(
                        f"[{asset_id}] WARNING: BLIP Joint point cloud extraction returned None"
                    )
                else:
                    results["joint"].append((asset_id, pc))
                    used_captions_joint.append(chosen_caption)
            except Exception as e:
                print(f"[{asset_id}] ERROR in BLIP Joint: {e}")
                traceback.print_exc()

    print("\n[rank shard] Generation summary:")
    for k, v in results.items():
        print(f"  {k}: {len(v)} generated point clouds")

    payload: Dict[str, Any] = {
        "gen_clouds": results,
        "used_captions_text": used_captions_text,
        "used_captions_joint": used_captions_joint,
        "num_assets": len(asset_ids),
    }
    return payload


# ========================= Metric Wrapper =========================

def compute_metrics_for_modality(
    ref_clouds: List[PointCloud],
    gen_clouds: List[PointCloud],
    modality_name: str,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Compute P-FID and P-IS for one modality."""
    if not gen_clouds:
        return {
            "error": f"No generated point clouds for {modality_name}",
            "num_generated": 0,
        }

    print(f"\n{'=' * 60}")
    print(f"Computing metrics for: {modality_name}")
    print(f"  Reference clouds: {len(ref_clouds)}")
    print(f"  Generated clouds: {len(gen_clouds)}")
    print(f"{'=' * 60}")

    try:
        metrics = compute_p_fid_p_is(
            ref_point_clouds=ref_clouds,
            gen_point_clouds=gen_clouds,
            device=device,
            batch_size=64,
        )

        print(f"Results for {modality_name}:")
        p_fid = metrics.get("p_fid")
        p_is = metrics.get("p_is")

        if p_fid is not None:
            print(f"  P-FID: {p_fid:.4f}")
        else:
            print("  P-FID: N/A")

        if p_is is not None:
            print(f"  P-IS:  {p_is:.4f}")
        else:
            print("  P-IS:  N/A")

        return metrics
    except Exception as e:
        print(f"Error computing metrics for {modality_name}: {e}")
        traceback.print_exc()
        return {"error": str(e)}


# ========================= Main Script =========================

def main():
    parser = argparse.ArgumentParser(
        description="Distributed 3D generation evaluation with point cloud metrics"
    )
    parser.add_argument(
        "--gt_renders_root",
        type=str,
        required=True,
        help="Root directory with ground truth Blender renders",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory to save results",
    )
    parser.add_argument(
        "--num_assets",
        type=int,
        default=1250,
        help="Number of assets to evaluate",
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=4096,
        help="Number of points per point cloud (FPS sampling)",
    )
    parser.add_argument(
        "--n_views",
        type=int,
        default=20,
        help="Number of views for depth unprojection (Point-E uses 20)",
    )
    parser.add_argument(
        "--seed_base",
        type=int,
        default=42,
        help="Random seed base (used for split & caption selection)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu) – overridden per-rank",
    )
    parser.add_argument(
        "--save_clouds",
        action="store_true",
        help="Save extracted point clouds as PLY files",
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        default="",
        help="CSV file with captions",
    )
    parser.add_argument(
        "--image_ckpt",
        type=str,
        default="microsoft/TRELLIS-image-large",
        help="Image→3D checkpoint",
    )
    parser.add_argument(
        "--text_ckpt",
        type=str,
        default="microsoft/TRELLIS-text-xlarge",
        help="Text→3D checkpoint",
    )
    parser.add_argument(
        "--blip_ckpt",
        type=str,
        default="",
        help="BLIP checkpoint for joint generation",
    )
    parser.add_argument(
        "--dist_backend",
        type=str,
        default="nccl",
        help="Distributed backend (nccl|gloo)",
    )
    parser.add_argument(
        "--selected_ids",
        type=str,
        default="",
        help="comma-separated sha256s (optional filter)",
    )
    parser.add_argument(
        "--selected_ids_file",
        type=str,
        default="",
        help="file with one sha256 per line (optional filter)",
    )
    parser.add_argument(
        "--asset_ids_file",
        type=str,
        default="",
        help=(
            "Optional: path to a file with one asset_id per line; "
            "if provided, overrides internal split logic to match another script exactly."
        ),
    )
    parser.add_argument("--skip_image_to_3d", action="store_true",
                        help="Skip image→3D pipeline + metrics")
    parser.add_argument("--skip_text_to_3d", action="store_true",
                        help="Skip text→3D pipeline + metrics")
    parser.add_argument("--skip_blip_joint", action="store_true",
                        help="Skip BLIP joint image→3D pipeline + metrics")

    args = parser.parse_args()

    run_image = not args.skip_image_to_3d
    run_text  = not args.skip_text_to_3d
    run_joint = not args.skip_blip_joint

    # ========== Distributed setup ==========
    dist_setup(backend=args.dist_backend)
    rank = get_rank()
    world = get_world_size()
    is_main = rank == 0

    # Choose device per rank
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = "cuda"
    else:
        device = "cpu"
    args.device = device

    # Seeds
    random.seed(args.seed_base)
    np.random.seed(args.seed_base)
    torch.manual_seed(args.seed_base + rank)

    if is_main:
        os.makedirs(args.results_dir, exist_ok=True)

    # Load captions globally
    captions_by_sha: Dict[str, List[str]] = {}
    if args.metadata_csv:
        captions_by_sha = load_captions_by_sha(args.metadata_csv)
    if is_main:
        print(f"[info] Captions available for {len(captions_by_sha)} assets")

    root = Path(args.gt_renders_root)

    # ========= Step 1: build SAME global asset split on all ranks =========

    if args.asset_ids_file:
        # Use a precomputed split (e.g., from your other script)
        with open(args.asset_ids_file) as f:
            asset_ids = [line.strip() for line in f if line.strip()]
        if is_main:
            print(f"[info] Loaded {len(asset_ids)} asset IDs from {args.asset_ids_file}")
    else:
        if is_main:
            print(f"[info] Scanning asset IDs (no image loads): {args.gt_renders_root}")
        all_ids = list_asset_ids(args.gt_renders_root)

        wanted = set()
        if args.selected_ids_file and os.path.isfile(args.selected_ids_file):
            with open(args.selected_ids_file) as f:
                wanted = {line.strip() for line in f if line.strip()}
        elif args.selected_ids:
            wanted = {x.strip() for x in args.selected_ids.split(",") if x.strip()}

        candidate_ids = [aid for aid in all_ids if not wanted or aid in wanted]

        rng = random.Random(args.seed_base)
        rng.shuffle(candidate_ids)

        # For this point-cloud eval, we require captions (text / joint)
        captioned_ids = [aid for aid in candidate_ids if aid in captions_by_sha]
        if len(captioned_ids) < args.num_assets and is_main:
            print(
                f"[warn] Only {len(captioned_ids)} assets have captions "
                f"(requested {args.num_assets}). Using all captioned assets."
            )

        asset_ids = captioned_ids[: min(args.num_assets, len(captioned_ids))]

    if is_main:
        print(f"[info] Global split size: {len(asset_ids)} assets across world_size={world}")
    barrier()

    # ========= Step 2: load GT (global set) on each rank =========
    # gt_clouds = load_gt_pointclouds(
    #     args.gt_renders_root,
    #     asset_ids,
    #     args.n_points,
    # )
    # if not gt_clouds:
    #     print("ERROR: No ground truth point clouds loaded for sampled assets!")
    #     return

    # valid_asset_ids = list(gt_clouds.keys())
    # if is_main:
    #     print(f"Proceeding with {len(valid_asset_ids)} assets that have GT point clouds")

    # gt_cloud_list = [gt_clouds[aid] for aid in valid_asset_ids]
    gt_clouds = load_gt_pointclouds(
        args.gt_renders_root,
        asset_ids,
        args.n_points,
    )

    # Extra safety: filter out any remaining invalid GT clouds
    gt_clouds = {
        aid: pc
        for aid, pc in gt_clouds.items()
        if pc is not None
        and pc.coords is not None
        and pc.coords.ndim == 2
        and pc.coords.shape[0] > 0
    }

    if not gt_clouds:
        print("ERROR: No valid ground truth point clouds loaded for sampled assets!")
        return

    valid_asset_ids = list(gt_clouds.keys())
    if is_main:
        print(f"Proceeding with {len(valid_asset_ids)} assets that have valid GT point clouds")

    gt_cloud_list = [gt_clouds[aid] for aid in valid_asset_ids]


    metrics_self = compute_p_fid_p_is(
        ref_point_clouds=gt_cloud_list,
        gen_point_clouds=gt_cloud_list,
        device="cuda",       # or your device
        batch_size=64,
    )

    print("P-FID(GT, GT) =", metrics_self["p_fid"])
    print("P-IS(GT)       =", metrics_self["p_is"])

    # GT stats only on rank 0
    if is_main:
        gt_stats = visualize_point_cloud_stats(gt_cloud_list)
        print("\nGround Truth Statistics:")
        print(json.dumps(gt_stats, indent=2))
    else:
        gt_stats = {}

    # Optionally save some GT clouds (rank 0 only)
    if is_main and args.save_clouds:
        clouds_dir = Path(args.results_dir) / "point_clouds" / "gt"
        clouds_dir.mkdir(parents=True, exist_ok=True)
        for i, (aid, pc) in enumerate(list(gt_clouds.items())[:10]):
            save_point_cloud_ply(pc, str(clouds_dir / f"{aid}.ply"))
        print(f"Saved sample GT clouds to {clouds_dir}")

    barrier()

    # ========= Step 3: load pipelines per rank =========
    if is_main:
        print("\n" + "=" * 60)
        print("Loading generation pipelines...")
        print("=" * 60)

    img_pipe = None
    if args.image_ckpt and run_image:
        img_pipe = TrellisImageTo3DPipeline.from_pretrained(args.image_ckpt)
        img_pipe.to(args.device)
        if is_main:
            print(f"✓ Loaded Image→3D pipeline: {args.image_ckpt}")

    txt_pipe = None
    if args.text_ckpt and run_text:
        txt_pipe = TrellisTextTo3DPipeline.from_pretrained(args.text_ckpt)
        txt_pipe.to(args.device)
        if is_main:
            print(f"✓ Loaded Text→3D pipeline: {args.text_ckpt}")

    blip = None
    joint_pipe = None
    if args.blip_ckpt and run_joint:
        joint_pipe = BlipTrellisImageTo3DPipeline.from_pretrained(args.image_ckpt)
        joint_pipe.to(args.device)
        model_args = ModelArguments(model_name_or_path=args.blip_ckpt)
        data_args = DataArguments()
        blip = BlipTextEmbedder(ckpt=args.blip_ckpt, device=args.device, model_args=model_args, data_args=data_args)
        if is_main:
            print(f"✓ Loaded BLIP joint pipeline: {args.blip_ckpt}")

    barrier()

    # ========= Step 4: shard valid_asset_ids across ranks =========
    local_asset_ids = shard_list(valid_asset_ids, rank, world)
    print(f"[rank {rank}] local shard size: {len(local_asset_ids)}")

    # Load conditioning images for local shard using FDKD + deterministic choice
    cond_images: Dict[str, Image.Image] = {}

    for local_idx, aid in enumerate(local_asset_ids):
        asset_dir = root / aid
        pngs = list(asset_dir.glob("*.png"))
        if not pngs:
            continue
        pngs = [p for p in pngs if not p.stem.endswith("_depth")]

        # Sort like in general eval
        pngs = sorted(pngs, key=lambda p: int(p.stem))
       

        n = len(pngs)
        if n < 1:
            continue

        # FDKD subset (up to 4 canonical views)
        if n >= args.n_views:
            idxs = list(range(args.n_views))
        else:
            idxs = select_even_indices(n, args.n_views)

        fd_imgs_paths = [pngs[i] for i in idxs if i < n]
        if not fd_imgs_paths:
            continue

        # Deterministic conditioning-view choice, mirroring general eval seed pattern
        img_seed = (args.seed_base * 1009 + local_idx) & 0x7FFFFFFF
        rng = random.Random(img_seed)
        sampled_j = rng.randint(0, len(fd_imgs_paths) - 1)

        cond_path = fd_imgs_paths[sampled_j]
        # cond_images[aid] = Image.open(cond_path).convert("RGB")
        raw = Image.open(cond_path)

        
        cond_images[aid] = raw


    pipelines = {
        "image": img_pipe,
        "text": txt_pipe,
        "joint": joint_pipe,
        "blip": blip,
    }

    # ========= Step 5: run generation & extraction per rank =========
    shard_payload = run_generation_and_extract(
        asset_ids=local_asset_ids,
        gt_clouds=gt_clouds,
        cond_images=cond_images,
        captions_by_sha=captions_by_sha,
        pipelines=pipelines,
        n_views=args.n_views,
        n_points=args.n_points,
        device=args.device,
        seed_base=args.seed_base,
        run_image=run_image,
        run_text=run_text,
        run_joint=run_joint,
    )

    # ========= Step 6: gather payloads to rank 0 =========
    gathered = gather_objects_to_rank0(shard_payload)
    barrier()

    if not is_main:
        return

    # ========= Step 7: aggregate generated clouds & captions on rank 0 =========
    aggregated_gen_clouds: Dict[str, List[Tuple[str, PointCloud]]] = {
        "image_to_3d": [],
        "text_to_3d": [],
        "joint": [],
    }
    used_captions_text_all: List[str] = []
    used_captions_joint_all: List[str] = []

    total_assets = 0

    for pay in gathered:
        if not pay:
            continue
        total_assets += pay.get("num_assets", 0)

        gc = pay.get("gen_clouds", {})
        for key in aggregated_gen_clouds.keys():
            aggregated_gen_clouds[key].extend(gc.get(key, []))

        used_captions_text_all.extend(pay.get("used_captions_text", []))
        used_captions_joint_all.extend(pay.get("used_captions_joint", []))

    if total_assets == 0:
        print("No assets processed; exiting.")
        return

    # # Save generated cloud statistics
    # for modality, asset_pc_pairs in aggregated_gen_clouds.items():
    #     if asset_pc_pairs:
    #         pcs = [pc for _, pc in asset_pc_pairs]
    #         stats = visualize_point_cloud_stats(pcs)
    #         print(f"\n{modality.upper()} Statistics (samples={len(pcs)}):")
    #         print(json.dumps(stats, indent=2))
    #         if args.save_clouds:
    #             clouds_dir = Path(args.results_dir) / "point_clouds" / modality
    #             clouds_dir.mkdir(parents=True, exist_ok=True)
    #             for i, pc in enumerate(pcs[:10]):
    #                 save_point_cloud_ply(pc, str(clouds_dir / f"sample_{i:04d}.ply"))

    # ========= Step 8: compute metrics for each modality on rank 0 =========
    print("\n" + "=" * 60)
    print("COMPUTING POINT CLOUD METRICS (P-FID and P-IS)")
    print("=" * 60)

    all_results: Dict[str, Any] = {
        "settings": {
            "gt_renders_root": args.gt_renders_root,
            "num_assets_requested": len(asset_ids),
            "num_assets_with_gt": len(valid_asset_ids),
            "n_points": args.n_points,
            "seed_base": args.seed_base,
            "image_ckpt": args.image_ckpt,
            "text_ckpt": args.text_ckpt,
            "blip_ckpt": args.blip_ckpt,
            "world_size": world,
        },
        "gt_statistics": gt_stats,
        "results": {},
    }

    def compute_for_modality(mod_key: str, name: str, result_key: str) -> None:
        pairs = aggregated_gen_clouds.get(mod_key, [])
        print(f"\n[metrics] {mod_key}: {len(pairs)} generated pairs")
        if not pairs:
            all_results["results"][result_key] = {
                "error": f"No generated samples for modality {name}",
                "num_pairs": 0,
            }
            return

        # Align generated to GT assets (single sample per asset per modality)
        gen_by_asset: Dict[str, List[PointCloud]] = {}
        for aid, pc in pairs:
            if aid not in gt_clouds:
                print(f"[metrics] skipping {aid} (no GT)")
                continue
            if aid in gen_by_asset:
                # Only keep the first sample for each asset
                continue
            gen_by_asset.setdefault(aid, []).append(pc)

        # Flatten keeping first sample per asset, preserving deterministic ordering
        gen_list: List[PointCloud] = []
        ref_list: List[PointCloud] = []
        for aid in valid_asset_ids:  # deterministic order
            if aid in gen_by_asset:
                pc = gen_by_asset[aid][0]
                gen_list.append(pc)
                ref_list.append(gt_clouds[aid])

        print(f"[metrics] {mod_key}: aligned {len(gen_list)} samples with GT")

        if not gen_list:
            all_results["results"][result_key] = {
                "error": f"No overlapping assets between GT and generated for modality {name}",
                "num_pairs": len(pairs),
            }
            return

        metrics = compute_metrics_for_modality(ref_list, gen_list, name, args.device)
        all_results["results"][result_key] = metrics

    compute_for_modality("image_to_3d", "Image→3D", "image_to_3d")
    compute_for_modality("text_to_3d", "Text→3D", "text_to_3d")
    compute_for_modality("joint", "BLIP Joint", "blip_joint")

    # ========= Step 9: save metrics JSON =========
    output_file = Path(args.results_dir) / "pointcloud_metrics.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"✓ Results saved to: {output_file}")
    print(f"{'=' * 60}")
    print("\nFinal Summary:")
    print(json.dumps(all_results, indent=2))

    # ========= Step 10: save used captions (similar to other script) =========
    caps_txt_path = os.path.join(args.results_dir, "used_captions_pointcloud.txt")
    with open(caps_txt_path, "w", encoding="utf-8") as f:
        f.write("# text_to_3d captions\n")
        for cap in used_captions_text_all:
            cap_clean = str(cap).replace("\n", " ").strip()
            if cap_clean:
                f.write(cap_clean + "\n")

        f.write("\n# image_to_3d_blip_joint captions\n")
        for cap in used_captions_joint_all:
            cap_clean = str(cap).replace("\n", " ").strip()
            if cap_clean:
                f.write(cap_clean + "\n")

    print("Saved used captions to:", caps_txt_path)

    # ================= SAVE POINT CLOUDS =======================
    print("\nSaving all point clouds for offline visualization...")

    pc_root = Path(args.results_dir) / "saved_pointclouds"
    (pc_root / "gt").mkdir(parents=True, exist_ok=True)

    # ---- Save GT clouds ----
    for aid, pc in gt_clouds.items():
        out_path = pc_root / "gt" / f"{aid}.ply"
        save_point_cloud_ply(pc, str(out_path))

    # ---- Save generated clouds for each modality ----
    for modality, asset_pc_pairs in aggregated_gen_clouds.items():
        mod_dir = pc_root / modality
        mod_dir.mkdir(parents=True, exist_ok=True)

        for aid, pc in asset_pc_pairs:
            out_path = mod_dir / f"{aid}.ply"
            save_point_cloud_ply(pc, str(out_path))

    # ---- Save index of which assets were used ----
    with open(pc_root / "asset_ids.txt", "w") as f:
        for aid in valid_asset_ids:
            f.write(aid + "\n")

    print(f"✓ Saved GT + generated point clouds to: {pc_root}")



if __name__ == "__main__":
    main()
    for modality, asset_pc_pairs in aggregated_gen_clouds.items():
        mod_dir = pc_root / modality
        mod_dir.mkdir(parents=True, exist_ok=True)

        for aid, pc in asset_pc_pairs:
            out_path = mod_dir / f"{aid}.ply"
            save_point_cloud_ply(pc, str(out_path))

    # ---- Save index of which assets were used ----
    with open(pc_root / "asset_ids.txt", "w") as f:
        for aid in valid_asset_ids:
            f.write(aid + "\n")

    print(f"✓ Saved GT + generated point clouds to: {pc_root}")



if __name__ == "__main__":
    main()
