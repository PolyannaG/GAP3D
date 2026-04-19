import os, sys, json, math, random, glob, traceback, ast, argparse
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
from PIL import Image
import pickle
import io
import warnings
from pathlib import Path
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.transforms.functional import InterpolationMode
from transformers.image_transforms import convert_to_rgb
import open_clip

# Safer default on clusters
os.environ["SPCONV_ALGO"] = "native"

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline, TrellisTextTo3DPipeline, BlipTrellisImageTo3DPipeline
from TRELLIS.trellis.utils.render_utils import (
    yaw_pitch_r_fov_to_extrinsics_intrinsics,
    render_frames,
    render_snapshot,
)

from dist_helpers import (
    dist_is_initialized,
    dist_setup,
    get_rank,
    get_world_size,
    barrier,
    shard_list,
    gather_objects_to_rank0,
)

from utils import (
    select_even_indices,
    frechet_distance,
    kid_mmd2_poly_degree3,
    load_captions_by_sha,
    find_assets,
    list_asset_ids,
    load_assets_for_ids,

)

from model_helpers import BlipTextEmbedder, ModelArguments, DataArguments


# =========================== Defaults ===========================
IMAGE_PRETRAINED = "microsoft/TRELLIS-image-large"
TEXT_PRETRAINED  = "microsoft/TRELLIS-text-xlarge"
SS_FLOW_CKPT    = None  # optional finetuned sparse structure flow checkpoint
SLAT_FLOW_CKPT  = None  # optional finetuned structured latent flow checkpoint

FDKD_VIEWS   = 4           # yaw 0/90/180/270
FDKD_PITCH   = 30.0
CLIP_VIEWS   = 8           # yaw every 45°
CLIP_PITCH   = 30.0
RADIUS       = 2.0
FOV_DEG      = 40.0

GEN_RES      = 512
BG_COLOR     = (0, 0, 0)
SSAA         = 4
STRICT_COLOR_ONLY = True

CLIP_MODEL_NAME   = "ViT-L-14"
CLIP_PRETRAINED   = "openai"
DINO_VARIANT_HUB  = "dinov2_vitl14"
DINO_FEAT_DIM     = 1024
DINO_INPUT_SIZE   = 518



# =================== Render helpers (Trellis) =================
def render_generated_four_views_with_channel(sample_obj):
    rets = render_snapshot(
        samples=sample_obj,
        resolution=GEN_RES,
        bg_color=BG_COLOR,
        offset=(0.0, math.radians(FDKD_PITCH)),
        r=RADIUS,
        fov=FOV_DEG,
        ssaa=SSAA,
    )
    if 'color' in rets:
        return [Image.fromarray(c) for c in rets['color']], 'color'
    else:
        print("Warning: no 'color' in render results, falling back to 'normal'", file=sys.stderr)
    if 'normal' in rets:
        return [Image.fromarray(c) for c in rets['normal']], 'normal'
    return [], 'none'

def render_generated_eight_views(sample_obj):
    yaws = [i * (2*math.pi/CLIP_VIEWS) for i in range(CLIP_VIEWS)]
    pitchs = [math.radians(CLIP_PITCH)] * CLIP_VIEWS
    extr, intr = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs=RADIUS, fovs=FOV_DEG)
    rets = render_frames(sample_obj, extr, intr, {'resolution': GEN_RES, 'bg_color': BG_COLOR, 'ssaa': SSAA})
    key = 'color' if 'color' in rets else 'normal'
    return [Image.fromarray(c) for c in rets[key]], key

# =========================== Inception-DINOv2 Features =========================
class InceptionPool3(nn.Module):
    def __init__(self, device):
        super().__init__()
        weights = models.Inception_V3_Weights.IMAGENET1K_V1
        m = models.inception_v3(weights=weights, aux_logits=True, transform_input=False)
        m.fc = nn.Identity()
        self.model = m.eval().to(device)
        self.pre   = weights.transforms()
    @torch.no_grad()
    def __call__(self, img_list, device):
        x = torch.stack([self.pre(img) for img in img_list]).to(device)
        return self.model(x)

class DINOv2Feat:
    def __init__(self, device):
        self.device = device
        self.model = None
        self.pre = transforms.Compose([
            convert_to_rgb,
            transforms.Resize(DINO_INPUT_SIZE, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(DINO_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
    
    def _ensure(self):
        if self.model is not None:
            return

        def _load():
            return (torch.hub.load('facebookresearch/dinov2', DINO_VARIANT_HUB, pretrained=True)
                    .to(self.device).eval())

        # If distributed, have rank 0 populate the hub cache first.
        if 'dist_is_initialized' in globals() and dist_is_initialized():
            if get_rank() == 0:
                self.model = _load()
            # Make sure cache is fully written before others touch it
            barrier()
            if self.model is None:  # non-rank0 will enter here
                self.model = _load()
        else:
            # single-process case
            self.model = _load()

    @torch.no_grad()
    def __call__(self, img_list):
        self._ensure()
        x = torch.stack([self.pre(img) for img in img_list]).to(self.device)
        out = self.model.forward_features(x)
        # feats = out["x_prenorm"][:,0]
        feats = out["x_norm_clstoken"]
        assert feats.shape[1] == DINO_FEAT_DIM
        return feats

# =========================== CLIP =============================

@torch.no_grad()
def clip_image_to_image_perview(cond_imgs_per_asset, gen_views_per_asset, device):
    assert open_clip is not None, "open_clip_torch not installed"
    model, _, pre = open_clip.create_model_and_transforms(CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=device)
    model.eval()
    sims=[]
    for cond_views, gen_views in zip(cond_imgs_per_asset, gen_views_per_asset):
        if not cond_views or not gen_views: 
            continue

        c = torch.stack([pre(v) for v in cond_views]).to(device)
        g = torch.stack([pre(v) for v in gen_views]).to(device)

        cfeat = F.normalize(model.encode_image(c), dim=-1)
        assert cfeat.shape[0] == 1, "expect 1 cond image per asset"

        gfeat = F.normalize(model.encode_image(g), dim=-1)
        assert gfeat.shape[0] == CLIP_VIEWS, f"expect {CLIP_VIEWS} gen views per asset"

        sims.append((gfeat @ cfeat.T).mean().item())

    return float(np.mean(sims) * 100.0) if sims else None

@torch.no_grad()
def clip_text_to_image_perview(list_of_caption_lists, gen_views_per_asset, device):
    assert open_clip is not None, "open_clip_torch not installed"
    model, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=device)
    model.eval()
    tok = open_clip.get_tokenizer("ViT-B-32")
    sims=[]
    for caps, gv in zip(list_of_caption_lists, gen_views_per_asset):
        assert len(caps) >= 1, f"No captions found for asset!"
        if not caps or not gv: 
            continue

        ims = torch.stack([pre(v) for v in gv]).to(device)
        gfeat = F.normalize(model.encode_image(ims), dim=-1)
        assert gfeat.shape[0] == CLIP_VIEWS, f"expect {CLIP_VIEWS} gen views per asset"

        text = tok(caps).to(device)
        tfeat = F.normalize(model.encode_text(text), dim=-1)

        sims.append((gfeat @ tfeat.T).mean().item())

    return float(np.mean(sims) * 100.0) if sims else None


# encapsulate metrics math
def _compute_metrics_from_features(
    X_incep, X_dino,
    Y_img_incep, Y_img_dino, kept_img, skipped_img,
    Y_txt_incep, Y_txt_dino, kept_txt, skipped_txt,
    Y_joint_incep, Y_joint_dino, kept_joint, skipped_joint,
    clip_payload, device
):
    base_settings = {
        "fd_kd_views": {"num": FDKD_VIEWS, "pitch_deg": FDKD_PITCH},
        "clip_views":  {"num": CLIP_VIEWS, "pitch_deg": CLIP_PITCH},
        "radius": RADIUS, "fov_deg": FOV_DEG, "gen_res": GEN_RES, "ssaa": SSAA,
        "image_ckpt": IMAGE_PRETRAINED, "text_ckpt": TEXT_PRETRAINED,
        "strict_color_only": STRICT_COLOR_ONLY,
        "kd_definition": "Unbiased MMD^2 with polynomial kernel degree 3; report KD=(MMD^2 * 100).",
        "clip_alignment": "Average per-view cosine.",
    }
    results = {"settings": base_settings}

    # image→3D
    if Y_img_incep:
        Yi = np.concatenate(Y_img_incep, axis=0)
        Yd = np.concatenate(Y_img_dino,  axis=0)
        results["image_to_3d"] = {
            "fd_inception": float(frechet_distance(X_incep, Yi)),
            "kd_inception": float(kid_mmd2_poly_degree3(X_incep, Yi) * 100.0),
            "fd_dinov2":    float(frechet_distance(X_dino,  Yd)),
            "kd_dinov2":    float(kid_mmd2_poly_degree3(X_dino,  Yd) * 100.0),
            "num_eval_images": int(len(Yi)),
            "num_assets_kept": int(len(Y_img_incep)),
            "kept_color": kept_img, "skipped_noncolor": skipped_img,
        }
    else:
        results["image_to_3d"] = {"num_eval_images": 0, "note": "no run or no successful generations (color)"}

    # text→3D
    if Y_txt_incep:
        Yi = np.concatenate(Y_txt_incep, axis=0)
        Yd = np.concatenate(Y_txt_dino,  axis=0)
        results["text_to_3d"] = {
            "fd_inception": float(frechet_distance(X_incep, Yi)),
            "kd_inception": float(kid_mmd2_poly_degree3(X_incep, Yi) * 100.0),
            "fd_dinov2":    float(frechet_distance(X_dino,  Yd)),
            "kd_dinov2":    float(kid_mmd2_poly_degree3(X_dino,  Yd) * 100.0),
            "num_eval_images": int(len(Yi)),
            "num_assets_kept": int(len(Y_txt_incep)),
            "kept_color": kept_txt, "skipped_noncolor": skipped_txt,
        }
    else:
        results["text_to_3d"] = {"num_eval_images": 0, "note": "no run or no successful generations (color)"}

    # image→3D (BLIP joint)
    if Y_joint_incep:
        Yi = np.concatenate(Y_joint_incep, axis=0)
        Yd = np.concatenate(Y_joint_dino,  axis=0)
        results["image_to_3d_blip_joint"] = {
            "fd_inception": float(frechet_distance(X_incep, Yi)),
            "kd_inception": float(kid_mmd2_poly_degree3(X_incep, Yi) * 100.0),
            "fd_dinov2":    float(frechet_distance(X_dino,  Yd)),
            "kd_dinov2":    float(kid_mmd2_poly_degree3(X_dino,  Yd) * 100.0),
            "num_eval_images": int(len(Yi)),
            "num_assets_kept": int(len(Y_joint_incep)),
            "kept_color": kept_joint, "skipped_noncolor": skipped_joint,
        }
    else:
        results["image_to_3d_blip_joint"] = {"note": "not run or no successful joint generations"}

    # CLIP
    if clip_payload and any(clip_payload.values()):
        results["clip"] = clip_payload
    else:
        results["clip"] = {"note": "open_clip_torch not installed or no views"}

    # Self-FD sanity
    try:
        perm = np.random.permutation(len(X_incep))
        results["sanity"] = {
            "fd_self_inception": float(frechet_distance(X_incep, X_incep[perm])),
            "fd_self_dino":      float(frechet_distance(X_dino,  X_dino[np.random.permutation(len(X_dino))])),
        }
    except Exception:
        print("Warning: failed to compute self-FD sanity metrics", file=sys.stderr)
        pass

    return results


# ============================= Core Eval =============================

def render_and_collect(sample_obj, incep, dino, device,
                       cond_img_for_clip=None,
                       captions_for_clip=None,
                       accum_img=None, accum_dino=None,
                       accum_clip_views=None, accum_cond_imgs=None,
                       accum_clip_txt_views=None, accum_txt_caps=None,
                       kept_counter=None, skipped_counter=None):
    gen4, ch = render_generated_four_views_with_channel(sample_obj)
    if STRICT_COLOR_ONLY and ch != 'color':
        if skipped_counter is not None: 
            skipped_counter[0] += 1
        return
    accum_img.append(incep(gen4, device=device).cpu().numpy() )
    accum_dino.append( dino(gen4).cpu().numpy() )
    if kept_counter is not None: 
        kept_counter[0] += 1
    
    clip8, ch8 = render_generated_eight_views(sample_obj)
    if ch8 == 'color':
        if accum_clip_views is not None: 
            accum_clip_views.append(clip8)
        if accum_cond_imgs is not None and cond_img_for_clip is not None: 
            accum_cond_imgs.append([cond_img_for_clip])
        if accum_clip_txt_views is not None and captions_for_clip is not None: 
            accum_clip_txt_views.append(clip8)
        if accum_txt_caps is not None and captions_for_clip is not None: 
            accum_txt_caps.append(captions_for_clip[:])
    else:
        print("Warning: generated 8-view render not color, skipping CLIP accumulation", file=sys.stderr)

def evaluate_split_collect_features(
    asset_ids: List[str],
    assets: Dict[str, List[Image.Image]],
    caps_by_sha: Dict[str, List[str]],
    device: str,
    blip_ckpt: Optional[str],
    run_image: bool = True,
    run_text: bool = True,
    run_joint: bool = True,
):
    # Feature extractors
    incep = InceptionPool3(device)
    dino  = DINOv2Feat(device)

    # Reference features for THIS shard
    X_incep_list, X_dino_list = [], []
    fd_views_per_asset = {}  # asset_id -> list of conditioning views

    for aid in asset_ids:
        n = len(assets[aid])
        if n < 1:
            continue
        idxs = list(range(FDKD_VIEWS)) if n >= FDKD_VIEWS else select_even_indices(n, FDKD_VIEWS)
        fd_imgs = [assets[aid][i] for i in idxs]

        X_incep_list.append(incep(fd_imgs, device=device).cpu().numpy())
        X_dino_list.append(dino(fd_imgs).cpu().numpy())

        # keep ALL conditioning views for this asset
        fd_views_per_asset[aid] = fd_imgs

    if not X_incep_list:
        return None  # empty shard
    X_incep = np.concatenate(X_incep_list, axis=0)
    X_dino  = np.concatenate(X_dino_list,  axis=0)

    #    # Pipelines
    img_pipe = None
    txt_pipe = None
    blip = None
    joint_img_pipe = None

    is_dist = dist_is_initialized()
    rank = get_rank() if is_dist else 0

    # --- Image→3D pipeline (uses dinov2 via torch.hub.load) ---
    if run_image:
        if is_dist:
            # rank 0 populates the torch.hub cache first
            if rank == 0:
                # img_pipe = TrellisImageTo3DPipeline.from_pretrained(IMAGE_PRETRAINED, sparse_structure_flow_ckpt=SS_FLOW_CKPT, slat_flow_ckpt=SLAT_FLOW_CKPT)
                img_pipe = TrellisImageTo3DPipeline.from_pretrained(IMAGE_PRETRAINED, sparse_structure_flow_ckpt=SS_FLOW_CKPT)
                # img_pipe = TrellisImageTo3DPipeline.from_pretrained(IMAGE_PRETRAINED)
                img_pipe.to(device)
            barrier()
            # after cache is ready, other ranks can safely load
            if rank != 0:
                img_pipe = TrellisImageTo3DPipeline.from_pretrained(IMAGE_PRETRAINED, sparse_structure_flow_ckpt=SS_FLOW_CKPT)
                # img_pipe = TrellisImageTo3DPipeline.from_pretrained(IMAGE_PRETRAINED, sparse_structure_flow_ckpt=SS_FLOW_CKPT, slat_flow_ckpt=SLAT_FLOW_CKPT)
                # img_pipe = TrellisImageTo3DPipeline.from_pretrained(IMAGE_PRETRAINED)
                img_pipe.to(device)
        else:
            # single-process case
            img_pipe = TrellisImageTo3DPipeline.from_pretrained(IMAGE_PRETRAINED, sparse_structure_flow_ckpt=SS_FLOW_CKPT)
            # img_pipe = TrellisImageTo3DPipeline.from_pretrained(IMAGE_PRETRAINED, sparse_structure_flow_ckpt=SS_FLOW_CKPT, slat_flow_ckpt=SLAT_FLOW_CKPT)
            # img_pipe = TrellisImageTo3DPipeline.from_pretrained(IMAGE_PRETRAINED)
            img_pipe.to(device)

            

    # --- Text→3D pipeline (does NOT hit dinov2 hub, safe to load normally) ---
    if run_text:
        txt_pipe = TrellisTextTo3DPipeline.from_pretrained(TEXT_PRETRAINED)
        txt_pipe.to(device)

    # --- BLIP joint pipeline (also uses image backbone / dinov2) ---
    if run_joint:
        if is_dist:
            if rank == 0:
                joint_img_pipe = BlipTrellisImageTo3DPipeline.from_pretrained(IMAGE_PRETRAINED, sparse_structure_flow_ckpt=SS_FLOW_CKPT, slat_flow_ckpt=SLAT_FLOW_CKPT)
                # joint_img_pipe = BlipTrellisImageTo3DPipeline.from_pretrained(IMAGE_PRETRAINED, sparse_structure_flow_ckpt=SS_FLOW_CKPT)

                joint_img_pipe.to(device)
            barrier()
            if rank != 0:
                # joint_img_pipe = BlipTrellisImageTo3DPipeline.from_pretrained(IMAGE_PRETRAINED, sparse_structure_flow_ckpt=SS_FLOW_CKPT)
                joint_img_pipe = BlipTrellisImageTo3DPipeline.from_pretrained(IMAGE_PRETRAINED, sparse_structure_flow_ckpt=SS_FLOW_CKPT, slat_flow_ckpt=SLAT_FLOW_CKPT)
                joint_img_pipe.to(device)
        else:
            # joint_img_pipe = BlipTrellisImageTo3DPipeline.from_pretrained(IMAGE_PRETRAINED, sparse_structure_flow_ckpt=SS_FLOW_CKPT)
            joint_img_pipe = BlipTrellisImageTo3DPipeline.from_pretrained(IMAGE_PRETRAINED, sparse_structure_flow_ckpt=SS_FLOW_CKPT, slat_flow_ckpt=SLAT_FLOW_CKPT)
            joint_img_pipe.to(device)

        
        model_args = ModelArguments(model_name_or_path=blip_ckpt)
        data_args = DataArguments()

        blip = BlipTextEmbedder(ckpt=blip_ckpt, device=device, model_args=model_args, data_args=data_args)


    # Holders
    YI_img_list, YD_img_list   = [], []
    YI_txt_list, YD_txt_list   = [], []
    YI_joint_list, YD_joint_list = [], []

    gen_clip_views_img, cond_imgs_kept_img = [], []
    gen_clip_views_txt, txt_caps_kept = [], []
    gen_clip_views_joint, txt_caps_kept_joint = [], []

    kept_img = [0]
    skipped_img = [0]
    kept_txt = [0]
    skipped_txt = [0]
    kept_joint = [0]
    skipped_joint = [0]

    for idx, aid in enumerate(asset_ids):
        if idx % 100 == 0:
            print(f"Evaluating asset {idx+1}/{len(asset_ids)}: {aid}")
        try:
            # make seeds rank-stable per asset
            img_seed = (42 * 1009 + idx) & 0x7fffffff
            txt_seed = (42 * 2003 + idx) & 0x7fffffff

            fd_imgs = fd_views_per_asset.get(aid, [])
            caps = caps_by_sha.get(aid)

            # -------- image→3D: run once per conditioning view --------
            # Sample one of the conditioning views per generation
            if run_image:
                sampled_j = random.Random(img_seed).randint(0, len(fd_imgs)-1) if fd_imgs else 0
                for j, cond_img in enumerate(fd_imgs):
                    if j != sampled_j:
                        continue  # only one conditioning view per asset for image→3D
                    this_seed = (img_seed + j) & 0x7fffffff
                    out_img = img_pipe.run(
                        image=cond_img,
                        num_samples=1,
                        seed=this_seed,
                        formats=["gaussian"],
                        preprocess_image=True,
                    )
                    sample_obj_img = out_img.get("gaussian")[0]
                    if sample_obj_img is not None:
                        # each (asset, view) is its own sample, like (asset, caption)
                        render_and_collect(
                            sample_obj_img, incep, dino, device,
                            cond_img_for_clip=cond_img,
                            accum_img=YI_img_list, accum_dino=YD_img_list,
                            accum_clip_views=gen_clip_views_img,
                            accum_cond_imgs=cond_imgs_kept_img,
                            kept_counter=kept_img, skipped_counter=skipped_img,
                        )
            if caps is None:
                print(f"[{aid}] warning: no captions found!")
                continue

            # ---------------- text→3D: run for ALL captions ----------------
            # Sample one of the captions per generation
            sampled_j = random.Random(txt_seed).randint(0, len(caps)-1) if caps else 0
            # sampled_j = 0  # always use first caption for text→3D  (longest)
            # sampled_j = len(caps) - 1 # always use last caption for text→3D  (shortest)
            if run_text:
                for j, prompt in enumerate(caps):
                    if j != sampled_j:
                        continue  # only one caption per asset for text→3D
                    this_seed = (txt_seed + j) & 0x7fffffff
                    out_txt = txt_pipe.run(
                        prompt=prompt,
                        num_samples=1,
                        seed=this_seed,
                        formats=["gaussian"],
                    )

                    sample_obj_txt = out_txt.get("gaussian")[0]
                    if sample_obj_txt is not None:
                        # treat each (asset, caption) as its own text→3D sample
                        render_and_collect(
                            sample_obj_txt, incep, dino, device,
                            captions_for_clip=[prompt],  # CLIP will see the specific caption
                            accum_img=YI_txt_list, accum_dino=YD_txt_list,
                            accum_clip_txt_views=gen_clip_views_txt, accum_txt_caps=txt_caps_kept,
                            kept_counter=kept_txt, skipped_counter=skipped_txt,
                        )

            # ---------------- BLIP joint ----------------
            if run_joint:
                for j, prompt in enumerate(caps):
                    if j != sampled_j:
                        continue  # only one caption per asset for joint image→3D
                    this_seed = (img_seed + j) & 0x7fffffff
                    image_embeds = blip.get_image_embeds(prompt, steps=50)
                    out_joint = joint_img_pipe.run(seed=this_seed, image_embeds=image_embeds, formats=["gaussian"])

                    sample_obj_joint = out_joint.get("gaussian")[0]
                    if sample_obj_joint is not None:
                        render_and_collect(
                            sample_obj_joint, incep, dino, device,
                            captions_for_clip=[prompt],
                            accum_img=YI_joint_list, accum_dino=YD_joint_list,
                            accum_clip_txt_views=gen_clip_views_joint, accum_txt_caps=txt_caps_kept_joint,
                            kept_counter=kept_joint, skipped_counter=skipped_joint,
                        )
               
        except Exception:
            print(f"[{aid}] error:\n{traceback.format_exc()}", file=sys.stderr)

    # pack shard payload
    payload = dict(
        X_incep=X_incep, X_dino=X_dino,
        YI_img=YI_img_list, YD_img=YD_img_list, kept_img=kept_img[0], skipped_img=skipped_img[0],
        YI_txt=YI_txt_list, YD_txt=YD_txt_list, kept_txt=kept_txt[0], skipped_txt=skipped_txt[0],
        YI_joint=YI_joint_list, YD_joint=YD_joint_list, kept_joint=kept_joint[0], skipped_joint=skipped_joint[0],
        clip_views_img=gen_clip_views_img, clip_cond_imgs=cond_imgs_kept_img,
        clip_views_txt=gen_clip_views_txt, clip_txt_caps=txt_caps_kept,
        clip_views_joint=gen_clip_views_joint, clip_txt_caps_joint=txt_caps_kept_joint,
        num_assets=len(asset_ids),
    )
    return payload


# ============================= Main =============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--renders_root", type=str, required=True)
    parser.add_argument("--metadata_csv", type=str, default="")
    parser.add_argument("--results_dir",  type=str, required=True)
    parser.add_argument("--num_assets",   type=int, default=1250)
    parser.add_argument("--seed_base",    type=int, default=42)
    parser.add_argument("--selected_ids", type=str, default="", help="comma-separated sha256s")
    parser.add_argument("--selected_ids_file", type=str, default="")
    parser.add_argument("--blip_ckpt", type=str, default="", help="path to BLIP3-o checkpoint for joint image→3D")
    parser.add_argument("--dist_backend", type=str, default="nccl", help="nccl|gloo")
    parser.add_argument("--skip_image_to_3d", action="store_true",
                        help="Skip image→3D pipeline + metrics")
    parser.add_argument("--skip_text_to_3d", action="store_true",
                        help="Skip text→3D pipeline + metrics")
    parser.add_argument("--skip_blip_joint", action="store_true",
                        help="Skip BLIP joint image→3D pipeline + metrics")

    args = parser.parse_args()

    dist_setup(backend=args.dist_backend)
    rank = get_rank()
    world = get_world_size()

    # per-rank torch hub dir
    default_hub = os.environ.get(
        "TORCH_HOME",
        os.path.join(os.path.expanduser("~"), ".cache", "torch"),
    )
    rank_hub = os.path.join(default_hub, f"rank_{rank}")
    os.makedirs(rank_hub, exist_ok=True)
    os.environ["TORCH_HOME"] = rank_hub

    run_image = not args.skip_image_to_3d
    run_text  = not args.skip_text_to_3d
    run_joint = not args.skip_blip_joint
    # choose device per-rank
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = "cuda"
    else:
        device = "cpu"

    random.seed(args.seed_base); np.random.seed(args.seed_base); torch.manual_seed(args.seed_base + rank)
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)

    caps_by_sha = load_captions_by_sha(args.metadata_csv)
    if rank == 0:
        print(f"[info] Captions available for {len(caps_by_sha)} assets")

    # Step 1: build SAME global split deterministically on all ranks
    if rank == 0:
        print(f"[info] Scanning asset IDs (no image loads): {args.renders_root}")
    all_ids = list_asset_ids(args.renders_root)

    wanted = set()
    if args.selected_ids_file and os.path.isfile(args.selected_ids_file):
        with open(args.selected_ids_file) as f:
            wanted = {line.strip() for line in f if line.strip()}
    elif args.selected_ids:
        wanted = {x.strip() for x in args.selected_ids.split(",") if x.strip()}
    
    candidate_ids = [aid for aid in all_ids if not wanted or aid in wanted]
    # Deterministic shuffle
    rng = random.Random(args.seed_base)
    rng.shuffle(candidate_ids)

    # If we need text or joint metrics, require captions
    if run_text or run_joint:
        captioned_ids = [aid for aid in candidate_ids if aid in caps_by_sha]

        if len(captioned_ids) < args.num_assets:
            if rank == 0:
                print(
                    f"[warn] Only {len(captioned_ids)} assets have captions "
                    f"(requested {args.num_assets}). Using all captioned assets."
                )
            asset_ids = captioned_ids
        else:
            asset_ids = captioned_ids[:args.num_assets]
    else:
        # Image-only eval: no caption requirement
        asset_ids = candidate_ids[:min(args.num_assets, len(candidate_ids))]

    if rank == 0:
        print(f"[info] Global split size: {len(asset_ids)} assets across world_size={world}")
    barrier()

    # Step 2: shard and load only local assets
    local_ids = shard_list(asset_ids, rank, world)
    print(f"[rank {rank}] local shard size: {len(local_ids)}")

    if local_ids:
        print(f"[rank {rank}] Loading renders for local assets...")
        assets = load_assets_for_ids(args.renders_root, local_ids)
    else:
        assets = {}

    if rank == 0:
        print(f"[info] Captions available for {len(caps_by_sha)} assets")

    # Step 3: collect features on each rank
    shard_payload = evaluate_split_collect_features(
        asset_ids=local_ids,
        assets=assets,
        caps_by_sha=caps_by_sha,
        device=device,
        blip_ckpt=args.blip_ckpt if args.blip_ckpt else None,
        run_image=run_image,
        run_text=run_text,
        run_joint=run_joint,
    )

    # Step 4: gather to rank 0
    gathered = gather_objects_to_rank0(shard_payload)
    barrier()

    if rank != 0:
        return

    # Step 5: merge features on rank 0
    X_incep_all, X_dino_all = [], []
    YI_img_all, YD_img_all, kept_img_sum, skipped_img_sum = [], [], 0, 0
    YI_txt_all, YD_txt_all, kept_txt_sum, skipped_txt_sum = [], [], 0, 0
    YI_joint_all, YD_joint_all, kept_joint_sum, skipped_joint_sum = [], [], 0, 0

    # CLIP holders
    cond_imgs_kept_img_all, gen_clip_views_img_all   = [], []
    txt_caps_kept_joint_all, gen_clip_views_joint_all = [], []
    txt_caps_kept_all, gen_clip_views_txt_all = [], []

    total_assets = 0

    for pay in gathered:
        if not pay:
            continue

        total_assets += pay["num_assets"]
        X_incep_all.append(pay["X_incep"])
        X_dino_all.append(pay["X_dino"])

        YI_img_all.extend(pay["YI_img"])
        YD_img_all.extend(pay["YD_img"])
        kept_img_sum += pay["kept_img"]
        skipped_img_sum += pay["skipped_img"]

        YI_txt_all.extend(pay["YI_txt"])
        YD_txt_all.extend(pay["YD_txt"])
        kept_txt_sum += pay["kept_txt"]
        skipped_txt_sum += pay["skipped_txt"]

        if pay["YI_joint"]:
            YI_joint_all.extend(pay["YI_joint"])
            YD_joint_all.extend(pay["YD_joint"])
            kept_joint_sum += pay["kept_joint"]
            skipped_joint_sum += pay["skipped_joint"]

        gen_clip_views_img_all.extend(pay["clip_views_img"])
        cond_imgs_kept_img_all.extend(pay["clip_cond_imgs"])

        gen_clip_views_txt_all.extend(pay["clip_views_txt"])
        txt_caps_kept_all.extend(pay["clip_txt_caps"])

        gen_clip_views_joint_all.extend(pay["clip_views_joint"])
        txt_caps_kept_joint_all.extend(pay["clip_txt_caps_joint"])

    if total_assets == 0:
        print("No assets processed; exiting.")
        return

    X_incep = np.concatenate(X_incep_all, axis=0)
    X_dino  = np.concatenate(X_dino_all,  axis=0)

    # Step 6: CLIP
    clip_payload = {}
    if open_clip is not None:
        try:
            device0 = "cuda" if torch.cuda.is_available() else "cpu"
            clip_img_base = (
                clip_image_to_image_perview(
                    cond_imgs_kept_img_all, gen_clip_views_img_all, device0
                ) if gen_clip_views_img_all else None
            )
            clip_txt_base = (
                clip_text_to_image_perview(
                    txt_caps_kept_all, gen_clip_views_txt_all, device0
                ) if gen_clip_views_txt_all else None
            )
            clip_txt_joint = (
                clip_text_to_image_perview(
                    txt_caps_kept_joint_all, gen_clip_views_joint_all, device0
                ) if gen_clip_views_joint_all else None
            )
            clip_payload = {
                "image_image_baseline": clip_img_base,
                "text_image_baseline": clip_txt_base,
                "text_image_blip_joint": clip_txt_joint,
            }
        except Exception:
            clip_payload = {"note": "CLIP scoring error"}

    # Step 7: final metrics on rank 0
    results = _compute_metrics_from_features(
        X_incep, X_dino,
        YI_img_all, YD_img_all, kept_img_sum, skipped_img_sum,
        YI_txt_all, YD_txt_all, kept_txt_sum, skipped_txt_sum,
        YI_joint_all, YD_joint_all, kept_joint_sum, skipped_joint_sum,
        clip_payload, device
    )

    out_top = {
        "settings": {
            "renders_root": args.renders_root,
            "metadata_csv": args.metadata_csv,
            "results_dir":  args.results_dir,
            "image_ckpt":   IMAGE_PRETRAINED,
            "text_ckpt":    TEXT_PRETRAINED,
            "num_assets":   total_assets,
            "seed_base":    args.seed_base,
            "same_split_all_three": True,
            "world_size": get_world_size(),
        },
        "results": results,
    }

    out_json = os.path.join(args.results_dir, "metrics_tri.json")
    with open(out_json, "w") as f:
        json.dump(out_top, f, indent=2)
    print(json.dumps(out_top, indent=2))
    print("\nSaved metrics to:", out_json)

    caps_txt_path = os.path.join(args.results_dir, "used_captions.txt")
    with open(caps_txt_path, "w", encoding="utf-8") as f:
        f.write("\n# image_to_3d_blip_joint captions\n")
        for caps in txt_caps_kept_joint_all:
            for cap in caps:
                cap_clean = str(cap).replace("\n", " ").strip()
                if cap_clean:
                    f.write(cap_clean + "\n")
    print("Saved used captions to:", caps_txt_path)

if __name__ == "__main__":
    main()
th open(caps_txt_path, "w", encoding="utf-8") as f:
        f.write("\n# image_to_3d_blip_joint captions\n")
        for caps in txt_caps_kept_joint_all:
            for cap in caps:
                cap_clean = str(cap).replace("\n", " ").strip()
                if cap_clean:
                    f.write(cap_clean + "\n")
    print("Saved used captions to:", caps_txt_path)

if __name__ == "__main__":
    main()
