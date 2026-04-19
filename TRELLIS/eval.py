
#!/usr/bin/env python3
import os, sys, json, math, random, traceback
from pathlib import Path

import numpy as np
from scipy.linalg import sqrtm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

# Trellis
from trellis.pipelines import TrellisImageTo3DPipeline, TrellisTextTo3DPipeline
from trellis.utils.render_utils import (
    yaw_pitch_r_fov_to_extrinsics_intrinsics,
    render_frames,
    render_snapshot,
)

# Safer default on clusters
os.environ["SPCONV_ALGO"] = "native"

# =========================== Config ===========================
IMAGE_PRETRAINED = "microsoft/TRELLIS-image-large"
TEXT_PRETRAINED  = "microsoft/TRELLIS-text-xlarge"

# Optional subset via env
SELECTED_IDS       = os.environ.get("EVAL_SELECTED_IDS", "").strip()
SELECTED_IDS_FILE  = os.environ.get("EVAL_SELECTED_IDS_FILE", "").strip()

# Paths
RESULTS_DIR  = "eval_results"
REN_DIR      = "/path/to/toys4k/renders"
META_CSV     = "/path/to/toys4k/metadata.csv"

# Eval scope
# NUM_ASSETS   = 1250
NUM_ASSETS   = 1250
SEED_BASE    = 42

# Camera protocol (paper)
FDKD_VIEWS   = 4           # yaw 0/90/180/270
FDKD_PITCH   = 30.0
CLIP_VIEWS   = 8           # yaw every 45°
CLIP_PITCH   = 30.0
RADIUS       = 2.0
FOV_DEG      = 40.0

# Rendering for generated views
GEN_RES      = 512
BG_COLOR     = (0, 0, 0)
SSAA         = 4

# Only score COLOR frames for FD/KD (skip normals)
STRICT_COLOR_ONLY = True

CLIP_MODEL_NAME   = "ViT-L-14"        # or "ViT-L-14-336"
CLIP_PRETRAINED   = "openai"          # keep "openai" for these two
DINO_VARIANT_HUB  = "dinov2_vitl14"   # ViT-L/14
DINO_FEAT_DIM     = 1024              # ViT-L/14 CLS dim
DINO_INPUT_SIZE   = 518               # standard eval crop used in DINOV2

# Default text prompt fallback
DEFAULT_PROMPT = "a high quality 3D toy"

# =========================== Utils ===========================
def open_rgb_black(path: Path) -> Image.Image:
    """Open PNG safely: if RGBA, composite to black; always return RGB."""
    img = Image.open(path)
    if img.mode == 'RGBA':
        bg = Image.new('RGBA', img.size, (0,0,0,255))
        img = Image.alpha_composite(bg, img)
    return img.convert('RGB')

def frechet_distance(X: np.ndarray, Y: np.ndarray, eps: float = 1e-6) -> float:
    """Fréchet distance (FID-style) between two sets of features X, Y."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    mu_x, mu_y = X.mean(0), Y.mean(0)
    cov_x = np.cov(X, rowvar=False) + np.eye(X.shape[1]) * eps
    cov_y = np.cov(Y, rowvar=False) + np.eye(Y.shape[1]) * eps
    cov_prod_sqrt = sqrtm(cov_x.dot(cov_y))
    if np.iscomplexobj(cov_prod_sqrt):
        cov_prod_sqrt = cov_prod_sqrt.real
    diff = mu_x - mu_y
    return float(diff @ diff + np.trace(cov_x + cov_y - 2 * cov_prod_sqrt))

# --- KID-style KD: unbiased MMD^2 with a degree-3 polynomial kernel, reported ×100 ---
def kid_mmd2_poly_degree3(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Kernel Distance (KD) per paper: unbiased MMD^2 with polynomial kernel of degree 3.
    Kernel: k(x,y) = ((x^T y)/d + 1)^3, where d = feature dimension.
    Returns MMD^2 (unbiased U-statistic). Report KD as (MMD^2 * 100).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n, m = X.shape[0], Y.shape[0]
    if n < 2 or m < 2:
        # U-statistic needs at least 2 samples per set; fallback to biased but keep semantics.
        # (Still safe; just avoids division by zero.)
        use_unbiased = False
    else:
        use_unbiased = True

    d = X.shape[1]
    scale = 1.0 / d

    # Gram matrices under polynomial kernel
    # K = ((X @ Y^T)/d + 1)^3
    def poly3(A, B):
        return ((A @ B.T) * scale + 1.0) ** 3

    Kxx = poly3(X, X)
    Kyy = poly3(Y, Y)
    Kxy = poly3(X, Y)

    if use_unbiased:
        np.fill_diagonal(Kxx, 0.0)
        np.fill_diagonal(Kyy, 0.0)
        term_x = Kxx.sum() / (n * (n - 1))
        term_y = Kyy.sum() / (m * (m - 1))
        term_xy = Kxy.mean()  # all nm pairs
        mmd2 = term_x + term_y - 2.0 * term_xy
    else:
        # Biased V-statistic fallback (non-negative)
        mmd2 = Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()

    return float(mmd2)

# =========================== Features =========================
class InceptionPool3(nn.Module):
    """InceptionV3 pool3 (2048-D), RAW feats (no L2)."""
    def __init__(self, device):
        super().__init__()
        weights = models.Inception_V3_Weights.IMAGENET1K_V1
        m = models.inception_v3(weights=weights, aux_logits=True, transform_input=False)  # torchvision quirk
        m.fc = nn.Identity()
        self.model = m.eval().to(device)
        self.pre   = weights.transforms()
    @torch.no_grad()
    def __call__(self, img_list, device):
        x = torch.stack([self.pre(img) for img in img_list]).to(device)
        return self.model(x)

class DINOv2Feat:
    """DINOv2 ViT-B/14 features for FD/KD.
    Prefer RAW CLS (x_prenorm[:,0]) to preserve mean/covariance geometry.
    """
    def __init__(self, device):
        self.device = device
        self.model = None
        self.pre = transforms.Compose([
            transforms.Resize(DINO_INPUT_SIZE, antialias=True),
            transforms.CenterCrop(DINO_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    def _ensure(self):
        if self.model is None:
            self.model = (
                torch.hub.load('facebookresearch/dinov2', DINO_VARIANT_HUB, pretrained=True)
                .to(self.device).eval()
            )
    @torch.no_grad()
    def __call__(self, img_list):
        self._ensure()
        x = torch.stack([self.pre(img) for img in img_list]).to(self.device)
        # Always get the dict, regardless of the forward() convenience flag
        out = self.model.forward_features(x)   # model is in .eval()

        # Prefer raw CLS for FD/KD
        # feats = out["x_prenorm"][:, 0] if "x_prenorm" in out else out["x_norm_clstoken"]
        feats = out["x_prenorm"][:, 0]
        assert feats.shape[1] == DINO_FEAT_DIM, f"expect DINO feat dim {DINO_FEAT_DIM}"

        return feats  # [N, 1024]

# =========================== CLIP =============================
try:
    import open_clip
except Exception:
    open_clip = None

@torch.no_grad()
def clip_image_to_image_perview(cond_imgs_per_asset, gen_views_per_asset, device):
    """
    CLIP image↔image alignment (per-view cosine, averaged).
    For each asset: compute CLIP image embeddings for each generated view and each conditioning image,
    take cosine(sim) for every (gen_view, cond_img) pair; average per asset, then average over assets.
    Returns average cosine × 100 (float) or None.
    """
    assert open_clip is not None, "open_clip_torch not installed"
    model, _, pre = open_clip.create_model_and_transforms(CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=device)
    model.eval()

    sims = []
    for cond_views, gen_views in zip(cond_imgs_per_asset, gen_views_per_asset):
        if not cond_views or not gen_views:
            continue

        c = torch.stack([pre(v) for v in cond_views]).to(device)
        g = torch.stack([pre(v) for v in gen_views]).to(device)

        cfeat = F.normalize(model.encode_image(c), dim=-1)  # [C, D]
        assert cfeat.shape[0] == 1, "expect 1 cond image per asset"
        gfeat = F.normalize(model.encode_image(g), dim=-1)  # [G, D]
        assert gfeat.shape[0] == CLIP_VIEWS, f"expect {CLIP_VIEWS} gen views per asset"

        # cosine for all pairs, then mean
        pair_sims = (gfeat @ cfeat.T)  # [G, C]
        assert pair_sims.shape[1] == 1 and pair_sims.shape[0] == CLIP_VIEWS, "cosine matrix shape is wrong"
        sims.append(pair_sims.mean().item())

    return float(np.mean(sims) * 100.0) if sims else None

@torch.no_grad()
def clip_text_to_image_perview(list_of_caption_lists, gen_views_per_asset, device):
    """
    CLIP text↔image alignment (per-view cosine, averaged).
    For each asset: compute text embeddings for all captions, image embeddings for each generated view,
    take cosine(sim) for every (gen_view, caption) pair; average per asset, then average over assets.
    Returns average cosine × 100 (float) or None.
    """
    assert open_clip is not None, "open_clip_torch not installed"
    model, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=device)
    model.eval()
    tok = open_clip.get_tokenizer("ViT-B-32")

    sims = []
    for caps, gv in zip(list_of_caption_lists, gen_views_per_asset):
        assert len(caps) >= 1, f"No captions found for asset!"
        if not caps or not gv:
            continue
        ims = torch.stack([pre(v) for v in gv]).to(device)          # [G, 3, H, W]
        gfeat = F.normalize(model.encode_image(ims), dim=-1)        # [G, D]
        assert gfeat.shape[0] == CLIP_VIEWS, f"expect {CLIP_VIEWS} gen views per asset"

        text = tok(caps).to(device)                                 # [T, L]
        tfeat = F.normalize(model.encode_text(text), dim=-1)        # [T, D]
        
        pair_sims = (gfeat @ tfeat.T)                               # [G, T]
        sims.append(pair_sims.mean().item())

    return float(np.mean(sims) * 100.0) if sims else None

# ====================== Data & captions =======================
def load_captions_by_sha(meta_csv_path: str):
    """Returns: dict[sha256] -> [caption, ...]; robust to JSON/Python literal/str."""
    import pandas as pd, ast
    if not os.path.isfile(meta_csv_path):
        return {}
    df = pd.read_csv(meta_csv_path)
    required = {"sha256", "captions"}
    if not required.issubset(df.columns):
        return {}
    caps_by_sha = {}
    for _, row in df.iterrows():
        sha = str(row["sha256"]).strip()
        caps_raw = row["captions"]
        if isinstance(caps_raw, float):  # NaN
            continue
        caps = []
        if isinstance(caps_raw, list):
            caps = [str(c).strip() for c in caps_raw if isinstance(c, str) and c.strip()]
        elif isinstance(caps_raw, str):
            s = caps_raw.strip()
            try: parsed = json.loads(s)
            except Exception:
                try: parsed = ast.literal_eval(s)
                except Exception: parsed = s
            if isinstance(parsed, list):
                caps = [str(c).strip() for c in parsed if isinstance(c, str) and c.strip()]
            elif isinstance(parsed, str) and parsed.strip():
                caps = [parsed.strip()]
        if caps:
            seen=set(); clean=[]
            for c in caps:
                if c not in seen:
                    seen.add(c); clean.append(c)
            caps_by_sha[sha]=clean
    print(caps_by_sha)
    return caps_by_sha

def find_assets(renders_root: str):
    """Collect 4-view renders from renders/<sha256>/ using transforms.json order, else sorted *.png."""
    assets = {}
    root = Path(renders_root)
    for sha_dir in sorted(root.iterdir()):
        if not sha_dir.is_dir(): continue
        imgs = []
        tjson = sha_dir / "transforms.json"
        if tjson.exists():
            try:
                data = json.loads(tjson.read_text())
                frames = data.get("frames") or data.get("views") or []
                for fr in frames:
                    fname = fr.get("file_path") or fr.get("file_name") or fr.get("image")
                    if not fname: continue
                    p = sha_dir / Path(fname).name
                    if p.exists(): imgs.append(p)
            except Exception:
                imgs = []
        if not imgs:
            pngs = list(sha_dir.glob("*.png"))
            try: imgs = sorted(pngs, key=lambda x: int(x.stem))
            except ValueError: imgs = sorted(pngs)
        if imgs:
            assets[sha_dir.name] = [open_rgb_black(p) for p in imgs]
    if not assets:
        print(f"ERROR: no *.png under {renders_root}", file=sys.stderr); sys.exit(1)
    return assets

def select_even_indices(n, k):
    if n == 0: return []
    if k >= n: return list(range(n))
    idxs = [round(i*(n/k)) % n for i in range(k)]
    out, seen = [], set()
    for i in idxs:
        if i not in seen: seen.add(i); out.append(i)
    j=0
    while len(out) < k and len(out) < n:
        if j not in seen: seen.add(j); out.append(j)
        j += 1
    return out

# =================== Render helpers (Trellis) =================
def render_generated_four_views_with_channel(sample_obj):
    """4-view rig via render_snapshot; returns (frames, channel='color'|'normal'|'none')."""
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
        frames = [Image.fromarray(c) for c in rets['color']]
        return frames, 'color'
    elif 'normal' in rets:
        frames = [Image.fromarray(c) for c in rets['normal']]
        return frames, 'normal'
    return [], 'none'

def render_generated_eight_views(sample_obj):
    """8-view CLIP sweep: yaw every 45°, pitch 30°, r=2, fov=40°."""
    yaws   = [i * (2*math.pi/CLIP_VIEWS) for i in range(CLIP_VIEWS)]
    pitchs = [math.radians(CLIP_PITCH)] * CLIP_VIEWS
    extr, intr = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs=RADIUS, fovs=FOV_DEG)
    rets = render_frames(sample_obj, extr, intr, {'resolution': GEN_RES, 'bg_color': BG_COLOR, 'ssaa': SSAA})
    key = 'color' if 'color' in rets else 'normal'
    return [Image.fromarray(c) for c in rets[key]], key

# ============================= Main ============================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(SEED_BASE); np.random.seed(SEED_BASE); torch.manual_seed(SEED_BASE)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------- Load reference renders -------
    print(f"[info] Loading reference renders from: {REN_DIR}")
    assets = find_assets(REN_DIR)
    all_ids = sorted(assets.keys())

    # Optional selection
    wanted = set()
    if SELECTED_IDS_FILE and os.path.isfile(SELECTED_IDS_FILE):
        with open(SELECTED_IDS_FILE) as f: wanted = {line.strip() for line in f if line.strip()}
    elif SELECTED_IDS:
        wanted = {x.strip() for x in SELECTED_IDS.split(",") if x.strip()}
    asset_ids = [aid for aid in all_ids if aid in wanted] if wanted else all_ids
    random.shuffle(asset_ids)
    asset_ids = asset_ids[:min(NUM_ASSETS, len(asset_ids))]
    print(f"[info] Evaluating {len(asset_ids)} assets.")

    # Extract reference features (per-image; 4 per asset)
    incep = InceptionPool3(device)
    dino  = DINOv2Feat(device)

    X_incep_list, X_dino_list = [], []
    cond_imgs_per_asset = []
    canonical_ok = 0
    for aid in asset_ids:
        n = len(assets[aid])
        if n < 1: continue
        # Prefer first 4 (if canonical export), else take evenly spaced 4
        idxs = list(range(FDKD_VIEWS)) if n >= FDKD_VIEWS else select_even_indices(n, FDKD_VIEWS)
        fd_imgs = [assets[aid][i] for i in idxs]
        # reference feats (concatenate; no per-asset mean)
        X_incep_list.append( incep(fd_imgs, device=device).cpu().numpy() )
        X_dino_list.append(  dino(fd_imgs).cpu().numpy() )
        # 1 conditioning image for image↔image CLIP
        cond_imgs_per_asset.append( [fd_imgs[0]] )
        if n == 4: 
            canonical_ok += 1
        else:
            print(f"[warn] {aid}: {n} renders found (not canonical 4-view export)")
    if not X_incep_list:
        print("No reference features extracted; nothing to evaluate.", file=sys.stderr)
        sys.exit(1)
    X_incep = np.concatenate(X_incep_list, axis=0)  # [4N, 2048]
    X_dino  = np.concatenate(X_dino_list,  axis=0)  # [4N, D]
    print(f"[info] Reference features: Inception {X_incep.shape}, DINO {X_dino.shape}; canonical folders={canonical_ok}/{len(asset_ids)}")

    # ------- Captions; pipelines -------
    caps_by_sha = load_captions_by_sha(META_CSV)
    print(f"[info] Loaded captions for {len(caps_by_sha)} assets (by sha256).")

    img_pipe = TrellisImageTo3DPipeline.from_pretrained(IMAGE_PRETRAINED)
    img_pipe.cuda()
    txt_pipe = TrellisTextTo3DPipeline.from_pretrained(TEXT_PRETRAINED)
    txt_pipe.cuda()

    
    # ------- Generate & score -------
    YI_img_list, YD_img_list = [], []
    YI_txt_list, YD_txt_list = [], []
    # CLIP holders that are guaranteed aligned
    cond_imgs_kept_img = []     # list of [cond_img] per successful image→3D asset
    gen_clip_views_img = []     # list of [8 PIL images] per successful asset
    txt_caps_kept = []          # list of [captions...] per successful text→3D asset
    gen_clip_views_txt = []     # list of [8 PIL images] per successful asset
    kept_img = kept_txt = skipped_img = skipped_txt = 0


    for idx, (aid, cond_views) in enumerate(zip(asset_ids, cond_imgs_per_asset)):
        try:
            img_seed = (SEED_BASE * 1009 + idx) & 0x7fffffff
            txt_seed = (SEED_BASE * 2003 + idx) & 0x7fffffff

        
            # ---- Image→3D ----
            out = img_pipe.run(
                image=cond_views[0], 
                num_samples=1, seed=img_seed, 
                formats=["gaussian","mesh"], preprocess_image=True, 
            )
            sample_obj = None
            
            
            sample_obj = out.get("gaussian")[0]
            # if sample_obj is None:
            #     print(f"[img] {aid}: no gaussian!")
            assert sample_obj is not None, f"[img] {aid}: no gaussian; cannot proceed"
          
            gen4, ch = render_generated_four_views_with_channel(sample_obj)
            if STRICT_COLOR_ONLY and ch != 'color':
                skipped_img += 1
            else:
                # FD/KD features (always 4-view)
                YI_img_list.append( incep(gen4, device=device).cpu().numpy() )
                YD_img_list.append(  dino(gen4).cpu().numpy() )
                kept_img += 1

                # CLIP views (8-view) — keep only on success
                clip8, ch8 = render_generated_eight_views(sample_obj)
                assert ch8 == 'color', f"[img] {aid}: unexpected channel {ch8}; cannot proceed"
                gen_clip_views_img.append(clip8)
                cond_imgs_kept_img.append(cond_views)   # align 1:1 with gen_clip_views_img

            # ---- Text→3D ----
            caps = caps_by_sha.get(aid)
            prompt_list = caps if (caps and len(caps)>0) else [DEFAULT_PROMPT]
            prompt = prompt_list[0]

            out_t = txt_pipe.run(prompt=prompt, num_samples=1, seed=txt_seed, formats=["gaussian","mesh"])
            sample_obj_t = None
            for key in ("gaussian", "mesh", "radiance_field"):
                v = out_t.get(key)
                if isinstance(v, (list, tuple)) and len(v) > 0: sample_obj_t = v[0]; break
                if v is not None: sample_obj_t = v; break

            if sample_obj_t is None:
                print(f"[txt] {aid}: no gaussian/mesh; skip")
            else:
                gen4_t, ch_t = render_generated_four_views_with_channel(sample_obj_t)
                if STRICT_COLOR_ONLY and ch_t != 'color':
                    skipped_txt += 1
                else:
                    YI_txt_list.append( incep(gen4_t, device=device).cpu().numpy() )
                    YD_txt_list.append(  dino(gen4_t).cpu().numpy() )
                    kept_txt += 1

                    clip8_t, ch8t = render_generated_eight_views(sample_obj_t)
                    if ch8t == 'color':
                        gen_clip_views_txt.append(clip8_t)
                        txt_caps_kept.append(prompt_list)       # align 1:1 with gen_clip_views_txt


        except Exception:
            print(f"[{aid}] error:\n{traceback.format_exc()}", file=sys.stderr)

    print(f"[summary] kept img={kept_img}, skipped img(normals)={skipped_img}")
    print(f"[summary] kept txt={kept_txt}, skipped txt(normals)={skipped_txt}")

    # -------------------------- Results ---------------------------
    results = {"settings": {
        "num_assets_requested": NUM_ASSETS,
        "num_assets_with_ref": len(asset_ids),
        "fd_kd_views": {"num": FDKD_VIEWS, "pitch_deg": FDKD_PITCH},
        "clip_views":  {"num": CLIP_VIEWS, "pitch_deg": CLIP_PITCH},
        "radius": RADIUS, "fov_deg": FOV_DEG, "gen_res": GEN_RES, "ssaa": SSAA,
        "seed_base": SEED_BASE, "renders_root": REN_DIR,
        "image_ckpt": IMAGE_PRETRAINED, "text_ckpt": TEXT_PRETRAINED,
        "metadata_csv": META_CSV,
        "strict_color_only": STRICT_COLOR_ONLY,
        "canonical_ref_folders": int(canonical_ok),
        "kd_definition": "Unbiased MMD^2 with polynomial kernel degree 3; reported as (MMD^2 * 100).",
        "clip_alignment": "Average of per-view cosine similarities (no pre-mean-pooling)."
    }}

    # ----- Image→3D FD/KD (KD = unbiased poly3 MMD^2 ×100) -----
    if YI_img_list:
        Yi_img = np.concatenate(YI_img_list, axis=0)  # [4M, D]
        Yd_img = np.concatenate(YD_img_list,  axis=0)

        results["image_to_3d"] = {
            "fd_inception": float(frechet_distance(X_incep, Yi_img)),
            "kd_inception": float(kid_mmd2_poly_degree3(X_incep, Yi_img) * 100.0),
            "fd_dinov2":    float(frechet_distance(X_dino,  Yd_img)),
            "kd_dinov2":    float(kid_mmd2_poly_degree3(X_dino,  Yd_img) * 100.0),
            "num_eval_images": int(len(Yi_img)),
            "num_assets_kept": int(len(YI_img_list)),
        }
    else:
        results["image_to_3d"] = {"num_eval_images": 0, "note": "no successful generations (color)"}

    # ----- Text→3D FD/KD (same KD definition) -----
    if YI_txt_list:
        Yi_txt = np.concatenate(YI_txt_list, axis=0)
        Yd_txt = np.concatenate(YD_txt_list,  axis=0)
        results["text_to_3d"] = {
            "fd_inception": float(frechet_distance(X_incep, Yi_txt)),
            "kd_inception": float(kid_mmd2_poly_degree3(X_incep, Yi_txt) * 100.0),
            "fd_dinov2":    float(frechet_distance(X_dino,  Yd_txt)),
            "kd_dinov2":    float(kid_mmd2_poly_degree3(X_dino,  Yd_txt) * 100.0),
            "num_eval_images": int(len(Yi_txt)),
            "num_assets_kept": int(len(YI_txt_list)),
        }
    else:
        results["text_to_3d"] = {"num_eval_images": 0, "note": "no successful generations (color)"}

    
    # ----- CLIP alignment (8 generated views only), per-view cosine avg -----
    if open_clip is None:
        results["clip"] = {"note": "open_clip_torch not installed"}
    else:
        try:
            clip_img = clip_image_to_image_perview(
                cond_imgs_kept_img,        # already aligned
                gen_clip_views_img,        # already aligned
                device=device
            ) if gen_clip_views_img else None
        except Exception:
            clip_img = None; print("[CLIP image↔image] error", file=sys.stderr)

        try:
            clip_txt = clip_text_to_image_perview(
                txt_caps_kept,             # already aligned
                gen_clip_views_txt,        # already aligned
                device=device
            ) if gen_clip_views_txt else None
        except Exception:
            clip_txt = None; print("[CLIP text↔image] error", file=sys.stderr)

        results["clip"] = {
            "image_image": None if clip_img is None else float(clip_img),
            "text_image":  None if clip_txt is None else float(clip_txt),
            "num_img_assets": len(gen_clip_views_img),
            "num_txt_assets": len(gen_clip_views_txt),
        }


    # ----- Self-FD sanity -----
    try:
        perm = np.random.permutation(len(X_incep))
        results["sanity"] = {
            "fd_self_inception": float(frechet_distance(X_incep, X_incep[perm])),
            "fd_self_dino": float(frechet_distance(X_dino, X_dino[np.random.permutation(len(X_dino))])),
        }
    except Exception:
        pass

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_json = os.path.join(RESULTS_DIR, "metrics.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, f_indent=2) if hasattr(json, 'f_indent') else json.dumps(results, indent=2))
    print("\nSaved metrics to:", out_json)

if __name__ == "__main__":
    main()
    print("\nSaved metrics to:", out_json)

if __name__ == "__main__":
    main()
