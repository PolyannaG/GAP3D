import os, sys, json, glob, ast
from typing import Any, List
import numpy as np
from scipy.linalg import sqrtm
from scipy import linalg
from PIL import Image
import io
import warnings
from pathlib import Path
import pandas as pd

def select_even_indices(n, k):
    if n == 0:
        return []
    if k >= n:
        return list(range(n))
    if k == 1:
        return [0]
    # Spread k indices over [0, n-1] inclusive, then dedupe preserving order
    idxs = [int(round(i * (n - 1) / (k - 1))) for i in range(k)]
    out, seen = [], set()
    for i in idxs:
        if i not in seen:
            seen.add(i)
            out.append(i)
    # fill any missing slots from 0 upward
    j = 0
    while len(out) < k:
        if j not in seen:
            seen.add(j)
            out.append(j)
        j += 1
    return out


def frechet_distance(X: np.ndarray, Y: np.ndarray, eps: float = 1e-6) -> float:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(f"X and Y must be 2D arrays, got {X.ndim}D and {Y.ndim}D")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Feature dimensions do not match: {X.shape[1]} vs {Y.shape[1]}"
        )

    # Means
    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    # Covariances
    cov_x = np.cov(X, rowvar=False)
    cov_y = np.cov(Y, rowvar=False)

    cov_prod = cov_x.dot(cov_y)
    covmean, info = linalg.sqrtm(cov_prod, disp=False)

    if not np.isfinite(covmean).all():
        msg = (
            "Frechet distance calculation produced a non-finite covariance product; "
            f"adding {eps} to the diagonal of covariance estimates."
        )
        warnings.warn(msg)
        offset = np.eye(cov_x.shape[0]) * eps
        cov_prod = (cov_x + offset).dot(cov_y + offset)
        covmean = linalg.sqrtm(cov_prod)

    # Numerical error might introduce tiny imaginary components
    if np.iscomplexobj(covmean):
        # If imaginary parts are not negligible, something's wrong
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Large imaginary component in covmean: {m}")
        covmean = covmean.real

    diff = mu_x - mu_y
    diff_term = diff.dot(diff)
    trace_term = np.trace(cov_x) + np.trace(cov_y) - 2 * np.trace(covmean)

    return float(diff_term + trace_term)

def open_rgb_black(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode == 'RGBA':
        bg = Image.new('RGBA', img.size, (0,0,0,255))
        img = Image.alpha_composite(bg, img)
    return img.convert('RGB')


def kid_mmd2_poly_degree3(X: np.ndarray, Y: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64); Y = np.asarray(Y, dtype=np.float64)
    n, m = X.shape[0], Y.shape[0]
    use_unbiased = n >= 2 and m >= 2
    d = X.shape[1]
    scale = 1.0 / d
    def poly3(A,B):
        return ((A @ B.T) * scale + 1.0) ** 3
    Kxx = poly3(X, X); Kyy = poly3(Y, Y)
    Kxy = poly3(X, Y)
    if use_unbiased:
        np.fill_diagonal(Kxx, 0.0); np.fill_diagonal(Kyy, 0.0)
        term_x = Kxx.sum() / (n*(n-1))
        term_y = Kyy.sum() / (m*(m-1))
        term_xy = Kxy.mean()
        mmd2 = term_x + term_y - 2.0*term_xy
    else:
        mmd2 = Kxx.mean() + Kyy.mean() - 2.0*Kxy.mean()
    return float(mmd2)

def load_captions_by_sha(meta_csv_path: str):
    if not meta_csv_path or not os.path.isfile(meta_csv_path): 
        print(f"Warning: captions CSV {meta_csv_path} not found", file=sys.stderr)
        return {}
    df = pd.read_csv(meta_csv_path)
    if not {"sha256","captions"}.issubset(df.columns): 
        print(f"Warning: captions CSV {meta_csv_path} missing required columns", file=sys.stderr)
        return {}
    caps_by_sha = {}
    for _, row in df.iterrows():
        sha = str(row["sha256"]).strip()
        caps_raw = row["captions"]
        caps=[]
        
        if isinstance(caps_raw,str):
            s=caps_raw.strip()
            try: 
                parsed=json.loads(s)
            except Exception:
                print(f"Warning: failed to json.loads captions for {sha}, trying ast.literal_eval", file=sys.stderr)
                try: 
                    parsed=ast.literal_eval(s)
                except Exception:
                    print(f"Warning: failed to parse captions for {sha}, using raw string", file=sys.stderr) 
                    parsed=s
            if isinstance(parsed,list):
                caps=[str(c).strip() for c in parsed if isinstance(c,str) and c.strip()]
            elif isinstance(parsed,str) and parsed.strip():
                caps=[parsed.strip()]
        if caps:
            seen=set()
            out=[]
            for c in caps:
                if c not in seen:
                    seen.add(c)
                    out.append(c)
            caps_by_sha[sha]=out
        else:
            print(f"Warning: no valid captions found for {sha}", file=sys.stderr)
    return caps_by_sha

def find_assets(renders_root: str):
    assets = {}
    root = Path(renders_root)
    for sha_dir in sorted(root.iterdir()):
        if not sha_dir.is_dir(): continue
        imgs=[]
        tjson = sha_dir / "transforms.json"
        if tjson.exists():
            try:
                data=json.loads(tjson.read_text())
                frames=data.get("frames")
                for fr in frames:
                    fname = fr.get("file_path")
                    p = sha_dir / Path(fname).name
                    if p.exists(): 
                        imgs.append(p)
            except Exception:
                print(f"Warning: failed to load transforms.json for {sha_dir}, skipping", file=sys.stderr)
                imgs=[]
        if not imgs:
            pngs=list(sha_dir.glob("*.png"))
            try: 
                imgs=sorted(pngs,key=lambda x:int(x.stem))
            except ValueError:
                print(f"Warning: non-integer png names in {sha_dir}, sorting lexically", file=sys.stderr)
                imgs=sorted(pngs)
        if imgs:
            assets[sha_dir.name]=[open_rgb_black(p) for p in imgs]
    if not assets:
        print(f"ERROR: no *.png under {renders_root}", file=sys.stderr)
        sys.exit(1)
    return assets

def list_asset_ids(renders_root: str):
    """Return asset IDs (dir names) without loading images."""
    root = Path(renders_root)
    ids = []
    for sha_dir in sorted(root.iterdir()):
        if not sha_dir.is_dir():
            continue
        # treat folders containing either transforms.json or any *.png as assets
        if (sha_dir / "transforms.json").exists() or any(sha_dir.glob("*.png")):
            ids.append(sha_dir.name)
    if not ids:
        print(f"ERROR: no assets under {renders_root}", file=sys.stderr); sys.exit(1)
    return ids

def load_assets_for_ids(renders_root: str, subset_ids: List[str]):
    """Load images only for the selected asset IDs."""
    root = Path(renders_root)
    assets = {}
    for aid in subset_ids:
        sha_dir = root / aid
        if not sha_dir.is_dir():
            continue
        imgs = []
        tjson = sha_dir / "transforms.json"
        if tjson.exists():
            try:
                data = json.loads(tjson.read_text())
                frames = data.get("frames")
                for fr in frames:
                    fname = fr.get("file_path")
                    p = sha_dir / Path(fname).name
                    if p.exists(): 
                        imgs.append(p)
            except Exception:
                print(f"Warning: failed to load transforms.json for {sha_dir}, skipping", file=sys.stderr)
                imgs = []
        if not imgs:
            pngs = list(sha_dir.glob("*.png"))
            try:
                imgs = sorted(pngs, key=lambda x: int(x.stem))
            except ValueError:
                print(f"Warning: non-integer png names in {sha_dir}, sorting lexically", file=sys.stderr)
                imgs = sorted(pngs)
        if imgs:
            assets[aid] = [open_rgb_black(p) for p in imgs]
    if not assets:
        print(f"ERROR: selected assets had no *.png under {renders_root}", file=sys.stderr)
        sys.exit(1)
    return assets