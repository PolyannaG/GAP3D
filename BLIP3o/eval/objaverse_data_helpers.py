import os
from pathlib import Path
import pandas as pd
from typing import Dict, Any, List

from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import Dataset as HFDataset, Features, Value

from blip3o.train.train import (
    preprocess, preprocess_multimodal,
    DataCollatorForSupervisedDataset,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent/ "eval"))
from model_helpers import DataArguments
from utils import list_asset_ids

import torch


import hashlib

def _deterministic_view_index(sha: str, num_views: int, base_seed: int) -> int:
    """
    Deterministic 'random' index in [0, num_views) for (sha, base_seed).
    """
    key = f"{sha}-{base_seed}".encode("utf-8")
    h = hashlib.sha256(key).digest()
    v = int.from_bytes(h[:4], "big")  # 32 bits is plenty
    return v % num_views


def _canonical_view_path_for_sha(renders_root: str, sha: str) -> str:
    """Choose a canonical view for DB embedding (single image per asset)."""
    sha_dir = Path(renders_root) / sha
    tjson = sha_dir / "transforms.json"
    imgs: List[Path] = []

    if tjson.exists():
        try:
            import json
            data = json.loads(tjson.read_text())
            frames = data.get("frames", [])
            for fr in frames:
                fname = fr.get("file_path")
                p = sha_dir / Path(fname).name
                if p.exists():
                    imgs.append(p)
        except Exception:
            imgs = []

    if not imgs:
        pngs = list(sha_dir.glob("*.png"))
        try:
            imgs = sorted(pngs, key=lambda x: int(x.stem))
        except ValueError:
            imgs = sorted(pngs)

    if not imgs:
        raise FileNotFoundError(f"No PNG views for asset {sha} under {sha_dir}")

    return str(imgs[0])  # deterministic canonical view

def _all_view_paths_for_sha(renders_root: str, sha: str) -> list[str]:
    """Return all PNG views for an asset, in a deterministic order."""
    sha_dir = Path(renders_root) / sha
    tjson = sha_dir / "transforms.json"
    imgs: List[Path] = []

    if tjson.exists():
        try:
            import json
            data = json.loads(tjson.read_text())
            frames = data.get("frames", [])
            for fr in frames:
                fname = fr.get("file_path")
                p = sha_dir / Path(fname).name
                if p.exists():
                    imgs.append(p)
        except Exception:
            imgs = []

    if not imgs:
        pngs = list(sha_dir.glob("*.png"))
        try:
            imgs = sorted(pngs, key=lambda x: int(x.stem))
        except ValueError:
            imgs = sorted(pngs)

    if not imgs:
        raise FileNotFoundError(f"No PNG views for asset {sha} under {sha_dir}")

    return [str(p) for p in imgs]



import hashlib

def _sha_to_int_id(sha: str) -> int:
    sha = sha.strip()
    # Try hex-based conversion first
    try:
        raw = int(sha[:16], 16)
    except ValueError:
        # Fallback: stable int64 hash
        h = hashlib.sha256(sha.encode("utf-8")).digest()
        raw = int.from_bytes(h[:8], "big")  # first 64 bits
    
    return raw & ((1 << 63) - 1)  # fit into signed int64


# def load_as_hfds(renders_root: str, metadata_csv: str) -> HFDataset:
#     """
#     DB dataset
#     Uses *only the longest caption* per asset.
#     """
#     caps_by_sha = load_captions_by_sha(metadata_csv)
#     asset_ids = sorted(list_asset_ids(renders_root))

#     def gen():
#         for sha in asset_ids:
#             caps = caps_by_sha.get(sha, [])
#             if not caps:
#                 continue

#             # list is already [longest → shortest]
#             longest = " ".join(str(caps[0]).split())
#             if not longest:
#                 continue

#             try:
#                 img_path = _canonical_view_path_for_sha(renders_root, sha)
#             except FileNotFoundError:
#                 continue

#             yield {
#                 "image_path": os.path.abspath(img_path),
#                 "txt": longest,
#                 "type": "T2I",
#                 "id": _sha_to_int_id(sha),
#             }

#     features = Features({
#         "image_path": Value("string"),
#         "txt":        Value("string"),
#         "type":       Value("string"),
#         "id":         Value("int64"),
#     })
#     return HFDataset.from_generator(gen, features=features)



def load_single_caption_by_sha(meta_csv_path: str):
    """
    Loads exactly one caption per sha from the captions CSV.

    Assumes the CSV has columns:
        sha256 : string
        captions : string (plain text caption)

    Returns:
        dict: { sha256_str : caption_str }
    """
    if not meta_csv_path or not os.path.isfile(meta_csv_path):
        print(f"Warning: captions CSV {meta_csv_path} not found", file=sys.stderr)
        return {}

    df = pd.read_csv(meta_csv_path)

    if not {"sha256", "captions"}.issubset(df.columns):
        print(f"Warning: captions CSV {meta_csv_path} missing required columns", file=sys.stderr)
        print("Columns found:", list(df.columns), file=sys.stderr)
        return {}

    # Drop rows with missing sha or caption
    df = df.dropna(subset=["sha256", "captions"])

    caps_by_sha = {}
    for _, row in df.iterrows():
        sha = str(row["sha256"]).strip()
        caption = str(row["captions"]).strip()

        if not sha or not caption:
            continue

        # Keep only the first caption if there are duplicates for the same sha
        if sha not in caps_by_sha:
            caps_by_sha[sha] = caption

    return caps_by_sha


def _sha_view_to_int_id(sha: str, view_idx: int) -> int:
    """Deterministic int64 id for a (sha, view_idx) pair."""
    base = _sha_to_int_id(sha)
    return ((base << 8) ^ view_idx) & ((1 << 63) - 1)

def load_as_hfds(renders_root: str, metadata_csv: str, base_seed: int = 0) -> HFDataset:
    """
    DB dataset.
    Uses *one sample per image view*.
    Caption strategy: map captions by index; if fewer captions than views,
    reuse the last caption.
    """
    caps_by_sha = load_single_caption_by_sha(metadata_csv)
    asset_ids = sorted(list_asset_ids(renders_root))

    # def gen():
    #     for sha in asset_ids:
    #         caps = caps_by_sha.get(sha, [])
    #         if not caps:
    #             continue

    #         # clean up all captions once
    #         caps = [" ".join(str(c).split()) for c in caps]
    #         caps = [c for c in caps if c]  # drop empty
    #         if not caps:
    #             continue

    #         try:
    #             view_paths = _all_view_paths_for_sha(renders_root, sha)
    #         except FileNotFoundError:
    #             continue

    def gen():
        for sha in asset_ids:
            caps = caps_by_sha.get(sha, None)
            if not caps:
                continue

            longest = " ".join(str(caps).split())
            if not longest:
                continue

            try:
                view_paths = _all_view_paths_for_sha(renders_root, sha)
            except FileNotFoundError:
                continue

            view_idx = _deterministic_view_index(sha, len(view_paths), base_seed)
            img_path = view_paths[view_idx]

            yield {
                "image_path": os.path.abspath(img_path),
                "txt": longest,
                "type": "T2I",
                "id": _sha_view_to_int_id(sha, view_idx),
            }

            # for view_idx, img_path in enumerate(view_paths):
            #     # caption choice:
            #     #  - if you want *same caption for all views*, use caps[0]
            #     #  - if you want different captions when available, index into caps
            #     # cap = caps[min(view_idx, len(caps) - 1)]
            #     cap = caps[0]

            #     yield {
            #         "image_path": os.path.abspath(img_path),
            #         "txt":        cap,
            #         "type":       "T2I",
            #         "id":         _sha_view_to_int_id(sha, view_idx),
            #     }

    features = Features({
        "image_path": Value("string"),
        "txt":        Value("string"),
        "type":       Value("string"),
        "id":         Value("int64"),
    })
    return HFDataset.from_generator(gen, features=features)


class T2IDataset(Dataset):
    """DB dataset (image-level), for computing ground truth image embeddings."""
    def __init__(self, hf_split, gen_image_processor, tokenizer, data_args: DataArguments, image_aspect_ratio="square"):
        self.ds = hf_split
        self.proc = gen_image_processor
        self.tok = tokenizer
        self.data_args = data_args
        self.aspect = image_aspect_ratio
        self.data_args.is_multimodal = True

    def __len__(self): return len(self.ds)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        item = self.ds[i]
        with Image.open(item["image_path"]) as im:
            pil = im.convert("RGB")
        gen_px = self.proc.preprocess([pil], return_tensors="pt")["pixel_values"]

        txt = item["txt"]
        conv = [
            {"from": "human", "value": f"Please generate image based on the following caption: {txt}"},
            {"from": "gpt",   "value": "<image>"},
        ]
        sources, _ = preprocess_multimodal([conv], self.data_args)
        tokd = preprocess(sources, self.tok, has_image=True)

        return {
            "input_ids": tokd["input_ids"][0],
            "labels":    tokd["labels"][0],
            "gen_image": gen_px,
            "ids":       int(item["id"]),
            "image_path": item.get("image_path", ""),
        }


from torch.utils.data import Subset  # at top if not already imported

def build_loader_blip_style(
    renders_root: str,
    metadata_csv: str,
    gen_image_processor,
    tokenizer,
    n_query: int,
    batch_size: int,
    workers: int,
    data_args: DataArguments,
    image_aspect_ratio: str,
    world_size: int,
    rank: int,
    seed: int,
    max_samples: int | None = None,
):

    hf = load_as_hfds(renders_root, metadata_csv, base_seed=seed).with_format("python")

    # quick sanity peek (rank 0 only)
    if rank == 0 and len(hf) > 0:
        print(hf)
        print(
            "Sample[0]:",
            {
                k: (v if k != "txt" else (v[:120] + ("..." if len(v) > 120 else "")))
                for k, v in hf[0].items()
            },
        )
        from PIL import Image as _Img
        with _Img.open(hf[0]["image_path"]) as im:
            print("Opened image:", im.size, im.mode)

    if world_size > 1:
        hf = hf.shard(num_shards=world_size, index=rank)

    ds = T2IDataset(
        hf_split=hf,
        gen_image_processor=gen_image_processor,
        tokenizer=tokenizer,
        data_args=data_args,
        image_aspect_ratio=image_aspect_ratio,
    )
    base_collator = DataCollatorForSupervisedDataset(n_query=n_query, tokenizer=tokenizer)

    def collate_with_paths(features):
        # keep raw info we need BEFORE collator squashes things
        image_paths = [f["image_path"] for f in features]
        ids = [f["ids"] for f in features]

        batch = base_collator(features)

        # reattach non-tensor metadata
        batch["image_path"] = image_paths
        # make sure ids survives as a tensor too (useful for seeding etc.)
        batch["ids"] = torch.tensor(ids, dtype=torch.long)

        return batch

    if max_samples is not None:
        print(f"[rank {rank}] Truncating dataset to max_samples={max_samples}")
        ds = Subset(ds, range(max_samples))

    L = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=workers > 0,
        drop_last=False,
        persistent_workers=workers > 0,
        collate_fn=collate_with_paths,   # <-- use wrapper instead of base_collator
    )
    return L, len(ds)

    # if max_samples is not None:
    #     print(f"[rank {rank}] Truncating dataset to max_samples={max_samples}")
    #     ds = Subset(ds, range(max_samples))

    # L = DataLoader(
    #     ds,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=workers,
    #     pin_memory=workers > 0,
    #     drop_last=False,
    #     persistent_workers=workers > 0,
    #     collate_fn=base_collator,
    # )
    # return L, len(ds)
