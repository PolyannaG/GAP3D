# toys4k_data_helpers.py
import os
from pathlib import Path
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
from utils import load_captions_by_sha, list_asset_ids


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

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


def _sha_to_int_id(sha: str) -> int:
    """Deterministic signed int64 id from sha256 (fits into HF int64)."""
    sha = sha.strip()
    raw = int(sha[:16], 16)        # 64 bits
    return raw & ((1 << 63) - 1)   # clamp to signed int64



# -------------------------------------------------------
# DB (gallery) HF Dataset
# -------------------------------------------------------

def load_toys4k_db_as_hfds(renders_root: str, metadata_csv: str) -> HFDataset:
    """
    DB dataset for toys4k.
    Uses *only the longest caption* per asset.
    """
    caps_by_sha = load_captions_by_sha(metadata_csv)
    asset_ids = sorted(list_asset_ids(renders_root))

    def gen():
        for sha in asset_ids:
            caps = caps_by_sha.get(sha, [])
            if not caps:
                continue

            # list is already [longest → shortest]
            longest = " ".join(str(caps[0]).split())
            if not longest:
                continue

            try:
                img_path = _canonical_view_path_for_sha(renders_root, sha)
            except FileNotFoundError:
                continue

            yield {
                "image_path": os.path.abspath(img_path),
                "txt": longest,
                "type": "T2I",
                "id": _sha_to_int_id(sha),
            }

    features = Features({
        "image_path": Value("string"),
        "txt":        Value("string"),
        "type":       Value("string"),
        "id":         Value("int64"),
    })
    return HFDataset.from_generator(gen, features=features)


# -------------------------------------------------------
# Query HF Dataset – caption-level
# Uses only *longest caption*
# -------------------------------------------------------

def load_toys4k_captions_as_hfds(renders_root: str, metadata_csv: str) -> HFDataset:
    """
    Query dataset for toys4k.
    One caption per asset: the longest one.
    """
    caps_by_sha = load_captions_by_sha(metadata_csv)
    asset_ids = sorted(list_asset_ids(renders_root))

    def gen():
        for sha in asset_ids:
            caps = caps_by_sha.get(sha, [])
            if not caps:
                continue

            longest = " ".join(str(caps[0]).split())
            if not longest:
                continue

            yield {
                "image_id": _sha_to_int_id(sha),
                "txt": longest,
                "type": "T2I",
            }

    features = Features({
        "image_id": Value("int64"),
        "txt":      Value("string"),
        "type":     Value("string"),
    })
    return HFDataset.from_generator(gen, features=features)


# -------------------------------------------------------
# PyTorch datasets
# -------------------------------------------------------

class Toys4kT2IDataset(Dataset):
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


class Toys4kCaptionQueryDataset(Dataset):
    """Query (caption-level) dataset (only longest caption)."""
    def __init__(self, hf_split, tokenizer, data_args: DataArguments):
        self.ds = hf_split
        self.tok = tokenizer
        self.data_args = data_args
        self.data_args.is_multimodal = True

    def __len__(self): return len(self.ds)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        item = self.ds[i]
        txt = item["txt"]

        conv = [
            {"from": "human", "value": f"Please generate image based on the following caption: {txt}"},
            {"from": "gpt",   "value": "<image>"},
        ]
        sources, _ = preprocess_multimodal([conv], self.data_args)
        tokd = preprocess(sources, self.tok, has_image=False)

        return {
            "input_ids": tokd["input_ids"][0],
            "labels":    tokd["labels"][0],
            "ids":       int(item["image_id"]),
            "image_path": "",
        }


# -------------------------------------------------------
# Loader builders
# -------------------------------------------------------

def build_toys4k_loader(
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
):
    hf = load_toys4k_db_as_hfds(renders_root, metadata_csv).with_format("python")

     # sanity peek (rank 0 only)
    if rank == 0 and len(hf) > 0:
        print(hf)
        print("Sample[0]:", {k: (v if k != "txt" else (v[:120] + ("..." if len(v) > 120 else ""))) for k, v in hf[0].items()})
        with Image.open(hf[0]["image_path"]) as im:
            print("Opened image:", im.size, im.mode)


    if world_size > 1:
        hf = hf.shard(num_shards=world_size, index=rank)

    ds = Toys4kT2IDataset(hf, gen_image_processor, tokenizer, data_args, image_aspect_ratio)
    collate = DataCollatorForSupervisedDataset(n_query=n_query, tokenizer=tokenizer)

    L = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True,
        drop_last=False, persistent_workers=workers > 0,
        collate_fn=collate,
    )
    return L, len(ds)


def build_toys4k_caption_query_loader(
    renders_root: str,
    metadata_csv: str,
    tokenizer,
    n_query: int,
    batch_size: int,
    workers: int,
    data_args: DataArguments,
    world_size: int,
    rank: int,
):
    hf = load_toys4k_captions_as_hfds(renders_root, metadata_csv).with_format("python")

    if rank == 0 and len(hf) > 0:
        print(hf)
        print("Caption sample[0]:", {k: (v if k != "txt" else (v[:120] + ("..." if len(v) > 120 else ""))) for k, v in hf[0].items()})


    if world_size > 1:
        hf = hf.shard(num_shards=world_size, index=rank)

    ds = Toys4kCaptionQueryDataset(hf, tokenizer, data_args)
    collate = DataCollatorForSupervisedDataset(n_query=n_query, tokenizer=tokenizer)

    L = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True,
        drop_last=False, persistent_workers=workers > 0,
        collate_fn=collate,
    )
    return L, len(ds)

from torch.utils.data import Subset  # at top if not already imported

def build_toys4k_loader_blip_style(
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
    """
    BLIP-style toys4k loader:

    - Uses load_toys4k_db_as_hfds, which already returns exactly one (longest)
      caption per asset.
    - Mirrors build_coco_loader_blip_style in structure.
    """
    hf = load_toys4k_db_as_hfds(renders_root, metadata_csv).with_format("python")

    # quick sanity peek (rank 0 only)
    if rank == 0 and len(hf) > 0:
        print(hf)
        print(
            "toys4k Sample[0]:",
            {
                k: (v if k != "txt" else (v[:120] + ("..." if len(v) > 120 else "")))
                for k, v in hf[0].items()
            },
        )
        from PIL import Image as _Img
        with _Img.open(hf[0]["image_path"]) as im:
            print("Opened toys4k image:", im.size, im.mode)

    if world_size > 1:
        hf = hf.shard(num_shards=world_size, index=rank)

    ds = Toys4kT2IDataset(
        hf_split=hf,
        gen_image_processor=gen_image_processor,
        tokenizer=tokenizer,
        data_args=data_args,
        image_aspect_ratio=image_aspect_ratio,
    )
    base_collator = DataCollatorForSupervisedDataset(n_query=n_query, tokenizer=tokenizer)

    if max_samples is not None:
        print(f"[rank {rank}] Truncating toys4k dataset to max_samples={max_samples}")
        ds = Subset(ds, range(max_samples))

    L = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=workers > 0,
        drop_last=False,
        persistent_workers=workers > 0,
        collate_fn=base_collator,
    )
    return L, len(ds)
