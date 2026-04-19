import os
import sys
import json
import ast
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset

from datasets import Dataset as HFDataset, Features, Value

from blip3o.train.train import (
    preprocess,
    preprocess_multimodal,
    DataCollatorForSupervisedDataset,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "eval"))
from model_helpers import DataArguments
from utils import list_asset_ids


def _deterministic_view_index(sha: str, num_views: int, base_seed: int) -> int:
    """
    Deterministic 'random' index in [0, num_views) for (sha, base_seed).
    """
    key = f"{sha}-{base_seed}".encode("utf-8")
    h = hashlib.sha256(key).digest()
    v = int.from_bytes(h[:4], "big")  # 32 bits is plenty
    return v % num_views


def _canonical_view_path_for_sha(renders_root: str, sha: str) -> str:
    """
    Choose a canonical view for DB embedding (single image per asset).
    """
    sha_dir = Path(renders_root) / sha
    tjson = sha_dir / "transforms.json"
    imgs: List[Path] = []

    if tjson.exists():
        try:
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


def _all_view_paths_for_sha(renders_root: str, sha: str) -> List[str]:
    """Return all PNG views for an asset, in a deterministic order."""
    sha_dir = Path(renders_root) / sha
    tjson = sha_dir / "transforms.json"
    imgs: List[Path] = []

    if tjson.exists():
        try:
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


def _sha_view_to_int_id(sha: str, view_idx: int) -> int:
    """Deterministic int64 id for a (sha, view_idx) pair."""
    base = _sha_to_int_id(sha)
    return ((base << 8) ^ view_idx) & ((1 << 63) - 1)


def _deterministic_caption_index(sha: str, num_caps: int, base_seed: int) -> int:
    """
    Deterministic 'random' index in [0, num_caps) for (sha, base_seed).
    """
    key = f"{sha}-cap-{base_seed}".encode("utf-8")
    h = hashlib.sha256(key).digest()
    v = int.from_bytes(h[:4], "big")
    return v % num_caps


def load_captions_list_by_sha(meta_csv_path: str) -> Dict[str, List[str]]:
    """
    Loads a *list* of captions per sha from the toys4k captions CSV.

    Assumes the CSV has columns:
        sha256  : string
        captions: string (JSON / Python list repr of captions, or a single string)

    Returns:
        dict: { sha256_str : [caption1, caption2, ...] }
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

    caps_by_sha: Dict[str, List[str]] = {}

    for _, row in df.iterrows():
        sha = str(row["sha256"]).strip()
        raw_caps = row["captions"]

        if not sha:
            continue

        # Normalize into a list[str]
        caps: List[str] = []

        if isinstance(raw_caps, str):
            s = raw_caps.strip()

            if s.startswith("[") and s.endswith("]"):
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, (list, tuple)):
                        caps = [str(c).strip() for c in parsed]
                    else:
                        caps = [str(parsed).strip()]
                except Exception:
                    caps = [s]
            else:
                caps = [s]
        else:
            caps = [str(raw_caps).strip()]

        caps = [c for c in caps if c]  # drop empty

        if not caps:
            continue

        # If multiple rows have the same sha, accumulate all captions
        if sha not in caps_by_sha:
            caps_by_sha[sha] = caps
        else:
            caps_by_sha[sha].extend(caps)

    # Optional: deduplicate captions per asset while preserving order
    for sha, caps in caps_by_sha.items():
        seen = set()
        deduped = []
        for c in caps:
            if c not in seen:
                seen.add(c)
                deduped.append(c)
        caps_by_sha[sha] = deduped

    return caps_by_sha


def load_as_hfds(
    renders_root: str,
    metadata_csv: str,
    base_seed: int = 0,
    selected_caps_out: Optional[str] = None,
) -> HFDataset:
    """
    DB dataset for toys4k.

    Uses *exactly one view* and *exactly one caption* per asset:
      - view is chosen deterministically via _deterministic_view_index
      - caption is chosen deterministically via _deterministic_caption_index
        from the list of captions associated with that sha.

    If selected_caps_out is not None, also writes a CSV with the chosen
    caption per asset (sha).
    """
    caps_by_sha = load_captions_list_by_sha(metadata_csv)
    asset_ids = sorted(list_asset_ids(renders_root))

    selection_records: List[Dict[str, Any]] = []

    def gen():
        for sha in asset_ids:
            caps = caps_by_sha.get(sha, [])
            if not caps:
                continue

            # Clean captions once
            caps_clean = [" ".join(str(c).split()) for c in caps]
            caps_clean = [c for c in caps_clean if c]  # drop empty
            if not caps_clean:
                continue

            try:
                view_paths = _all_view_paths_for_sha(renders_root, sha)
            except FileNotFoundError:
                continue

            # Deterministic view choice
            view_idx = _deterministic_view_index(sha, len(view_paths), base_seed)
            img_path = os.path.abspath(view_paths[view_idx])

            # Deterministic caption choice
            cap_idx = _deterministic_caption_index(sha, len(caps_clean), base_seed)
            caption = caps_clean[cap_idx]

            sample_id = _sha_view_to_int_id(sha, view_idx)

            # Record for CSV
            selection_records.append(
                {
                    "sha": sha,
                    "id": sample_id,
                    "view_idx": view_idx,
                    "view_path": img_path,
                    "caption_idx": cap_idx,
                    "caption": caption,
                }
            )

            yield {
                "image_path": img_path,
                "txt":        caption,
                "type":       "T2I",
                "id":         sample_id,
            }

    features = Features({
        "image_path": Value("string"),
        "txt":        Value("string"),
        "type":       Value("string"),
        "id":         Value("int64"),
    })
    ds = HFDataset.from_generator(gen, features=features)

    # Write the selection CSV (once generator has been fully consumed by from_generator)
    if selected_caps_out is not None and selection_records:
        out_path = Path(selected_caps_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_sel = pd.DataFrame(selection_records)
        df_sel.to_csv(out_path, index=False)
        print(f"Saved toys4k caption selection mapping to {out_path}")

    return ds


class T2IDataset(Dataset):
    """
    Image-level toys4k dataset, for computing ground-truth image embeddings.
    """
    def __init__(
        self,
        hf_split: HFDataset,
        gen_image_processor,
        tokenizer,
        data_args: DataArguments,
        image_aspect_ratio: str = "square",
    ):
        self.ds = hf_split
        self.proc = gen_image_processor
        self.tok = tokenizer
        self.data_args = data_args
        self.aspect = image_aspect_ratio
        self.data_args.is_multimodal = True

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        item = self.ds[i]
        with Image.open(item["image_path"]) as im:
            pil = im.convert("RGB")
        gen_px = self.proc.preprocess([pil], return_tensors="pt")["pixel_values"]

        txt = item["txt"]
        conv = [
            {
                "from": "human",
                "value": f"Please generate image based on the following caption: {txt}",
            },
            {
                "from": "gpt",
                "value": "<image>",
            },
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
    max_samples: Optional[int] = None,
    selected_caps_out: Optional[str] = None,
):
    """
    Build DataLoader + length for toys4k, BLIP-style.

    selected_caps_out (str, optional):
        If given, rank 0 will write a CSV with the chosen caption and view
        per asset at that path.
    """
    # Only rank 0 writes the CSV; other ranks pass None to avoid races
    caps_out_for_this_rank = selected_caps_out if rank == 0 else None

    hf = load_as_hfds(
        renders_root=renders_root,
        metadata_csv=metadata_csv,
        base_seed=seed,
        selected_caps_out=caps_out_for_this_rank,
    ).with_format("python")

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
    base_collator = DataCollatorForSupervisedDataset(
        n_query=n_query,
        tokenizer=tokenizer,
    )

    def collate_with_paths(features: List[Dict[str, Any]]) -> Dict[str, Any]:
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
        collate_fn=collate_with_paths,
    )
    return L, len(ds)


