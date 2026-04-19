import os, json, argparse, random, hashlib
from typing import List, Dict, Tuple, Any
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pycocotools.coco import COCO

from blip3o.model import blip3oQwenForCausalLM, blip3oQwenConfig
from blip3o.train.train import (
    preprocess, preprocess_multimodal,
    DataCollatorForSupervisedDataset,
)

from datasets import Dataset as HFDataset, Features, Value

# Import eval module from one level up from run dir
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent/ "eval"))
from model_helpers import DataArguments
from torch.utils.data import Subset


def load_coco_as_hfds(coco_root: str, split: str):
    """
    DB (image-level) HF Dataset:
      image_path: absolute path
      txt:        concatenated captions (newline-separated)
      type:       'T2I'
      id:         image id (int64)
    """
    ann_file = os.path.join(coco_root, "annotations", f"captions_{split}.json")
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()

    def gen():
        for img_id in img_ids:
            info = coco.loadImgs(img_id)[0]
            path = os.path.abspath(os.path.join(coco_root, split, info["file_name"]))
            if not os.path.exists(path):
                continue

            ann_ids = sorted(coco.getAnnIds(imgIds=img_id))
            anns = coco.loadAnns(ann_ids)

            caps, seen = [], set()
            for a in anns:
                cap = a.get("caption")
                if not cap:
                    continue
                cap = " ".join(cap.split())  # normalize whitespace
                if cap not in seen:
                    seen.add(cap)
                    caps.append(cap)
            if not caps:
                continue

            txt = " \n ".join(caps)
            yield {"image_path": path, "txt": txt, "type": "T2I", "id": int(img_id)}

    features = Features({
        "image_path": Value("string"),
        "txt": Value("string"),
        "type": Value("string"),
        "id":   Value("int64"),
    })
    return HFDataset.from_generator(gen, features=features)

def load_coco_captions_as_hfds(coco_root: str, split: str):
    """
    Query (caption-level) HF Dataset:
      image_id: int64
      txt:      single caption (normalized)
      type:     'T2I_CAP'
    """
    ann_file = os.path.join(coco_root, "annotations", f"captions_{split}.json")
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()

    def gen():
        for img_id in img_ids:
            ann_ids = sorted(coco.getAnnIds(imgIds=img_id))  # determinism
            anns = coco.loadAnns(ann_ids)
            seen = set()
            for a in anns:
                cap = a.get("caption")
                if not cap:
                    continue
                cap = " ".join(cap.split())
                if cap in seen:
                    continue
                seen.add(cap)
                yield {"image_id": int(img_id), "txt": cap, "type": "T2I_CAP"}

    features = Features({
        "image_id": Value("int64"),
        "txt":      Value("string"),
        "type":     Value("string"),
    })
    return HFDataset.from_generator(gen, features=features)

class CocoT2IDataset(Dataset):
    """DB dataset (image-level), used to compute *image* latents for the gallery/DB."""
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
        gen_px = self.proc.preprocess([pil], return_tensors="pt")["pixel_values"]  # DB (ground-truth) image tensor

        txt = item["txt"]  # concatenated captions (OK for DB only)
        conv = [
            {"from": "human", "value": f"Please generate image based on the following caption: {txt}"},
            {"from": "gpt",   "value": "<image>"},
        ]
        sources, _ = preprocess_multimodal([conv], self.data_args)
        tokd = preprocess(sources, self.tok, has_image=True)
        input_ids = tokd["input_ids"][0]
        labels    = tokd["labels"][0]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "gen_image": gen_px,
            "ids": int(item["id"]),  # image id
            "image_path": item.get("image_path", ""),
        }

class CocoCaptionQueryDataset(Dataset):
    """
    Caption-level queries for text→image retrieval.
    One row per caption; carries its image_id as the GT id.
    """
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
        tokd = preprocess(sources, self.tok, has_image=False)  # NO image

        input_ids = tokd["input_ids"][0]
        labels    = tokd["labels"][0]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "ids": int(item["image_id"]),  # crucial: query id == ground-truth image id
            "image_path": "",
        }


def build_coco_loader(
    coco_root: str,
    split: str,
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
    hf = load_coco_as_hfds(coco_root, split).with_format("python")

    # sanity peek (rank 0 only)
    if rank == 0 and len(hf) > 0:
        print(hf)
        print("Sample[0]:", {k: (v if k != "txt" else (v[:120] + ("..." if len(v) > 120 else ""))) for k, v in hf[0].items()})
        with Image.open(hf[0]["image_path"]) as im:
            print("Opened image:", im.size, im.mode)

    if world_size > 1:
        hf = hf.shard(num_shards=world_size, index=rank)

    ds = CocoT2IDataset(
        hf_split=hf,
        gen_image_processor=gen_image_processor,
        tokenizer=tokenizer,
        data_args=data_args,
        image_aspect_ratio=image_aspect_ratio,
    )
    base_collator = DataCollatorForSupervisedDataset(n_query=n_query, tokenizer=tokenizer)
    L = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=workers > 0, drop_last=False,
        persistent_workers=workers > 0, collate_fn=base_collator
    )
    return L, len(ds)

def build_coco_caption_query_loader(
    coco_root: str,
    split: str,
    tokenizer,
    n_query: int,
    batch_size: int,
    workers: int,
    data_args: DataArguments,
    world_size: int,
    rank: int,
):
    hf_cap = load_coco_captions_as_hfds(coco_root, split).with_format("python")

    if rank == 0 and len(hf_cap) > 0:
        print(hf_cap)
        print("Caption sample[0]:", {k: (v if k != "txt" else (v[:120] + ("..." if len(v) > 120 else ""))) for k, v in hf_cap[0].items()})

    if world_size > 1:
        hf_cap = hf_cap.shard(num_shards=world_size, index=rank)

    ds = CocoCaptionQueryDataset(hf_split=hf_cap, tokenizer=tokenizer, data_args=data_args)
    base_collator = DataCollatorForSupervisedDataset(n_query=n_query, tokenizer=tokenizer)

    L = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=workers > 0, drop_last=False,
        persistent_workers=workers > 0, collate_fn=base_collator
    )
    return L, len(ds)

def load_coco_as_hfds_blip_style(
    coco_root: str,
    split: str,
    seed: int,
):
    """
    Build a Hugging Face Dataset for COCO with EXACTLY ONE caption per image.

    Caption selection is deterministic and controlled entirely by the given
    'seed' argument. If you run this multiple times with the same seed,
    the same (image -> caption) pairs are produced.

    Output HF Dataset columns:
      - image_path : str
      - txt        : str (one selected caption)
      - type       : 'T2I'
      - id         : int64 (COCO image id)
    """

    # Use your global seeding strategy for determinism
    random.seed(seed)

    ann_file = os.path.join(coco_root, "annotations", f"captions_{split}.json")
    coco = COCO(ann_file)

    # Image IDs sorted for deterministic iteration order
    img_ids = sorted(coco.getImgIds())

    def gen():
        for img_id in img_ids:
            info = coco.loadImgs(img_id)[0]
            path = os.path.abspath(os.path.join(coco_root, split, info["file_name"]))
            if not os.path.exists(path):
                continue

            # Sorted annotation IDs → deterministic caption order
            ann_ids = sorted(coco.getAnnIds(imgIds=img_id))
            anns = coco.loadAnns(ann_ids)

            # Collect unique normalized captions
            caps, seen = [], set()
            for a in anns:
                cap = a.get("caption")
                if not cap:
                    continue
                cap = " ".join(cap.split())  # normalize whitespace
                if cap not in seen:
                    seen.add(cap)
                    caps.append(cap)

            if not caps:
                continue

            # ----- Deterministic caption sampling using the seed -----
            # Draw ONE caption index from Python's RNG (seeded above).
            # Because iteration order is deterministic and RNG is deterministic,
            # this produces exactly the same result on every run.
            idx = random.randint(0, len(caps) - 1)
            txt = caps[idx]

            yield {
                "image_path": path,
                "txt": txt,
                "type": "T2I",
                "id": int(img_id),
            }

    features = Features({
        "image_path": Value("string"),
        "txt":        Value("string"),
        "type":       Value("string"),
        "id":         Value("int64"),
    })

    return HFDataset.from_generator(gen, features=features)


def build_coco_loader_blip_style(
    coco_root: str,
    split: str,
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
    max_samples: int = None,
):
    hf = load_coco_as_hfds_blip_style(coco_root, split, seed).with_format("python")

    # quick sanity peek (rank 0 only)
    if rank == 0 and len(hf) > 0:
        print(hf)
        print("Sample[0]:", {k: (v if k != "txt" else (v[:120] + ("..." if len(v) > 120 else ""))) for k, v in hf[0].items()})
        with Image.open(hf[0]["image_path"]) as im:
            print("Opened image:", im.size, im.mode)

    if world_size > 1:
        hf = hf.shard(num_shards=world_size, index=rank)

    ds = CocoT2IDataset(
        hf_split=hf,
        gen_image_processor=gen_image_processor,
        tokenizer=tokenizer,
        data_args=data_args,
        image_aspect_ratio=image_aspect_ratio,
    )
    base_collator = DataCollatorForSupervisedDataset(n_query=n_query, tokenizer=tokenizer)

    if max_samples is not None:
        print(f"Truncating COCO dataset to max_samples={max_samples}")
        ds = Subset(ds, range(max_samples))

    L = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=workers > 0, drop_last=False,
        persistent_workers=workers > 0, collate_fn=base_collator
    )
    return L, len(ds)
