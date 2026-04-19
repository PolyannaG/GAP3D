#!/usr/bin/env python3
import os, json, argparse, glob, re
import torch
from torch.utils.data import DataLoader

from blip3o.train.train import (
    make_supervised_data_module, smart_tokenizer_and_embedding_resize,
)
from typing import Optional, Iterable, Tuple, Dict, Union
from tabulate import tabulate
import torch, torch.nn.functional as F
import torch.distributed as dist
import random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "eval"))
from model_helpers import ModelArguments, DataArguments, BlipTextEmbedder
from dist_helpers import ddp_init_if_needed, rank_print

from coco_data_helpers import build_coco_loader_blip_style
from toys4k_data_helpers import build_toys4k_loader_blip_style


def set_seed(s):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def main(attn_implementation=None):
    world_size, rank, local_rank = ddp_init_if_needed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    ap = argparse.ArgumentParser(add_help=False)
    # dataset selection
    ap.add_argument(
        "--dataset",
        type=str,
        default="coco",
        choices=["coco", "toys4k"],
        help="Dataset to evaluate: coco or toys4k",
    )

    # COCO-specific
    ap.add_argument(
        "--coco_root",
        type=str,
        default="",
        help="Path to COCO root (annotations/, train2017/, val2017/)",
    )
    ap.add_argument(
        "--coco_split",
        type=str,
        default="train2017",
        choices=["train2017", "val2017"],
    )

    # toys4k-specific
    ap.add_argument(
        "--renders_root",
        type=str,
        default="",
        help="Root of toys4k asset folders",
    )
    ap.add_argument(
        "--metadata_csv",
        type=str,
        default="",
        help="Captions CSV for toys4k",
    )

    # Shared
    ap.add_argument(
        "--results_dir",
        type=str,
        default="./",
        help="Directory to save JSON results",
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=30,
        help="inference steps for sample_images_no_cfg_cls",
    )
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--map_to",
        type=str,
        default="dino",
        choices=["dino", "evaclip"],
    )
    ap.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max number of (image, caption) samples per rank to use for evaluation.",
    )
    ap.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to BLIP-3 checkpoint",
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run the evaluation on",
    )
    ap.add_argument(
        "--image_aspect_ratio",
        type=str,
        default="square",
        choices=["square", "original"],
    )
    eval_args, _ = ap.parse_known_args()

    # dataset-specific required args
    if eval_args.dataset == "coco":
        assert eval_args.coco_root, "--coco_root is required when --dataset=coco"
    else:
        assert eval_args.renders_root, "--renders_root is required when --dataset=toys4k"
        assert eval_args.metadata_csv, "--metadata_csv is required when --dataset=toys4k"

    original_max_samples = eval_args.max_samples
    if eval_args.max_samples:
        # Get max per rank
        per_rank = (eval_args.max_samples + world_size - 1) // world_size
        eval_args.max_samples = per_rank

    if eval_args.map_to == "evaclip":
        model_args = ModelArguments(
            model_name_or_path=eval_args.model_name_or_path,
            gen_vision_tower="eva-clip-E-14-plus",
            gen_pooling="early_pool2d_4",
            predict_dino_grid=False,
            num_register_tokens=0,
        )
    else:
        model_args = ModelArguments(model_name_or_path=eval_args.model_name_or_path)

    data_args = DataArguments()

    set_seed(eval_args.seed)
    if eval_args.device is not None and world_size == 1:
        device = eval_args.device

    blip = BlipTextEmbedder(
        ckpt=eval_args.model_name_or_path,
        device=device,
        model_args=model_args,
        data_args=data_args,
    )
    gen_vision_tower = blip.model.get_gen_vision_tower()
    gen_proc = gen_vision_tower.image_processor
    n_query = blip.model.get_n_query()

    if rank == 0:
        stat = []
        for i, (n, p) in enumerate(blip.model.named_parameters()):
            stat.append([i, n, tuple(p.shape), p.requires_grad])
        print(tabulate(stat, headers=["idx", "name", "shape", "trainable"]))

        # Calculate total parameters and trainable parameters
        total_params = sum(p.numel() for p in blip.model.get_model().parameters())
        trainable_params = sum(
            p.numel() for p in blip.model.get_model().parameters() if p.requires_grad
        )

        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")

    # -----------------------------
    # Build loader for selected dataset
    # -----------------------------
    if eval_args.dataset == "coco":
        dl, n_items = build_coco_loader_blip_style(
            coco_root=eval_args.coco_root,
            split=eval_args.coco_split,
            gen_image_processor=gen_proc,
            tokenizer=blip.tokenizer,
            n_query=n_query,
            batch_size=eval_args.batch_size,
            workers=eval_args.workers,
            data_args=data_args,
            image_aspect_ratio=eval_args.image_aspect_ratio,
            world_size=world_size,
            rank=rank,
            seed=eval_args.seed,
            max_samples=eval_args.max_samples,
        )
        rank_print(rank, f"COCO {eval_args.coco_split}: local shard size = {n_items}")
        ds_label = f"coco_{eval_args.coco_split}"
    else:
        dl, n_items = build_toys4k_loader_blip_style(
            renders_root=eval_args.renders_root,
            metadata_csv=eval_args.metadata_csv,
            gen_image_processor=gen_proc,
            tokenizer=blip.tokenizer,
            n_query=n_query,
            batch_size=eval_args.batch_size,
            workers=eval_args.workers,
            data_args=data_args,
            image_aspect_ratio=eval_args.image_aspect_ratio,
            world_size=world_size,
            rank=rank,
            seed=eval_args.seed,
            max_samples=eval_args.max_samples,
        )
        rank_print(rank, f"toys4k: local shard size = {n_items}")
        ds_label = "toys4k"

    # -----------------------------
    # Evaluate
    # -----------------------------
    with torch.no_grad():
        if blip.model.predict_summary_token:
            metrics = blip.model.evaluate_mapper_cls(
                dataloader=dl,
                device=str(device),
                k_neigh=10,
            )
        elif blip.model.predict_dino_grid:
            print("Evaluating DINO grid...")
            metrics = blip.model.evaluate_mapper_dino(
                dataloader=dl,
                device=str(device),
                k_neigh=10,
                num_inference_steps=eval_args.steps,
            )
        else:
            metrics = blip.model.evaluate_mapper(
                dataloader=dl,
                device=str(device),
                k_neigh=10,
                num_inference_steps=eval_args.steps,
            )

        # Distributed aggregation
        if world_size > 1 and dist.is_initialized():
            # Aggregate weighted by sample count
            total_items_tensor = torch.tensor(float(n_items), device=device)
            dist.all_reduce(total_items_tensor, op=dist.ReduceOp.SUM)
            total_items = total_items_tensor.item()

            global_metrics = {}
            for k, v in metrics.items():
                if not isinstance(v, (int, float)):
                    # Non-numeric metrics can't be reduced sensibly; keep rank 0's version
                    print(f"Rank {rank} keeping local non-numeric metric {k}")
                    global_metrics[k] = v
                    continue

                # convert local mean to local sum
                local_sum = torch.tensor(float(v) * n_items, device=device)
                dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)  # sum of sums

                global_metrics[k] = (local_sum / total_items).item()
            metrics = global_metrics

        if rank == 0:
            print("=== EVAL METRICS ===")
            for k, v in metrics.items():
                print(f"{k}: {v:.6f}" if isinstance(v, (int, float)) else f"{k}: {v}")

            # ---- Save JSON report ----
            os.makedirs(eval_args.results_dir, exist_ok=True)

            results = {
                "dataset": eval_args.dataset,
                "coco_root": os.path.abspath(eval_args.coco_root)
                if eval_args.dataset == "coco"
                else None,
                "coco_split": eval_args.coco_split if eval_args.dataset == "coco" else None,
                "renders_root": os.path.abspath(eval_args.renders_root)
                if eval_args.dataset == "toys4k"
                else None,
                "metadata_csv": os.path.abspath(eval_args.metadata_csv)
                if eval_args.dataset == "toys4k"
                else None,
                "steps": eval_args.steps,
                "map_to": eval_args.map_to,
                "seed": eval_args.seed,
                "model_name_or_path": eval_args.model_name_or_path,
                "world_size": world_size,
                "max_samples": original_max_samples,  # global requested
            }
            results.update(metrics)

            out_name = f"eval_metrics_{ds_label}_{eval_args.map_to}.json"
            out_path = os.path.join(eval_args.results_dir, out_name)

            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

            print(f"Saved JSON report to: {out_path}")


if __name__ == "__main__":
    main()
