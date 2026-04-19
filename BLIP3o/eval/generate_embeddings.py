import os, json, argparse, glob, re
import torch
from torch.utils.data import DataLoader

from blip3o.train.train import (
    make_supervised_data_module, smart_tokenizer_and_embedding_resize,
)
from typing import Optional, Iterable, Tuple, Dict, Union, Any
from tabulate import tabulate
import torch, torch.nn.functional as F
import torch.distributed as dist
import random
from pathlib import Path
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "eval"))
from model_helpers import ModelArguments, DataArguments, BlipTextEmbedder
from dist_helpers import ddp_init_if_needed, rank_print

# from objaverse_data_helpers import build_loader_blip_style
from toys4k_data_helpers_for_embedding_generation import build_loader_blip_style


def set_seed(s):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def main(attn_implementation=None):
    world_size, rank, local_rank = ddp_init_if_needed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    ap = argparse.ArgumentParser(add_help=False)

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

   
    # dl, n_items = build_loader_blip_style(
    #     renders_root=eval_args.renders_root,
    #     metadata_csv=eval_args.metadata_csv,
    #     gen_image_processor=gen_proc,
    #     tokenizer=blip.tokenizer,
    #     n_query=n_query,
    #     batch_size=eval_args.batch_size,
    #     workers=eval_args.workers,
    #     data_args=data_args,
    #     image_aspect_ratio=eval_args.image_aspect_ratio,
    #     world_size=world_size,
    #     rank=rank,
    #     seed=eval_args.seed,
    #     max_samples=eval_args.max_samples,
    # )

    dl, n_items = build_loader_blip_style(
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
        selected_caps_out=os.path.join(eval_args.results_dir, "toys4k_caption_selection.csv"),
    )

    rank_print(rank, f"local shard size = {n_items}")
    ds_label = "objevarse"

    
    with torch.no_grad():
        if blip.model.predict_dino_grid:
            print("Predicting DINO grid...")
            metrics = infer_mapper_dino(
                model = blip.model,
                dataloader=dl,
                device=str(device),
                k_neigh=10,
                num_inference_steps=eval_args.steps,
                save_dir=eval_args.results_dir,
                base_seed=eval_args.seed,
            )
        else:
            metrics = blip.model.evaluate_mapper(
                dataloader=dl,
                device=str(device),
                k_neigh=10,
                num_inference_steps=eval_args.steps,
            )

    
def _index_batch(batch: Dict[str, Any], keep_idx):
    """Take a subset of the batch along the batch dimension."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v[keep_idx]
        elif isinstance(v, list):
            out[k] = [v[i] for i in keep_idx]
        else:
            out[k] = v
    return out


@torch.no_grad()
def infer_mapper_dino(
    model,
    dataloader,
    device=None,
    num_inference_steps: int = 30,
    k_neigh: int = 10,  # unused, but kept for API compat
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    use_cache: Optional[bool] = None,
    max_eval_samples: Optional[int] = None,
    save_dir: Optional[str] = None,
    base_seed: int = 0,
):
    assert save_dir is not None, "Please provide save_dir to save inferred embeddings."
    save_root = Path(save_dir) / "renders_cond"
    save_root.mkdir(parents=True, exist_ok=True)

    NUM_SAMPLES_PER_CAPTION = 1

    model.get_model().dit.eval()
    model.get_model().eval()

    print(f"Inferring on device {device}")
    model.get_model().to(device)

    from tqdm import tqdm

    for batch in tqdm(dataloader, desc="Eval mapper"):
        # ------------------------------
        # 1) Skip captions that are fully done (all 0..4.pt exist)
        # ------------------------------
        image_paths = batch["image_path"]  # list[str] of length B
        keep_idx = []

        for i, ipath in enumerate(image_paths):
            p = Path(ipath)
            sha = p.parent.name
            out_dir = save_root / sha

            all_exist = all((out_dir / f"{k}.pt").exists()
                            for k in range(NUM_SAMPLES_PER_CAPTION))

            if not all_exist:
                keep_idx.append(i)

        if len(keep_idx) == 0:
            # everything in this batch already has 0..4.pt
            continue

        if len(keep_idx) < len(image_paths):
            batch = _index_batch(batch, keep_idx)

        # ------------------------------
        # 2) For each k = 0..4, generate only missing ones
        # ------------------------------
        for k_sample in range(NUM_SAMPLES_PER_CAPTION):
            image_paths = batch["image_path"]  # possibly reduced

            # Find which captions still need this specific k_sample
            need_idx = []
            for i, ipath in enumerate(image_paths):
                p = Path(ipath)
                sha = p.parent.name
                out_dir = save_root / sha
                if not (out_dir / f"{k_sample}.pt").exists():
                    need_idx.append(i)

            # If nothing to do for this k, skip
            if len(need_idx) == 0:
                continue

            # Shrink batch to just the captions needing this k_sample
            sub_batch = _index_batch(batch, need_idx)

            # --------------------------
            # 2a) Move to device
            # --------------------------
            batch_on_dev: Dict[str, Any] = {}
            for k, v in sub_batch.items():
                batch_on_dev[k] = (
                    v.to(device, non_blocking=True)
                    if isinstance(v, torch.Tensor) else v
                )

            # --------------------------
            # 2b) Forward through BLIP-3 to get img_hidden_states
            # --------------------------
            cfg = model.config
            cur_output_attentions = (
                output_attentions
                if output_attentions is not None else cfg.output_attentions
            )
            cur_output_hidden_states = (
                output_hidden_states
                if output_hidden_states is not None else cfg.output_hidden_states
            )
            cur_return_dict = (
                return_dict if return_dict is not None else cfg.use_return_dict
            )

            input_ids = batch_on_dev.get("input_ids", None)
            position_ids = batch_on_dev.get("position_ids", None)
            attention_mask = batch_on_dev.get("attention_mask", None)
            past_key_values = batch_on_dev.get("past_key_values", None)
            labels = batch_on_dev.get("labels", None)
            gen_image = batch_on_dev.get("gen_image", None)
            und_image = batch_on_dev.get("und_image", None)
            grid_thw = batch_on_dev.get("grid_thw", None)
            i_s_pos = batch_on_dev.get("i_s_pos", None)
            image_sizes = batch_on_dev.get("image_sizes", None)

            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                latents,
                sum_latents,
                reg_latents,
            ) = model.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                gen_image,
                und_image,
                grid_thw,
                i_s_pos,
                image_sizes,
            )

            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=cur_output_attentions,
                output_hidden_states=cur_output_hidden_states,
                return_dict=cur_return_dict,
            )
            hidden_states = outputs[0]

            if labels is not None:
                img_hidden_states = []
                for b in range(hidden_states.shape[0]):
                    img_hidden_states.append(
                        hidden_states[b,
                                      i_s_pos[b]:i_s_pos[b] + model.get_n_query(),
                                      :]
                    )
                img_hidden_states = torch.stack(img_hidden_states, dim=0)
                img_hidden_states = model.get_model().down_projector(
                    img_hidden_states
                )
            else:
                raise RuntimeError("Expected labels is not None for img_hidden_states")

            B_sub = img_hidden_states.shape[0]

            # --------------------------
            # 2c) Build per-sample generators (num_images_per_prompt=1)
            # --------------------------
            ids_sub = sub_batch["ids"]
            if isinstance(ids_sub, torch.Tensor):
                ids_sub = ids_sub.cpu().tolist()

            generators: list[torch.Generator] = []
            for j in range(B_sub):
                sample_id = int(ids_sub[j])

                # Deterministic seed from (base_seed, sample_id, k_sample)
                seed = (
                    base_seed * 1000003
                    ^ sample_id * 10007
                    ^ k_sample
                ) & 0xFFFFFFFF

                g = torch.Generator(device=device)
                g.manual_seed(seed)
                generators.append(g)

            # --------------------------
            # 2d) Sample with num_images_per_prompt = 1
            # --------------------------
            if not hasattr(model, "_inference_scheduler"):
                model._inference_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                    "Alpha-VLLM/Lumina-Next-SFT-diffusers", subfolder="scheduler"
                )
            scheduler = model._inference_scheduler

            summary_embed, reg_embeds, patch_embeds = model.sample_images_cfg_dino(
                img_hidden_states,
                scheduler=scheduler,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=1,
                generator=generators,
            )

            # --------------------------
            # 2e) Flatten patch embeds & concatenate
            # --------------------------
            Bp, S, H, W = patch_embeds.shape
            patch_flat = patch_embeds.permute(0, 2, 3, 1).reshape(Bp, H * W, S)
            summary_pred = summary_embed.unsqueeze(1)
            image_embeds = torch.cat(
                [summary_pred, reg_embeds, patch_flat], dim=1
            )  # [B_sub, 1+4+H*W, D]

            # --------------------------
            # 2f) Save as <sha>/<k_sample>.pt (skip if exists)
            # --------------------------
            sub_image_paths = sub_batch["image_path"]
            assert len(sub_image_paths) == image_embeds.shape[0]

            for emb, ipath in zip(image_embeds, sub_image_paths):
                p = Path(ipath)
                sha = p.parent.name

                out_dir = save_root / sha
                out_dir.mkdir(parents=True, exist_ok=True)

                out_path = out_dir / f"{k_sample}.pt"
                if out_path.exists():
                    continue

                torch.save(emb.half().cpu(), out_path)

if __name__ == "__main__":
    main()
