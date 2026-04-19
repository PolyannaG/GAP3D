import os, json, argparse, random, hashlib
from typing import List, Dict, Tuple, Any
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import sys

from coco_data_helpers import (
    build_coco_loader,
    build_coco_caption_query_loader,
)
from toys4k_data_helpers import (
    build_toys4k_loader,
    build_toys4k_caption_query_loader,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "eval"))
from model_helpers import ModelArguments, DataArguments, BlipTextEmbedder
from dist_helpers import barrier, rank_print, ddp_init_if_needed, ddp_env


def set_seed(s):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.no_grad()
def batch_ground_truth_latents(model, batch, device, float_dtype, map_to: str = "dino"):
    """GT image embedding from real images (database)."""
    model.eval().to(device)
    b = move_batch_to_device(batch, device, float_dtype=float_dtype)
    gi = b.get("gen_image", None)
    if gi is not None:
        gi = gi.to(device=device, dtype=model.get_gen_vision_tower().dtype)
    if map_to == "dino":
        (_, _, _, _, _, _, latents, summary, regs) = model.prepare_inputs_labels_for_multimodal(
            b.get("input_ids", None),
            None,
            b.get("attention_mask", None),
            None,
            b.get("labels", None),
            gi,
            None,
            None,
            b.get("i_s_pos", None),
            None,
        )
        return latents, summary  # [B,C,H,W], [B,D]
    else:  # evaclip
        (_, _, _, _, _, _, latents) = model.prepare_inputs_labels_for_multimodal(
            b.get("input_ids", None),
            None,
            b.get("attention_mask", None),
            None,
            b.get("labels", None),
            gi,
            None,
            None,
            b.get("i_s_pos", None),
            None,
        )
        return latents


def move_batch_to_device(batch, device, float_dtype=None):
    out = {}
    for k, v in batch.items():
        if k == "i_s_pos":
            out[k] = [int(x) for x in v]
        elif k in ("ids",):
            out[k] = v
        elif torch.is_tensor(v):
            t = v.to(device, non_blocking=True)
            if float_dtype is not None and t.is_floating_point():
                t = t.to(float_dtype)
            out[k] = t
        elif isinstance(v, list) and v and torch.is_tensor(v[0]):
            t = torch.stack(v, dim=0).to(device, non_blocking=True)
            if float_dtype is not None and t.is_floating_point():
                t = t.to(float_dtype)
            out[k] = t
        else:
            out[k] = v
    return out


@torch.no_grad()
def per_image_text_predicted_latents_via_loader(
    bliptextembedder,
    loader,
    device,
    steps: int,
    map_to: str = "dino",
    max_samples: int | None = None,
):
    """
    For each sample use its (already tokenized) caption prompt to predict image embeddings.
    """
    vec_chunks, vec_chunks_sum, ids_all = [], [], []
    total = 0
    for batch in tqdm(loader, desc="Caption-level text→predicted latents"):
        if max_samples is not None and total >= max_samples:
            break
        b = move_batch_to_device(batch, device, float_dtype=None)

        if map_to == "dino":
            pred_sum, _, pred_lat = bliptextembedder.get_image_embeds_batch(b, steps=steps)
        else:  # evaclip
            pred_lat = bliptextembedder.get_image_embeds_batch(b, steps=steps)

        if pred_lat.dim() == 4:
            pred_lat = pred_lat.mean(dim=(2, 3))  # [B, D]
        pred_lat = F.normalize(pred_lat.float(), p=2, dim=1).to(torch.float16).cpu()

        if map_to == "dino":
            if pred_sum.dim() == 4:
                pred_sum = pred_sum.squeeze(-1).squeeze(-1)  # [B, D]
            pred_sum = F.normalize(pred_sum.float(), p=2, dim=1).to(torch.float16).cpu()

        bsz = pred_lat.shape[0]
        total += bsz

        vec_chunks.append(pred_lat)
        if map_to == "dino":
            vec_chunks_sum.append(pred_sum)
        ids_all.extend(b["ids"])

    if map_to == "dino":
        return (
            torch.cat(vec_chunks, dim=0),
            torch.cat(vec_chunks_sum, dim=0),
            torch.tensor(ids_all, dtype=torch.long),
        )
    else:
        return torch.cat(vec_chunks, dim=0), None, torch.tensor(ids_all, dtype=torch.long)


# Cache helpers
def _cache_prefix(tag: str, args_dict: dict):
    j = json.dumps(args_dict, sort_keys=True, default=str)
    h = hashlib.md5(j.encode()).hexdigest()[:10]
    return f"{tag}.{h}"


def try_load_cached(cache_dir: str, prefix: str):
    vp = os.path.join(cache_dir, f"{prefix}.vec.fp16.npy")
    sp = os.path.join(cache_dir, f"{prefix}.sum.fp16.npy")
    ip = os.path.join(cache_dir, f"{prefix}.ids.int64.npy")
    if all(os.path.isfile(p) for p in (vp, sp, ip)):
        vec = torch.from_numpy(np.load(vp, allow_pickle=False))
        summ = torch.from_numpy(np.load(sp, allow_pickle=False))
        ids = torch.from_numpy(np.load(ip, allow_pickle=False))
        return vec, summ, ids
    return None, None, None


def save_cache(
    cache_dir: str,
    prefix: str,
    vec: torch.Tensor,
    summ: torch.Tensor,
    ids: torch.Tensor,
    map_to: str = "dino",
):
    os.makedirs(cache_dir, exist_ok=True)
    np.save(
        os.path.join(cache_dir, f"{prefix}.vec.fp16.npy"),
        vec.cpu().numpy().astype(np.float16),
        allow_pickle=False,
    )
    if map_to == "dino":
        np.save(
            os.path.join(cache_dir, f"{prefix}.sum.fp16.npy"),
            summ.cpu().numpy().astype(np.float16),
            allow_pickle=False,
        )
    np.save(
        os.path.join(cache_dir, f"{prefix}.ids.int64.npy"),
        ids.cpu().numpy().astype(np.int64),
        allow_pickle=False,
    )


# Retrieval metrics
@torch.no_grad()
def retrieval_recall_at_k(
    query_vec: torch.Tensor,  # [Q,D], L2-normalized
    db_vec: torch.Tensor,  # [N,D], L2-normalized
    query_ids: torch.Tensor,  # [Q]
    db_ids: torch.Tensor,  # [N]
    device,
    ks=(1, 5, 10),
    q_bs=4096,
    db_bs=65536,
):
    Q = query_vec.size(0)
    recalls = {f"R@{k}": 0 for k in ks}
    for s in tqdm(range(0, Q, q_bs), desc="Retrieval"):
        q = query_vec[s : s + q_bs].to(device).float()
        sims_all, idx_all = [], []
        M = db_vec.size(0)
        for t in range(0, M, db_bs):
            keys = db_vec[t : t + db_bs].to(device).float()
            sim = q @ keys.t()
            tk = min(max(ks), sim.size(1))
            s_k, i_k = sim.topk(k=tk, dim=1, largest=True, sorted=False)
            sims_all.append(s_k)
            idx_all.append(i_k + t)
            del keys, sim
        sims_cat = torch.cat(sims_all, dim=1)
        idx_cat = torch.cat(idx_all, dim=1)
        tk = max(ks)
        _, rel = sims_cat.topk(k=tk, dim=1, largest=True, sorted=False)
        idxK = idx_cat.gather(1, rel)
        top_ids = db_ids[idxK.cpu()]
        gtid = query_ids[s : s + q.size(0)].unsqueeze(1)
        for K in ks:
            hits = (top_ids[:, :K] == gtid).any(dim=1).sum().item()
            recalls[f"R@{K}"] += int(hits)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
    for K in ks:
        recalls[f"R@{K}"] = 100.0 * recalls[f"R@{K}"] / Q
    return recalls


def main():
    world_size, rank, local_rank = ddp_init_if_needed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Eval args
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument(
        "--coco-root",
        type=str,
        default="",
        help="Path to COCO root (annotations/, train2017/, val2017/)",
    )
    ap.add_argument(
        "--coco-split",
        type=str,
        default="train2017",
        choices=["train2017", "val2017"],
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--steps",
        type=int,
        default=30,
        help="inference steps for sample_images_no_cfg_cls",
    )
    ap.add_argument("--q-chunk", type=int, default=4096)
    ap.add_argument("--db-chunk", type=int, default=65536)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--recompute-cache",
        action="store_true",
        help="Ignore existing DB memmaps/.npy caches and recompute DB latents",
    )
    ap.add_argument(
        "--cache-dir",
        type=str,
        default=os.environ.get("ENCODER_CACHE_DIR", ".radio_cache"),
        help="Directory for .npy latent caches (DB only, used in in-memory path)",
    )
    ap.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to BLIP-3 checkpoint",
    )
    ap.add_argument(
        "--image_aspect_ratio",
        type=str,
        default="square",
        choices=["square", "original"],
    )
    ap.add_argument(
        "--map_to",
        type=str,
        default="dino",
        choices=["dino", "evaclip"],
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Max number of caption-level query samples (text-predicted latents) to "
            "use for evaluation."
        ),
    )
    ap.add_argument(
        "--results-dir",
        type=str,
        default="./",
        help="Directory to save JSON results",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default="coco",
        choices=["coco", "toys4k"],
    )
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

    eval_args, _ = ap.parse_known_args()

    # Enforce dataset-specific required args
    if eval_args.dataset == "coco":
        assert eval_args.coco_root, "--coco-root is required when --dataset=coco"
    else:  # toys4k
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
    float_dtype = torch.bfloat16

    blip = BlipTextEmbedder(
        ckpt=eval_args.model_name_or_path,
        device=device,
        model_args=model_args,
        data_args=data_args,
    )

    gen_vision_tower = blip.model.get_gen_vision_tower()
    data_args.gen_image_processor = gen_vision_tower.image_processor
    gen_proc = gen_vision_tower.image_processor
    n_query = blip.model.get_n_query()

    # ---------------------------
    # Build dataset loaders
    # ---------------------------
    if eval_args.dataset == "coco":
        # DB loader
        L_db, n_images = build_coco_loader(
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
        )
        rank_print(
            rank,
            f"COCO {eval_args.coco_split}: local image shard size = {n_images}",
        )

        # Caption query loader
        L_q, n_caps = build_coco_caption_query_loader(
            coco_root=eval_args.coco_root,
            split=eval_args.coco_split,
            tokenizer=blip.tokenizer,
            n_query=n_query,
            batch_size=eval_args.batch_size,
            workers=eval_args.workers,
            data_args=data_args,
            world_size=world_size,
            rank=rank,
        )
        rank_print(
            rank,
            f"COCO {eval_args.coco_split}: local caption shard size = {n_caps}",
        )

        cache_tag_base = {
            "dataset": "coco",
            "coco_root": os.path.abspath(eval_args.coco_root),
            "split": eval_args.coco_split,
            "world_size": world_size,
            "rank": rank,
            "model": model_args.model_name_or_path,
            "map_to": eval_args.map_to,
        }

    else:  # ----------------- toys4k -----------------
        # DB loader
        L_db, n_images = build_toys4k_loader(
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
        )
        rank_print(rank, f"toys4k: local image shard size = {n_images}")

        # Caption query loader
        L_q, n_caps = build_toys4k_caption_query_loader(
            renders_root=eval_args.renders_root,
            metadata_csv=eval_args.metadata_csv,
            tokenizer=blip.tokenizer,
            n_query=n_query,
            batch_size=eval_args.batch_size,
            workers=eval_args.workers,
            data_args=data_args,
            world_size=world_size,
            rank=rank,
        )
        rank_print(rank, f"toys4k: local caption shard size = {n_caps}")

        cache_tag_base = {
            "dataset": "toys4k",
            "renders_root": os.path.abspath(eval_args.renders_root),
            "metadata_csv": os.path.abspath(eval_args.metadata_csv),
            "world_size": world_size,
            "rank": rank,
            "model": model_args.model_name_or_path,
            "map_to": eval_args.map_to,
        }

    # ---------------------------
    # Compute DB (vision) latents (or reuse)
    # ---------------------------
    db_vec_local = db_sum_local = db_ids_local = None

    if not eval_args.recompute_cache:
        db_prefix = f"{eval_args.dataset}_db"  # "coco_db" or "toys4k_db"
        p_db = _cache_prefix(db_prefix, cache_tag_base)
        db_vec_local, db_sum_local, db_ids_local = try_load_cached(
            eval_args.cache_dir, p_db
        )
        if db_vec_local is not None and rank == 0:
            print(f"[cache] Loaded DB latents from {eval_args.cache_dir} with key {p_db}")

    if db_vec_local is None:
        # Need to compute DB latents
        gt_vecs, gt_summ, gt_ids = [], [], []
        for batch in tqdm(L_db, desc=f"Rank {rank} • DB latents from images"):
            if eval_args.map_to == "dino":
                lat, summary = batch_ground_truth_latents(
                    blip.model,
                    batch,
                    device,
                    float_dtype,
                    map_to=eval_args.map_to,
                )
            else:  # evaclip
                lat = batch_ground_truth_latents(
                    blip.model,
                    batch,
                    device,
                    float_dtype,
                    map_to=eval_args.map_to,
                )
            pooled = lat.mean(dim=(2, 3)).float() if lat.dim() == 4 else lat.float()
            gt_vecs.append(F.normalize(pooled, p=2, dim=1).to(torch.float16).cpu())
            if eval_args.map_to == "dino":
                gt_summ.append(
                    F.normalize(summary.float(), p=2, dim=1)
                    .to(torch.float16)
                    .cpu()
                )
            gt_ids.extend(batch["ids"])
        db_vec_local = torch.cat(gt_vecs, 0)
        if eval_args.map_to == "dino":
            db_sum_local = torch.cat(gt_summ, 0)
        db_ids_local = torch.tensor(gt_ids, dtype=torch.long)

        db_prefix = f"{eval_args.dataset}_db"
        save_cache(
            eval_args.cache_dir,
            _cache_prefix(db_prefix, cache_tag_base),
            db_vec_local,
            db_sum_local,
            db_ids_local,
            map_to=eval_args.map_to,
        )

    barrier()

    # ---------------------------
    # Query latents (caption-level)
    # ---------------------------
    q_vec_local, q_sum_local, q_ids_local = per_image_text_predicted_latents_via_loader(
        blip,
        L_q,
        device=device,
        steps=eval_args.steps,
        map_to=eval_args.map_to,
        max_samples=eval_args.max_samples,
    )

    # ---------------------------
    # Gather across DDP
    # ---------------------------

    def _gather_cpu_fp16(t):
        if world_size == 1:
            return [t] if rank == 0 else None
        obj = t.numpy()
        out = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(out, obj)
        if rank == 0:
            return [torch.from_numpy(x.copy()) for x in out]
        return None

    g_db_vec = _gather_cpu_fp16(db_vec_local)
    if eval_args.map_to == "dino":
        g_db_sum = _gather_cpu_fp16(db_sum_local)
    g_db_ids = _gather_cpu_fp16(db_ids_local)

    g_q_vec = _gather_cpu_fp16(q_vec_local)
    if eval_args.map_to == "dino":
        g_q_sum = _gather_cpu_fp16(q_sum_local)
    g_q_ids = _gather_cpu_fp16(q_ids_local)

    if rank != 0:
        return

    db_vec = torch.cat(g_db_vec, 0)
    if eval_args.map_to == "dino":
        db_sum = torch.cat(g_db_sum, 0)
    db_ids = torch.cat(g_db_ids, 0)

    q_vec = torch.cat(g_q_vec, 0)
    if eval_args.map_to == "dino":
        q_sum = torch.cat(g_q_sum, 0)
    q_ids = torch.cat(g_q_ids, 0)

    # ---------------------------
    # Evaluation (caption-level text→image)
    # ---------------------------
    rec_pool = retrieval_recall_at_k(
        q_vec,
        db_vec,
        q_ids,
        db_ids,
        device=device,
        ks=(1, 5, 10),
        q_bs=eval_args.q_chunk,
        db_bs=eval_args.db_chunk,
    )
    if eval_args.map_to == "dino":
        rec_sum = retrieval_recall_at_k(
            q_sum,
            db_sum,
            q_ids,
            db_ids,
            device=device,
            ks=(1, 5, 10),
            q_bs=eval_args.q_chunk,
            db_bs=eval_args.db_chunk,
        )

    if eval_args.dataset == "coco":
        ds_label = f"COCO {eval_args.coco_split}"
    else:
        ds_label = "toys4k"

    print(f"\n=== {ds_label} Text→Image Retrieval (Caption-level queries) ===")
    print(
        "[POOLED]  R@1: {:.2f}% | R@5: {:.2f}% | R@10: {:.2f}%".format(
            rec_pool["R@1"], rec_pool["R@5"], rec_pool["R@10"]
        )
    )
    if eval_args.map_to == "dino":
        print(
            "[SUMMARY] R@1: {:.2f}% | R@5: {:.2f}% | R@10: {:.2f}%".format(
                rec_sum["R@1"], rec_sum["R@5"], rec_sum["R@10"]
            )
        )
    print(
        f"(dataset={eval_args.dataset}, steps={eval_args.steps}, "
        f"split={eval_args.coco_split if eval_args.dataset=='coco' else 'n/a'}, "
        f"aspect={eval_args.image_aspect_ratio})"
    )

    # ---- Save JSON report ----
    os.makedirs(eval_args.results_dir, exist_ok=True)

    results = {
        "dataset": eval_args.dataset,
        "coco_root": os.path.abspath(eval_args.coco_root)
        if eval_args.dataset == "coco"
        else None,
        "renders_root": os.path.abspath(eval_args.renders_root)
        if eval_args.dataset == "toys4k"
        else None,
        "metadata_csv": os.path.abspath(eval_args.metadata_csv)
        if eval_args.dataset == "toys4k"
        else None,
        "split": eval_args.coco_split if eval_args.dataset == "coco" else None,
        "steps": eval_args.steps,
        "image_aspect_ratio": eval_args.image_aspect_ratio,
        "map_to": eval_args.map_to,
        "seed": eval_args.seed,
        "model_name_or_path": eval_args.model_name_or_path,
        "world_size": world_size,
        "n_images": int(db_vec.size(0)),
        "n_queries": int(q_vec.size(0)),
        "max_samples": original_max_samples,
        "recall_pooled": rec_pool,  # dict: {"R@1": float, "R@5": float, "R@10": float}
    }

    if eval_args.map_to == "dino":
        results["recall_summary"] = rec_sum

    out_name = f"t2i_retrieval_{eval_args.dataset}_{eval_args.map_to}.json"
    out_path = os.path.join(eval_args.results_dir, out_name)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved JSON report to: {out_path}")


if __name__ == "__main__":
    main()
