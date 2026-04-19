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
import pandas as pd

from coco_data_helpers import (
    build_coco_loader,
    build_coco_caption_query_loader,
)
from toys4k_data_helpers import (
    build_toys4k_loader,
    build_toys4k_caption_query_loader,
    _sha_to_int_id,
    _canonical_view_path_for_sha,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "eval"))
from model_helpers import ModelArguments, DataArguments, BlipTextEmbedder
from dist_helpers import barrier, rank_print, ddp_init_if_needed, ddp_env


# ---------------------------
# Helpers
# ---------------------------

def set_seed(s):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


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
    vec_chunks, vec_chunks_sum, ids_all, captions_all = [], [], [], []
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


def select_even_indices(n: int, k: int) -> List[int]:
    """
    Spread k indices evenly across [0, n-1] (your helper from below, re-used).
    """
    if n == 0:
        return []
    if k >= n:
        return list(range(n))
    if k == 1:
        return [0]
    idxs = [int(round(i * (n - 1) / (k - 1))) for i in range(k)]
    out, seen = [], set()
    for i in idxs:
        if i not in seen:
            seen.add(i)
            out.append(i)
    j = 0
    while len(out) < k:
        if j not in seen:
            seen.add(j)
            out.append(j)
        j += 1
    return out


# ---------------------------
# toys4k helpers: id → {class, image_path}
# ---------------------------

def build_toys4k_id_to_info(
    renders_root: str,
    metadata_csv: str,
) -> Dict[int, Dict[str, str]]:
    """
    Build mapping from integer ID (the one used in db_ids/q_ids) to
    {
      'sha': sha256,
      'class': object class inferred from file_identifier (e.g. 'hammer'),
      'image_path': canonical PNG view
    }
    """
    id_to_info: Dict[int, Dict[str, str]] = {}

    df = pd.read_csv(metadata_csv)
    required_cols = {"sha256", "file_identifier"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"metadata_csv missing required columns: {required_cols}")

    for _, row in df.iterrows():
        sha = str(row["sha256"]).strip()
        if not sha:
            continue
        file_identifier = str(row["file_identifier"])
        # class is first folder in file_identifier, e.g. 'hammer/hammer_075/...'
        obj_class = "unknown"
        if "/" in file_identifier:
            obj_class = file_identifier.split("/", 1)[0].strip() or "unknown"

        int_id = _sha_to_int_id(sha)
        try:
            img_path = _canonical_view_path_for_sha(renders_root, sha)
        except FileNotFoundError:
            # skip assets that don't have a canonical PNG view
            continue

        id_to_info[int_id] = {
            "sha": sha,
            "class": obj_class,
            "image_path": img_path,
        }

    return id_to_info


def qualitative_toys4k_analysis(
    db_vec: torch.Tensor,
    db_ids: torch.Tensor,
    q_vec: torch.Tensor,
    q_ids: torch.Tensor,
    id_to_info: Dict[int, Dict[str, str]],
    out_dir: str,
    id_to_caption: Dict[int, str] = None,
    top_k: int = 10,
):
    """
    For each query:
      - compute full similarity to DB
      - record:
          * class counts for top1, top5, top10
          * rank of correct image
      - save top-K images (and the GT image) to disk
      - save a JSON summary in its subfolder
    """
    os.makedirs(out_dir, exist_ok=True)

    N_db = db_vec.size(0)
    print(f"[qual] DB size = {N_db}, #queries (subset) = {q_vec.size(0)}")
    db_vec_f = db_vec.float()
    q_vec_f = q_vec.float()

    for qi in tqdm(range(q_vec_f.size(0)), desc="Qualitative toys4k"):
        q_feat = q_vec_f[qi]  # [D]
        qid = int(q_ids[qi].item())
        q_info = id_to_info.get(qid, {})
        q_cls = q_info.get("class", "unknown")
        q_caption = id_to_caption.get(qid, "") if id_to_caption else ""

        # cosine scores since everything is normalized
        sims = torch.matmul(q_feat, db_vec_f.t())  # [N_db]

        # top-K indices
        top_scores, top_idx = sims.topk(k=min(top_k, N_db), largest=True, sorted=True)
        top_idx = top_idx.cpu()
        top_scores = top_scores.cpu()
        top_db_ids = db_ids[top_idx].tolist()

        # rank of correct image (1-based)
        sorted_scores, sorted_idx = sims.sort(descending=True)
        sorted_ids = db_ids[sorted_idx].cpu()
        mask = (sorted_ids == qid)
        if mask.any():
            correct_rank = int(torch.nonzero(mask, as_tuple=False)[0].item()) + 1
        else:
            correct_rank = None

        def class_counts_for(first_ids: List[int]) -> Dict[str, int]:
            cc: Dict[str, int] = {}
            for did in first_ids:
                info = id_to_info.get(int(did), {})
                cls = info.get("class", "unknown")
                cc[cls] = cc.get(cls, 0) + 1
            return cc

        top1_classes = class_counts_for(top_db_ids[:1])
        top5_classes = class_counts_for(top_db_ids[:5])
        top10_classes = class_counts_for(top_db_ids[:10])

        # Make per-query subdir
        q_dir = os.path.join(out_dir, f"q{qi:03d}_id{qid}")
        os.makedirs(q_dir, exist_ok=True)

        # Save GT image as rank00
        if q_info and os.path.isfile(q_info.get("image_path", "")):
            try:
                with Image.open(q_info["image_path"]) as im:
                    im.convert("RGB").save(
                        os.path.join(
                            q_dir,
                            f"rank00_GT_id{qid}_cls_{q_cls}.png",
                        )
                    )
            except Exception as e:
                print(f"Warning: failed to save GT image for query {qi}: {e}")

        # Save top-K retrieved images
        for rank_idx, (db_i, db_id, score) in enumerate(
            zip(top_idx.tolist(), top_db_ids, top_scores.tolist()),
            start=1,
        ):
            info = id_to_info.get(int(db_id))
            if not info:
                continue
            img_path = info.get("image_path", "")
            cls = info.get("class", "unknown")
            if not img_path or not os.path.isfile(img_path):
                continue
            fname = (
                f"rank{rank_idx:02d}_dbIdx{db_i:05d}_id{int(db_id)}"
                f"_cls_{cls}_sim_{score:.4f}.png"
            )
            try:
                with Image.open(img_path) as im:
                    im.convert("RGB").save(os.path.join(q_dir, fname))
            except Exception as e:
                print(f"Warning: failed to save retrieved image {fname}: {e}")

        # JSON summary per query
        json_data = {
            "query_index": qi,
            "query_id": qid,
            "query_class": q_cls,
            "query_caption": q_caption,
            "correct_rank": correct_rank,
            "top1_class_counts": top1_classes,
            "top5_class_counts": top5_classes,
            "top10_class_counts": top10_classes,
            "top10_ids": top_db_ids[:10],
            "top10_classes": [
                id_to_info.get(int(did), {}).get("class", "unknown")
                for did in top_db_ids[:10]
            ],
        }
        with open(os.path.join(q_dir, "summary.json"), "w") as f:
            json.dump(json_data, f, indent=2)


# ---------------------------
# Main
# ---------------------------

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
    ap.add_argument(
        "--qualitative-dir",
        type=str,
        default=None,
        help="Directory to save per-query JSON + top-K images. Required for toys4k.",
    )

    eval_args, _ = ap.parse_known_args()

    # Enforce dataset-specific required args
    if eval_args.dataset == "coco":
        assert eval_args.coco_root, "--coco-root is required when --dataset=coco"
    else:  # toys4k
        assert eval_args.renders_root, "--renders_root is required when --dataset=toys4k"
        assert eval_args.metadata_csv, "--metadata_csv is required when --dataset=toys4k"
        assert eval_args.qualitative_dir, "--qualitative-dir is required for qualitative toys4k run"

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

        id_to_info = None  # qualitative toys4k only in this script

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

        # Build id→info mapping (only needed for toys4k qualitative)
        if rank == 0:
            print("[id_to_info] building mapping from metadata_csv and renders_root...")
        id_to_info = build_toys4k_id_to_info(
            renders_root=eval_args.renders_root,
            metadata_csv=eval_args.metadata_csv,
        )
        if rank == 0:
            print(f"[id_to_info] size = {len(id_to_info)}")

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

    

    num_total_q = q_vec.size(0)
    print(
        f"Using {num_total_q} queries for metrics + qualitative analysis "
        f"(controlled via --max-samples)"
    )

    # ---------------------------
    # Evaluation (caption-level text→image) on this subset
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

    print(
        f"\n=== {ds_label} Text→Image Retrieval "
        f"(Caption-level queries, {num_total_q} queries) ==="
    )
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

    # ---- Save JSON report (subset metrics) ----
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
        "n_queries_total": int(q_vec.size(0)),
        "n_queries_eval": int(q_vec.size(0)),  # same here
        "max_samples": original_max_samples,
        "recall_pooled": rec_pool,
    }

    if eval_args.map_to == "dino":
        results["recall_summary"] = rec_sum

    out_name = f"t2i_retrieval_qual_{eval_args.dataset}_{eval_args.map_to}.json"
    out_path = os.path.join(eval_args.results_dir, out_name)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved subset JSON report to: {out_path}")


    # ---------------------------
    # Qualitative analysis for toys4k: per-query JSON + images
    # ---------------------------
    if eval_args.dataset == "toys4k":
        # Build id_to_caption mapping from id_to_info
        id_to_caption = {}
        if id_to_info:
            df = pd.read_csv(eval_args.metadata_csv)
            for _, row in df.iterrows():
                sha = str(row["sha256"]).strip()
                if not sha:
                    continue
                int_id = _sha_to_int_id(sha)
                if int_id in id_to_info:
                    # Get the caption - parse JSON array and take first (longest) caption
                    captions_str = str(row.get("captions", "[]"))
                    try:
                        captions_list = json.loads(captions_str)
                        caption = captions_list[0] if captions_list else ""
                    except (json.JSONDecodeError, IndexError):
                        caption = ""
                    id_to_caption[int_id] = caption
        
        qualitative_toys4k_analysis(
            db_vec=db_vec,
            db_ids=db_ids,
            q_vec=q_vec,
            q_ids=q_ids,
            id_to_info=id_to_info,
            out_dir=eval_args.qualitative_dir,
            id_to_caption=id_to_caption,
            top_k=10,
        )
        print(f"Saved qualitative outputs under: {eval_args.qualitative_dir}")

if __name__ == "__main__":
    main()
