
import os, json, argparse, random, hashlib
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor
from PIL import Image
from tqdm import tqdm
from blip3o import conversation as conversation_lib
from dataclasses import dataclass, field
from pycocotools.coco import COCO
import transformers
from datasets import Dataset as HFDataset, Features, Value
from blip3o.constants import IMAGE_TOKEN_IDX

def load_coco_as_hfds(coco_root: str, split: str):
    ann_file = os.path.join(coco_root, "annotations", f"captions_{split}.json")
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()

    def gen():
        for img_id in img_ids:
            info = coco.loadImgs(img_id)[0]
            path = os.path.abspath(os.path.join(coco_root, split, info["file_name"]))
            if not os.path.exists(path): continue
            ann_ids = sorted(coco.getAnnIds(imgIds=img_id))
            anns = coco.loadAnns(ann_ids)
            caps, seen = [], set()
            for a in anns:
                cap = a.get("caption")
                if not cap: continue
                cap = " ".join(cap.split())
                if cap not in seen:
                    seen.add(cap); caps.append(cap)
            if not caps: continue
            txt = " \n ".join(caps)
            yield {"image_path": path, "txt": txt, "type": "T2I", "id": int(img_id)}

    features = Features({
        "image_path": Value("string"),
        "txt": Value("string"),
        "type": Value("string"),
        "id":   Value("int64"),
    })
    return HFDataset.from_generator(gen, features=features)


def move_batch_to_device(batch, device, float_dtype=None):
    out = {}
    for k, v in batch.items():
        if k == "i_s_pos": out[k] = [int(x) for x in v]
        elif k in ("ids",): out[k] = v
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
def batch_ground_truth_latents(model, batch, device, float_dtype):
    model.eval().to(device)
    b = move_batch_to_device(batch, device, float_dtype=float_dtype)
    gi = b.get("gen_image", None)
    if gi is not None:
        gi = gi.to(device=device, dtype=model.get_gen_vision_tower().dtype)
    (_, _, _, _, _, _, latents, summary, regs) = model.prepare_inputs_labels_for_multimodal(
        b.get("input_ids", None), None, b.get("attention_mask", None), None, b.get("labels", None),
        gi, None, None, b.get("i_s_pos", None), None,
    )
    return latents, summary

@torch.no_grad()
def per_image_text_predicted_latents_via_loader(model, loader, device, n_query: int):
    if not hasattr(model, "_inference_scheduler"):
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        model._inference_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            "Alpha-VLLM/Lumina-Next-SFT-diffusers", subfolder="scheduler"
        )
    scheduler = model._inference_scheduler
    vec_chunks, vec_chunks_sum, ids_all = [], [], []
    for batch in tqdm(loader, desc="Per-image text→predicted latents"):
        b = move_batch_to_device(batch, device, float_dtype=None)
        input_ids, attention_mask, labels, i_s_pos = b["input_ids"], b["attention_mask"], b["labels"], b["i_s_pos"]
        latent_queries = model.get_model().latent_queries.repeat(input_ids.size(0), 1, 1)
        H = latent_queries.shape[-1]
        latent_queries = latent_queries.contiguous().view(-1, H)
        image_idx = (input_ids == IMAGE_TOKEN_IDX)
        output_indicator = (labels != -100)
        text_embeds = model.get_model().embed_tokens(input_ids)
        gen_img_idx = torch.logical_and(output_indicator, image_idx)
        text_embeds = text_embeds.clone()
        text_embeds[gen_img_idx] = latent_queries
        labels = labels.clone()
        labels[image_idx] = -100
        outputs = model.model(inputs_embeds=text_embeds, attention_mask=attention_mask, return_dict=None)
        hidden_states = outputs[0]
        img_hidden_states = []
        for b_ix in range(hidden_states.shape[0]):
            img_hidden_states.append(hidden_states[b_ix, i_s_pos[b_ix]:i_s_pos[b_ix]+n_query, :])
        img_hidden_states = torch.stack(img_hidden_states, dim=0)
        img_hidden_states = model.get_model().down_projector(img_hidden_states)
        B = input_ids.size(0)
        assert gen_img_idx.sum().item() == B*n_query
        assert img_hidden_states.shape[1] == n_query
        pred_sum, pred_reg, pred_lat = model.sample_images_cfg_dino(
            img_hidden_states, scheduler=scheduler, num_inference_steps=30, num_images_per_prompt=1, generator=None,
        )
        pooled = pred_lat
        if pooled.dim()==4: 
            pooled = pooled.mean(dim=(2,3))
        else:
            raise ValueError(f"pred_lat has unexpected dim {pooled.dim()}")
        pooled = F.normalize(pooled.float(), p=2, dim=1).to(torch.float16).cpu()
        if pred_sum.dim()==4 or pred_sum.dim()==3: 
            # pred_sum = pred_sum.squeeze(-1).squeeze(-1)
            raise ValueError(f"pred_sum has unexpected dim {pred_sum.dim()}")
        pred_sum = F.normalize(pred_sum.float(), p=2, dim=1).to(torch.float16).cpu()
        vec_chunks.append(pooled); vec_chunks_sum.append(pred_sum); ids_all.extend(b["ids"])
    return torch.cat(vec_chunks, 0), torch.cat(vec_chunks_sum, 0), torch.tensor(ids_all, dtype=torch.long)

def _cache_prefix(tag: str, args_dict: dict):
    j = json.dumps(args_dict, sort_keys=True, default=str)
    return f"{tag}.{hashlib.md5(j.encode()).hexdigest()[:10]}"

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

def save_cache(cache_dir: str, prefix: str, vec: torch.Tensor, summ: torch.Tensor, ids: torch.Tensor):
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, f"{prefix}.vec.fp16.npy"),  vec.cpu().numpy().astype(np.float16), allow_pickle=False)
    np.save(os.path.join(cache_dir, f"{prefix}.sum.fp16.npy"),  summ.cpu().numpy().astype(np.float16), allow_pickle=False)
    np.save(os.path.join(cache_dir, f"{prefix}.ids.int64.npy"), ids.cpu().numpy().astype(np.int64),  allow_pickle=False)

def write_memmap(prefix: str, vec: torch.Tensor, summ: torch.Tensor, ids: torch.Tensor, rank: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    n, d  = vec.shape
    n2, d2 = summ.shape
    assert n==n2 and ids.numel()==n
    vpath = os.path.join(out_dir, f"{prefix}.vec.rank{rank}.fp16.mmap")
    spath = os.path.join(out_dir, f"{prefix}.sum.rank{rank}.fp16.mmap")
    ipath = os.path.join(out_dir, f"{prefix}.ids.rank{rank}.int64.mmap")
    meta  = os.path.join(out_dir, f"{prefix}.rank{rank}.meta.json")
    vmm = np.memmap(vpath, mode="w+", dtype=np.float16, shape=(n, d))
    smm = np.memmap(spath, mode="w+", dtype=np.float16, shape=(n, d2))
    imm = np.memmap(ipath, mode="w+", dtype=np.int64,   shape=(n,))
    vmm[:] = vec.cpu().numpy().astype(np.float16)
    smm[:] = summ.cpu().numpy().astype(np.float16)
    imm[:] = ids.cpu().numpy().astype(np.int64)
    vmm.flush(); smm.flush(); imm.flush()
    with open(meta, "w") as f: json.dump({"n": int(n), "d": int(d), "d_sum": int(d2)}, f)

def memmaps_exist(prefix: str, world_size: int, out_dir: str) -> bool:
    for r in range(world_size):
        meta = os.path.join(out_dir, f"{prefix}.rank{r}.meta.json")
        vmm  = os.path.join(out_dir, f"{prefix}.vec.rank{r}.fp16.mmap")
        smm  = os.path.join(out_dir, f"{prefix}.sum.rank{r}.fp16.mmap")
        imm  = os.path.join(out_dir, f"{prefix}.ids.rank{r}.int64.mmap")
        if not (os.path.isfile(meta) and os.path.isfile(vmm) and os.path.isfile(smm) and os.path.isfile(imm)):
            return False
    return True

def open_memmap_shards(prefix: str, world_size: int, out_dir: str):
    vecs, sums, ids, d, d_sum = [], [], [], None, None
    for r in range(world_size):
        meta = json.load(open(os.path.join(out_dir, f"{prefix}.rank{r}.meta.json")))
        n, d_r, d_sum_r = meta["n"], meta["d"], meta["d_sum"]
        if d is None: d = d_r
        if d_sum is None: d_sum = d_sum_r
        vecs.append(np.memmap(os.path.join(out_dir, f"{prefix}.vec.rank{r}.fp16.mmap"), mode="r", dtype=np.float16, shape=(n, d_r)))
        sums.append(np.memmap(os.path.join(out_dir, f"{prefix}.sum.rank{r}.fp16.mmap"), mode="r", dtype=np.float16, shape=(n, d_sum_r)))
        ids.append(np.memmap(os.path.join(out_dir, f"{prefix}.ids.rank{r}.int64.mmap"), mode="r", dtype=np.int64,   shape=(n,)))
    return vecs, sums, ids, d, d_sum

@torch.no_grad()
def retrieval_recall_at_k(query_vec, db_vec, query_ids, db_ids, device, ks=(1,5,10), q_bs=4096, db_bs=65536):
    N = query_vec.size(0)
    recalls = {f"R@{k}": 0 for k in ks}
    for s in tqdm(range(0, N, q_bs), desc="Retrieval"):
        q = query_vec[s:s+q_bs].to(device).float()
        sims_all, idx_all = [], []
        M = db_vec.size(0)
        for t in range(0, M, db_bs):
            keys = db_vec[t:t+db_bs].to(device).float()
            sim = q @ keys.t()
            tk = min(max(ks), sim.size(1))
            s_k, i_k = sim.topk(k=tk, dim=1, largest=True, sorted=False)
            sims_all.append(s_k); idx_all.append(i_k + t)
            del keys, sim
        sims_cat = torch.cat(sims_all, 1)
        idx_cat  = torch.cat(idx_all,  1)
        tk = max(ks)
        _, rel = sims_cat.topk(k=tk, dim=1, largest=True, sorted=False)
        idxK = idx_cat.gather(1, rel)
        top_ids = db_ids[idxK.cpu()]
        gtid = query_ids[s:s+q.size(0)].unsqueeze(1)
        for K in ks:
            hits = (top_ids[:, :K] == gtid).any(dim=1).sum().item()
            recalls[f"R@{K}"] += int(hits)
        torch.cuda.synchronize()
    for K in ks: recalls[f"R@{K}"] = 100.0 * recalls[f"R@{K}"] / N
    return recalls

@torch.no_grad()
def retrieval_q_vs_db_shards(q_vec, q_ids, db_shards, db_id_shards, device, ks=(1,5,10), q_bs=4096, db_bs=32768):
    db_ids_dev = [torch.from_numpy(np.array(mm, copy=False)).to(device) for mm in db_id_shards]
    totals = {f"R@{k}": 0 for k in ks}; Nq = q_vec.size(0)
    for s in tqdm(range(0, Nq, q_bs), desc="Retrieval (q vs sharded db)"):
        q = q_vec[s:s+q_bs].to(device).float()
        sims_all, idx_all, shard_all = [], [], []
        for r_d, db_mm in enumerate(db_shards):
            M = db_mm.shape[0]
            for t in range(0, M, db_bs):
                keys = torch.from_numpy(np.array(db_mm[t:t+db_bs], copy=False)).to(device).float()
                sim  = q @ keys.t()
                tk   = min(max(ks), sim.size(1))
                s_k, i_k = sim.topk(k=tk, dim=1, largest=True, sorted=False)
                sims_all.append(s_k); idx_all.append(i_k + t)
                shard_all.append(torch.full_like(i_k, r_d))
        sims  = torch.cat(sims_all, 1)
        idx   = torch.cat(idx_all,   1)
        shard = torch.cat(shard_all, 1)
        tk = max(ks)
        _, rel = sims.topk(k=tk, dim=1, largest=True, sorted=False)
        idxK   = idx.gather(1, rel)
        shardK = shard.gather(1, rel)
        top_ids = torch.empty_like(idxK)
        for r_d, ids_dev in enumerate(db_ids_dev):
            mask = (shardK == r_d)
            if mask.any(): top_ids[mask] = ids_dev[idxK[mask]]
        gt = q_ids[s:s+q.size(0)].unsqueeze(1)
        for K in ks:
            hits = (top_ids[:, :K].cpu() == gt).any(dim=1).sum().item()
            totals[f"R@{K}"] += int(hits)
        torch.cuda.synchronize()
    for K in ks: totals[f"R@{K}"] = 100.0 * totals[f"R@{K}"] / Nq
    return totals
