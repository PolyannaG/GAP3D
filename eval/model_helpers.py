from blip3o.model import blip3oQwenForCausalLM, blip3oQwenConfig
from blip3o import conversation as conversation_lib
from blip3o.train.train import (
    smart_tokenizer_and_embedding_resize,
)
from blip3o.constants import IMAGE_TOKEN_IDX
from blip3o.train.train import (
    smart_tokenizer_and_embedding_resize,
    preprocess_multimodal, preprocess,
    DataCollatorForSupervisedDataset,
)
from safetensors.torch import load_file as load_safetensors
from transformers import AutoProcessor
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, Tuple
import torch
import os
import glob
import json
from PIL import Image

# ====================== BLIP3-o ======================

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="qwen")
    freeze_backbone: bool = field(default=True)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    gen_vision_tower: Optional[str] = field(default="dinov2_vitl14_register")
    mm_vision_select_layer: Optional[int] = field(default=-2)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    pretrain_gen_mlp_adapter: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    gen_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    n_query: Optional[int] = field(default=64)  
    n_und_query: Optional[int] = field(default=0) 
    gen_pooling: Optional[str] = field(default="None")
    predict_summary_token: bool = field(default=False)
    predict_dino_grid: bool = field(default=True)
    num_register_tokens: Optional[int] = field(default=4)
    image_aspect_ratio: str = "square"

@dataclass
class DataArguments:
    # data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    # lazy_preprocess: bool = False
    is_multimodal: bool = True
    # image_folder: Optional[str] = field(default=None)
    # data_type: Optional[str] = field(default="mix")
    mm_use_im_patch_token: bool = field(default=False)
    max_seq_length: int = 512
    gen_image_processor: Any = None   
    image_processor: Any = None    
    n_query: Optional[int] = field(default=64) 
    n_und_query: Optional[int] = field(default=0)
    image_aspect_ratio: str = "square"

def _load_snapshot_dir_blip(ckpt_dir: str) -> Dict[str, torch.Tensor]:
    sd={}
    st=os.path.join(ckpt_dir,"model.safetensors")
    if os.path.isfile(st):
        sd.update(load_safetensors(st, device="cpu"))
    else:
        idx=os.path.join(ckpt_dir,"model.safetensors.index.json")
        if os.path.isfile(idx):
            with open(idx,"r") as f: index=json.load(f)
            for fname in sorted(set(index.get("weight_map",{}).values())):
                sd.update(load_safetensors(os.path.join(ckpt_dir,fname), device="cpu"))
        else:
            shard_paths=sorted(glob.glob(os.path.join(ckpt_dir,"model-*-of-*.safetensors")))
            if shard_paths:
                for p in shard_paths: sd.update(load_safetensors(p, device="cpu"))
            else:
                pt=os.path.join(ckpt_dir,"pytorch_model.bin")
                if os.path.isfile(pt):
                    sd.update(torch.load(pt, map_location="cpu"))
                else:
                    raise FileNotFoundError(f"No weights found in {ckpt_dir}")
    for extra in ("mm_projector.bin","gen_projector.bin"):
        p=os.path.join(ckpt_dir,extra)
        if os.path.isfile(p): 
            sd.update(torch.load(p, map_location="cpu"))
    if any(k.startswith("module.") for k in sd.keys()):
        sd={k.replace("module.","",1):v for k,v in sd.items()}
    return sd

def load_weights_exact_blip(model, ckpt_path: str, strict: bool=False) -> Tuple[list,list]:
    if os.path.isdir(ckpt_path):
        sd=_load_snapshot_dir_blip(ckpt_path)
    else:
        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file as load_safetensors
            sd=load_safetensors(ckpt_path, device="cpu")
        else:
            sd=torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    print(f"[BLIP] Loaded. Missing={len(missing)} Unexpected={len(unexpected)} strict={strict}")
    if missing: 
        print("  Missing:", missing)
    if unexpected: 
        print("  Unexpected:", unexpected)
    return missing, unexpected

class BlipTextEmbedder:
    """
    Minimal BLIP3-o text embedder for joint text→3D.
    Produces a single sequence embedding tensor from the last hidden state.
    """
    def __init__(self, ckpt: str, device: str = "cuda", model_args: ModelArguments = None, data_args: DataArguments = None):
        self.device = device
        print("[BLIP] Loading config...")
        config = blip3oQwenConfig.from_pretrained(ckpt)
        print("[BLIP] Initializing model...")
        self.model = blip3oQwenForCausalLM(config)
        
        # BLIP3-o also has vision modules; not needed for pure text, but safe to init tokenizer etc.
        self.model.get_model().initialize_vision_modules(model_args=model_args, fsdp=None)

        print("[BLIP] Loading weights...")
        load_weights_exact_blip(self.model, ckpt, strict=False)

        self.model.eval().to(device)
        self.model.config.use_cache = False
        for (_, p) in self.model.get_model().named_parameters(): 
            p.requires_grad = False
        for (_, p) in self.model.visual.named_parameters():      
            p.requires_grad = False
        for (_, p) in self.model.lm_head.named_parameters():     
            p.requires_grad = False
        
        gen_vision_tower = self.model.get_gen_vision_tower()
        gen_vision_tower.to(
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
            device=device,
        ).requires_grad_(False)
        
        try:
            processor = AutoProcessor.from_pretrained(ckpt)
            tokenizer = processor.tokenizer
        except Exception:
            print("[BLIP] Loading tokenizer separately...")
            tokenizer = AutoProcessor.from_pretrained(ckpt)
        
        tokenizer.model_max_length = data_args.max_seq_length
        if tokenizer.pad_token is None:
            print("[BLIP] Adding pad token to tokenizer...")
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(
                    pad_token="<pad>",
                    additional_special_tokens=["[IMG]", "[/IMG]", "<image>"],
                ),
                tokenizer=tokenizer,
                model=model,
            )
        elif not "<image>" in tokenizer.get_added_vocab():
            print("[BLIP] Adding image special tokens to tokenizer...")
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(additional_special_tokens=["[IMG]", "[/IMG]", "<image>"]),
                tokenizer=tokenizer,
                model=model,
            )

        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["llama3"]
        print(f"Using conversation format: {conversation_lib.default_conversation.version}")

        self.tokenizer = tokenizer
        self.collator = DataCollatorForSupervisedDataset(n_query=data_args.n_query, tokenizer=self.tokenizer)

        data_args.gen_image_processor = gen_vision_tower.image_processor
        data_args.image_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct").image_processor

        # print("[BLIP] Setting model config...")
        # print(self.model.config.image_aspect_ratio, model_args.image_aspect_ratio)
        # self.model.config.image_aspect_ratio = model_args.image_aspect_ratio

        # print(self.model.config.tokenizer_padding_side, tokenizer.padding_side)
        # self.model.config.tokenizer_padding_side = tokenizer.padding_side

        # print(self.model.config.tokenizer_model_max_length, tokenizer.model_max_length)
        # self.model.config.tokenizer_model_max_length = tokenizer.model_max_length

        # print(self.model.config.tune_mm_mlp_adapter, model_args.tune_mm_mlp_adapter)
        # self.model.config.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter

        # print(self.model.config.freeze_mm_mlp_adapter)
        # self.model.config.freeze_mm_mlp_adapter = False

        self.model.predict_summary_token = model_args.predict_summary_token
        self.model.predict_dino_grid = model_args.predict_dino_grid
        
        # print(self.model.config.num_register_tokens, model_args.num_register_tokens)
        # self.model.config.num_register_tokens = model_args.num_register_tokens

        # print(self.model.config.mm_use_im_start_end, model_args.mm_use_im_start_end)
        # self.model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end

        # print(self.model.config.mm_use_im_patch_token, model_args.mm_use_im_patch_token)
        # self.model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token

        # print(self.model.config.pad_token_id, tokenizer.pad_token_id)
        # self.model.config.pad_token_id = tokenizer.pad_token_id

        self.data_args = data_args
        self.model_args = model_args


    @torch.no_grad()
    def get_image_embeds(self, caption: str, steps: int = 50) -> torch.Tensor:
        # Conversation like train T2I
        conv = [
            {"from": "human", "value": f"Please generate image based on the following caption: {caption}"},
            {"from": "gpt",   "value": "<image>"},
        ]
        sources, inst_type = preprocess_multimodal([conv], self.data_args)
        assert inst_type == "gen", "Expected 'gen' for T2I (<image> on assistant side)."
        tokd = preprocess(sources, self.tokenizer, has_image=True)

        sample = {
            "input_ids": tokd["input_ids"][0],
            "labels":    tokd["labels"][0],
            "ids":       "single",
        }
        batch = self.collator([sample])  # no DataLoader needed
        # move to device
        for k in list(batch.keys()):
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(self.device)
        input_ids      = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels         = batch["labels"]
        i_s_pos        = batch["i_s_pos"]

        latent_queries = self.model.get_model().latent_queries.repeat(input_ids.size(0), 1, 1)  # [B,NQ,H]
        H = latent_queries.shape[-1]
        latent_queries = latent_queries.contiguous().view(-1, H)          # [B*NQ, H]
        image_idx = (input_ids == IMAGE_TOKEN_IDX)
        output_indicator = (labels != -100)
        text_embeds = self.model.get_model().embed_tokens(input_ids)           # [B,T,H]
        gen_img_idx = torch.logical_and(output_indicator, image_idx)
        text_embeds = text_embeds.clone()
        text_embeds[gen_img_idx] = latent_queries
        labels = labels.clone()
        labels[image_idx] = -100

        outputs = self.model.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=text_embeds,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )
        hidden_states = outputs[0]    # [B, T, H]
        img_hidden_states = []
        for b_ix in range(hidden_states.shape[0]):
            img_hidden_states.append(hidden_states[b_ix, i_s_pos[b_ix]:i_s_pos[b_ix]+self.data_args.n_query, :])
        img_hidden_states = torch.stack(img_hidden_states, dim=0)        # [B, NQ, H]
        img_hidden_states = self.model.get_model().down_projector(img_hidden_states)

        B = input_ids.size(0)
        assert gen_img_idx.sum().item() == B * self.data_args.n_query, f"gen_img_idx {gen_img_idx.sum().item()} vs B*n_query {B*self.data_args.n_query}"
        assert img_hidden_states.shape[1] == self.data_args.n_query, f"Got {img_hidden_states.shape[1]} latent slots, expected {self.data_args.n_query}"

        if not hasattr(self.model, "_inference_scheduler"):
            from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
            self.model._inference_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                "Alpha-VLLM/Lumina-Next-SFT-diffusers", subfolder="scheduler"
            )
        scheduler = self.model._inference_scheduler 

        pred_sum, pred_regs, pred_patch = self.model.sample_images_cfg_dino(
            img_hidden_states,
            scheduler=scheduler,
            num_inference_steps=steps,
            num_images_per_prompt=1,
            generator=None,
        )

        # assert pred_sum.shape[1] == 1024, f"Expected summary token dim 1024, got {pred_sum.shape[1]}"
        # assert pred_regs.shape[1] == 4 and pred_regs.shape[2] == 1024, f"Expected 4 reg tokens of dim 1024, got {pred_regs.shape}"
        # assert pred_patch.shape[1] == 1024 and pred_patch.shape[2] == 37 and pred_patch.shape[3] == 37, f"Expected latent tokens of dim [B, 1024, 37, 37], got {pred_patch.shape}"

        # summary: [B, D]; regs: [B, 4, D]; patches: [B, S, H, W] where S==D
        B, S, H, W = pred_patch.shape
        pred_patch = pred_patch.permute(0, 2, 3, 1).reshape(B, H*W, S)
        summary_pred = pred_sum.unsqueeze(1)  # [B,1,D]
        image_embeds = torch.cat([summary_pred, pred_regs, pred_patch], dim=1)  # [B,1+4+H*W,D]
        return image_embeds

    def get_image_embeds_batch(self, batch, steps: int = 50) -> torch.Tensor:
        if not hasattr(self.model, "_inference_scheduler"):
            from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
            self.model._inference_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                "Alpha-VLLM/Lumina-Next-SFT-diffusers", subfolder="scheduler"
            )
        scheduler = self.model._inference_scheduler

        input_ids      = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels         = batch["labels"]
        i_s_pos        = batch["i_s_pos"]

        latent_queries = self.model.get_model().latent_queries.repeat(input_ids.size(0), 1, 1)  # [B,NQ,H]
        H = latent_queries.shape[-1]
        latent_queries = latent_queries.contiguous().view(-1, H)          # [B*NQ, H]
        image_idx = (input_ids == IMAGE_TOKEN_IDX)
        output_indicator = (labels != -100)
        text_embeds = self.model.get_model().embed_tokens(input_ids)           # [B,T,H]
        gen_img_idx = torch.logical_and(output_indicator, image_idx)
        text_embeds = text_embeds.clone()
        text_embeds[gen_img_idx] = latent_queries
        labels = labels.clone()
        labels[image_idx] = -100

        outputs = self.model.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=text_embeds,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )
        hidden_states = outputs[0]    # [B, T, H]
        img_hidden_states = []
        for b_ix in range(hidden_states.shape[0]):
            img_hidden_states.append(hidden_states[b_ix, i_s_pos[b_ix]:i_s_pos[b_ix]+self.data_args.n_query, :])
        img_hidden_states = torch.stack(img_hidden_states, dim=0)        # [B, NQ, H]
        img_hidden_states = self.model.get_model().down_projector(img_hidden_states)

        B = input_ids.size(0)
        assert gen_img_idx.sum().item() == B * self.data_args.n_query, f"gen_img_idx {gen_img_idx.sum().item()} vs B*n_query {B*self.data_args.n_query}"
        assert img_hidden_states.shape[1] == self.data_args.n_query, f"Got {img_hidden_states.shape[1]} latent slots, expected {self.data_args.n_query}"

        if self.model_args.predict_dino_grid:
            pred_sum, pred_regs, pred_lat  = self.model.sample_images_cfg_dino(
                img_hidden_states,
                scheduler=scheduler,
                num_inference_steps=steps,
                num_images_per_prompt=1,
                generator=None,
            )
            return pred_sum, pred_regs, pred_lat
        else:
            pred_lat = self.model.sample_images(
                img_hidden_states,
                scheduler=scheduler,
                num_inference_steps=steps,
                num_images_per_prompt=1,
                generator=None,
            )
            return pred_lat




