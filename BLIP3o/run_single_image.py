import os, torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Dict, Any
from transformers import AutoProcessor
from blip3o.model import blip3oQwenForCausalLM, blip3oQwenConfig
from blip3o import conversation as conversation_lib

# --- use the SAME helpers as your train/eval ---
from blip3o.train.train import (
    smart_tokenizer_and_embedding_resize,
    preprocess_multimodal, preprocess,
    DataCollatorForSupervisedDataset,
)

from safetensors.torch import load_file as load_safetensors
import glob, json
from typing import Tuple
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Any, List

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers.image_processing_utils import BatchFeature
from transformers.image_transforms import convert_to_rgb
import numpy as np
import rembg



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
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    image_folder: Optional[str] = field(default=None)
    data_type: Optional[str] = field(default="mix")

    eval_coco_root: Optional[str] = None     # e.g. "/path/to/coco" (has annotations/, val2017/)
    eval_coco_split: str = "val2017"         # "train2017" or "val2017"

    eval_mapper_image_folder: Optional[str] = None  # folder of .tar shards for mapper eval
    eval_mapper_num_samples: Optional[int] = None   # cap N examples (None = all)

    max_seq_length: int = 512
    gen_image_processor: Any = None   
    image_processor: Any = None    
    n_query: Optional[int] = None
    n_und_query: Optional[int] = None


def preprocess_image(input: Image.Image):
        """
        Your original alpha-aware background handling + bbox crop (unchanged),
        then run the cropped result through self.model_transform (no normalization).
        """
        model_transform = transforms.Compose([
            convert_to_rgb,
            transforms.Resize(518, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # --- begin: your logic, unchanged except we skip the 518x518 resize ---
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize((int(input.width * scale), int(input.height * scale)),
                                    Image.Resampling.LANCZOS)
            
            rembg_session = rembg.new_session('u2net')
            output = rembg.remove(input, session=rembg_session)

        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore

        # (skip: output = output.resize((518, 518), Image.Resampling.LANCZOS))

        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        # --- end: your logic ---

        # now apply your transform (no normalization)
        transformed_image = [model_transform(output).numpy()]

        data = {"pixel_values": transformed_image}

        return BatchFeature(data=data, tensor_type=return_tensors)

# If you want the robust checkpoint loader from eval:
#   from your_eval_file import load_weights
# Here’s a small inlined variant that handles dir/file (simplified):
def _load_snapshot_dir(ckpt_dir: str) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file as load_safetensors
    import glob, json
    sd = {}
    st = os.path.join(ckpt_dir, "model.safetensors")
    if os.path.isfile(st):
        sd.update(load_safetensors(st, device="cpu"))
    else:
        idx = os.path.join(ckpt_dir, "model.safetensors.index.json")
        if os.path.isfile(idx):
            with open(idx, "r") as f: index = json.load(f)
            shard_files = sorted(set(index.get("weight_map", {}).values()))
            for fname in shard_files:
                sd.update(load_safetensors(os.path.join(ckpt_dir, fname), device="cpu"))
        else:
            shard_paths = sorted(glob.glob(os.path.join(ckpt_dir, "model-*-of-*.safetensors")))
            if shard_paths:
                for p in shard_paths:
                    sd.update(load_safetensors(p, device="cpu"))
            else:
                pt = os.path.join(ckpt_dir, "pytorch_model.bin")
                if os.path.isfile(pt):
                    sd.update(torch.load(pt, map_location="cpu"))
                else:
                    raise FileNotFoundError(f"No weights found in {ckpt_dir}")
    # merge projector bins if present
    for extra in ("mm_projector.bin", "gen_projector.bin"):
        p = os.path.join(ckpt_dir, extra)
        if os.path.isfile(p):
            sd.update(torch.load(p, map_location="cpu"))
    # strip DDP "module."
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd

def load_weights_exact(model, ckpt_path: str, strict: bool = False) -> Tuple[list, list]:
    if os.path.isdir(ckpt_path):
        sd = _load_snapshot_dir(ckpt_path)
    else:
        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file as load_safetensors
            sd = load_safetensors(ckpt_path, device="cpu")
        else:
            sd = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    print(f"Loaded. Missing={len(missing)} Unexpected={len(unexpected)} strict={strict}")
    if missing: print("  Missing:", missing)
    if unexpected: print("  Unexpected:", unexpected)
    return missing, unexpected


class SingleT2IDataset(Dataset):
    def __init__(self, image_path: str, caption: str, tokenizer, data_args: DataArguments):
        self.image_path = image_path
        self.caption = caption
        self.tokenizer = tokenizer
        self.da = data_args  # must have .gen_image_processor, .image_aspect_ratio, .n_query, .n_und_query, is_multimodal=True

    def __len__(self): return 1

    def __getitem__(self, idx) -> Dict[str, Any]:
        img = Image.open(self.image_path).convert("RGB")

        # Conversations exactly like train T2I:
        conv = [
            {"from": "human", "value": f"Please generate image based on the following caption: {self.caption}"},
            {"from": "gpt",   "value": "<image>"},
        ]

        # preprocess (same as train/eval)
        sources, inst_type = preprocess_multimodal([conv], self.da)
        assert inst_type == "gen", "Expected 'gen' for T2I (<image> on assistant side)."

        tokd = preprocess(sources, self.tokenizer, has_image=True)

        # gen_image preprocessing uses the gen_vision_tower image processor (same as train T2I/I2I)
        gen_px = self.da.gen_image_processor.preprocess([img], return_tensors="pt")["pixel_values"]
        # gen_px = preprocess_image(img)["pixel_values"]

        return {
            "input_ids": tokd["input_ids"][0],
            "labels":    tokd["labels"][0],
            "gen_image": gen_px,     # (B=1, C, H, W)
            "ids":       "single",   # mimic dataset ID
        }



# ==== Minimal end-to-end example ====
def run_single_image_forward(
    ckpt: str,
    image_path: str,
    caption: str,
    device: str = "cuda",
):
    # 0) Model + vision init (same order as eval)
    print("Loading config...")
    config = blip3oQwenConfig.from_pretrained(ckpt)
    print("Initializing model...")
    model = blip3oQwenForCausalLM(config)

    model_args = ModelArguments()

    data_args = DataArguments()
    data_args.n_query = model_args.n_query
    data_args.n_und_query = model_args.n_und_query
    data_args.image_aspect_ratio = model_args.image_aspect_ratio

    print("Initializing vision modules...")
    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=None)

    print("Loading weights...")
    load_weights_exact(model, ckpt, strict=False)

    model.eval().to(device)
    model.config.use_cache = False

    # freeze like eval (not required for fwd, but mirrors eval)
    for (_, p) in model.get_model().named_parameters(): p.requires_grad = False
    for (_, p) in model.visual.named_parameters():      p.requires_grad = False
    for (_, p) in model.lm_head.named_parameters():     p.requires_grad = False
    

    # gen vision tower/device/dtype (mirrors eval)
    gen_vision_tower = model.get_gen_vision_tower()
    gen_vision_tower.to(
        dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16,
        device=device,
    ).requires_grad_(False)

    # 1) Tokenizer/processor (same as eval)
    try:
        processor = AutoProcessor.from_pretrained(ckpt)
        tokenizer = processor.tokenizer
    except Exception:
        tokenizer = AutoProcessor.from_pretrained(ckpt)
    tokenizer.model_max_length = 512

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(
                pad_token="<pad>",
                additional_special_tokens=["[IMG]", "[/IMG]", "<image>"],
            ),
            tokenizer=tokenizer,
            model=model,
        )
    elif not "<image>" in tokenizer.get_added_vocab():
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

    data_args.gen_image_processor = gen_vision_tower.image_processor
    data_args.image_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct").image_processor

    model.config.image_aspect_ratio = model_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    model.config.freeze_mm_mlp_adapter = False

    model.predict_summary_token = model_args.predict_summary_token
    model.predict_dino_grid = model_args.predict_dino_grid
    model.config.num_register_tokens = model_args.num_register_tokens

    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.config.pad_token_id = tokenizer.pad_token_id


    # 4) Make single-image dataset + collator + loader (exactly like train/eval)
    ds = SingleT2IDataset(image_path=image_path, caption=caption, tokenizer=tokenizer, data_args=data_args)
    collator = DataCollatorForSupervisedDataset(n_query=data_args.n_query, tokenizer=tokenizer)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collator, num_workers=0)

    # 5) Call your eval with same signature
    with torch.no_grad():
        metrics = model.evaluate_mapper_dino(
            dataloader=dl,
            device=device,
            num_inference_steps=30,
            k_neigh=10,
            max_eval_samples=1,     # only 1 image
        )

    print("=== EVAL (single image) METRICS ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}" if isinstance(v, (int, float)) else f"{k}: {v}")
    return metrics

# ---- Example call ----
if __name__ == "__main__":
    ckpt_dir = "/path/to/your_checkpoint"
    image    = "assets/avocado_chair.png"
    caption  = "a chair looking like an avocado, with black background"
    run_single_image_forward(ckpt=ckpt_dir, image_path=image, caption=caption)
ackground"
    run_single_image_forward(ckpt=ckpt_dir, image_path=image, caption=caption)
