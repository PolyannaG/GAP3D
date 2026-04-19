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

from torchvision.transforms.functional import InterpolationMode

transform_und_images = transforms.Compose([
    transforms.Resize(448, interpolation=InterpolationMode.BICUBIC, antialias=True),
    transforms.CenterCrop(448),
])

import imageio
from PIL import Image

from pathlib import Path, sys
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


from TRELLIS.trellis.pipelines import BlipTrellisImageTo3DPipeline, TrellisTextTo3DPipeline
from TRELLIS.trellis.utils import render_utils, postprocessing_utils

from eval.model_helpers import load_weights_exact_blip, ModelArguments, DataArguments



class SingleI2IEditDataset(Dataset):
    """
    Single example: (input image + edit text) -> image latents.

    Conditioning image goes through the understanding tower (und_image + grid_thw),
    while text controls the edit. We still let the collator append latent-query tokens
    so generate_embeddings_dino can work the same way.
    """
    def __init__(self, image_path: str, edit_text: str, tokenizer, data_args: DataArguments):
        self.image_path = image_path
        self.edit_text = edit_text
        self.tokenizer = tokenizer
        self.da = data_args

    def __len__(self):
        return 1

    def __getitem__(self, idx) -> Dict[str, Any]:
        img = Image.open(self.image_path).convert("RGB")

        # --- 1) Build an image+text editing conversation ---
        #
        # IMPORTANT: <image> appears on the HUMAN side here.
        # That means preprocess_multimodal will treat this as an
        # "understanding" image (und_image).
        conv = [
            {
                "from": "human",
                "value": (
                    "Please generate image based on the following caption: Create a precise replica of: "
                    "<image>\n"
                    "but: "    
                    f"{self.edit_text}"
                    
                ),
            },
            {
                "from": "gpt",
                # We don't actually *need* <image> on the assistant side
                # for generation because the collator appends the latent
                # image tokens anyway. Keeping it as "" avoids confusing
                # preprocess_multimodal.
                "value": "",
            },
        ]

        # --- 2) Tokenize text, same path as training ---
        sources, inst_type = preprocess_multimodal([conv], self.da)
        # inst_type will be "und" because <image> is only on human side
        tokd = preprocess(sources, self.tokenizer, has_image=True)

                # === DEBUG: print final prompt + tokens ===
        input_ids = tokd["input_ids"][0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.tolist())

        print("\n=== FINAL TEXT AFTER preprocess_multimodal ===")
        print(sources[0][0]["value"])
        print("=============================================")

        print("\n=== TOKENS ===")
        print(tokens)

        # Highlight special image tokens
        from blip3o.constants import IMAGE_TOKEN_IDX, UND_IMAGE_TOKEN_IDX
        image_pos = (input_ids == IMAGE_TOKEN_IDX).nonzero().flatten().tolist()
        und_pos   = (input_ids == UND_IMAGE_TOKEN_IDX).nonzero().flatten().tolist()

        print("IMAGE_TOKEN_IDX positions:", image_pos)
        print("UND_IMAGE_TOKEN_IDX positions:", und_pos)
        print("Total UND slots:", len(und_pos))


        # --- 3) Conditioning image for the understanding tower ---
        resized = transform_und_images(img)
        image_inputs = self.da.image_processor([resized], return_tensors="pt")
        und_px   = image_inputs.pixel_values      # [1, C, H, W]
        grid_thw = image_inputs.image_grid_thw    # [1, 3]

        gen_px = self.da.gen_image_processor.preprocess([img], return_tensors="pt")["pixel_values"]

        return {
            "input_ids": tokd["input_ids"][0],
            "labels":    tokd["labels"][0],
            "und_image": und_px,          # will be batched by collator
            "grid_thw":  grid_thw,        # same
            "ids":       "single_edit",   # fake ID, fine for inference
            "gen_image": gen_px,     # (B=1, C, H, W)
        }





# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.




# ==== Minimal end-to-end example ====
def blip_init(
    ckpt: str,
    device: str = "cuda",
):
    # 0) Model + vision init (same order as eval)
    print("Loading config...")
    config = blip3oQwenConfig.from_pretrained(ckpt)
    print("Initializing model...")
    model = blip3oQwenForCausalLM(config)

    model_args = ModelArguments(n_und_query=256, model_name_or_path=ckpt)
    data_args = DataArguments(n_und_query=256)
   
    print("Initializing vision modules...")
    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=None)

    print("Loading weights...")
    load_weights_exact_blip(model, ckpt, strict=False)

    model.eval()
    model.to(device=device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16)
    model.config.use_cache = False

    # freeze like eval (not required for fwd, but mirrors eval)
    for (_, p) in model.get_model().named_parameters(): 
        p.requires_grad = False
    for (_, p) in model.visual.named_parameters():      
        p.requires_grad = False
    for (_, p) in model.lm_head.named_parameters():     
        p.requires_grad = False
    

    # gen vision tower/device/dtype (mirrors eval)
    gen_vision_tower = model.get_gen_vision_tower()
    gen_vision_tower.to(
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
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

    model.predict_summary_token = model_args.predict_summary_token
    model.predict_dino_grid = model_args.predict_dino_grid

    return model, tokenizer, data_args, device

def generate_edit_image_embeddings(
    model,
    tokenizer,
    data_args,
    image_path: str,
    edit_text: str,
    device: str = "cuda",
    steps: int = 50,
):
    """
    Image + text editing version of generate_image_embeddings.

    - image_path: conditioning image
    - edit_text:  textual instruction for how to edit the image
    """
    # 1) Dataset & collator
    ds = SingleI2IEditDataset(
        image_path=image_path,
        edit_text=edit_text,
        tokenizer=tokenizer,
        data_args=data_args,
    )
    collator = DataCollatorForSupervisedDataset(
        n_query=data_args.n_query,
        tokenizer=tokenizer,
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collator, num_workers=0)

    batch = next(iter(dl))

    # 2) Move tensors to device
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    # 3) Call the same BLIP3-o helper, but now batch has und_image + grid_thw
    with torch.no_grad():
        (summary_embed, reg_embeds, patch_embeds), metrics = model.generate_embeddings_dino(
            batch=batch,
            device=device,
            num_inference_steps=steps,
            cfg=True,
        )

    # 4) Pack latents into [B, 1+4+H*W, D] like your T2I case
    B, S, H, W = patch_embeds.shape
    patch_embeds = patch_embeds.permute(0, 2, 3, 1).reshape(B, H * W, S)
    summary_pred = summary_embed.unsqueeze(1)          # [B, 1, D]
    image_embeds = torch.cat([summary_pred, reg_embeds, patch_embeds], dim=1)

    print("Edit image embeds shape:", image_embeds.shape)
    print("=== EDIT (single image) METRICS ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}" if isinstance(v, (int, float)) else f"{k}: {v}")

    return image_embeds



def trellis_run_single_image_forward(pipeline, image_embeds, save_dir: str = None, device: str = "cuda", seed: int = 42):
    # Run the pipeline
    outputs = pipeline.run(
        seed=seed,
        image_embeds=image_embeds, # pass the image embeddings from BLIP3-o
    )
    # Render the outputs
    video = render_utils.render_video(outputs['gaussian'][0])['color']
    imageio.mimsave(f"{save_dir}/gs.mp4", video, fps=30)
    video = render_utils.render_video(outputs['radiance_field'][0])['color']
    imageio.mimsave(f"{save_dir}/rf.mp4", video, fps=30)
    video = render_utils.render_video(outputs['mesh'][0])['normal']
    imageio.mimsave(f"{save_dir}/mesh.mp4", video, fps=30)

    # GLB files can be extracted from the outputs
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        # Optional parameters
        simplify=0.95,          # Ratio of triangles to remove in the simplification process
        texture_size=1024,      # Size of the texture used for the GLB
    )
    glb.export(f"{save_dir}/model.glb")

    # Save Gaussians as PLY files
    outputs['gaussian'][0].save_ply(f"{save_dir}/model.ply")

def trellis_run_single_text_forward(pipeline, caption: str, save_dir: str = None, device: str = "cuda", seed: int = 42):
    # Run the pipeline
    outputs = pipeline.run(
        caption,
        seed=seed,
    )
    # Render the outputs
    video = render_utils.render_video(outputs['gaussian'][0])['color']
    imageio.mimsave(f"{save_dir}/gs.mp4", video, fps=30)
    video = render_utils.render_video(outputs['radiance_field'][0])['color']
    imageio.mimsave(f"{save_dir}/rf.mp4", video, fps=30)
    video = render_utils.render_video(outputs['mesh'][0])['normal']
    imageio.mimsave(f"{save_dir}/mesh.mp4", video, fps=30)

    # GLB files can be extracted from the outputs
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        # Optional parameters
        simplify=0.95,          # Ratio of triangles to remove in the simplification process
        texture_size=1024,      # Size of the texture used for the GLB
    )
    glb.export(f"{save_dir}/model.glb")

    # Save Gaussians as PLY files
    outputs['gaussian'][0].save_ply(f"{save_dir}/model.ply")


if __name__ == "__main__":
    ckpt_dir = "/path/to/your_checkpoint"
    base_dir = "run_outputs/editing"

    trellis_pipeline = BlipTrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    trellis_pipeline.cuda()

    edit_instruction = "Change from lego to normal metal."
    image = "assets/helicopter.png"
    # edit_instruction = "Make it red."
    # image = "assets/avocado_chair.png"
    model, tokenizer, data_args, device = blip_init(ckpt=ckpt_dir)

    image_embeds = generate_edit_image_embeddings(
        model=model,
        tokenizer=tokenizer,
        data_args=data_args,
        image_path=image,          # conditioning image
        edit_text=edit_instruction,
        device=device,
        steps=50,
    )

    save_dir = f"{base_dir}/helicopter_metal"
    os.makedirs(save_dir, exist_ok=True)
    trellis_run_single_image_forward(
        pipeline=trellis_pipeline,
        image_embeds=image_embeds,
        save_dir=save_dir,
        seed=42,
    )
    with open(f"{save_dir}/edit.txt", "w") as f:
        f.write(edit_instruction)
= f"{base_dir}/helicopter_metal"
    os.makedirs(save_dir, exist_ok=True)
    trellis_run_single_image_forward(
        pipeline=trellis_pipeline,
        image_embeds=image_embeds,
        save_dir=save_dir,
        seed=42,
    )
    with open(f"{save_dir}/edit.txt", "w") as f:
        f.write(edit_instruction)
xt", "w") as f:
        f.write(edit_instruction)
