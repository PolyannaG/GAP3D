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



os.environ['SPCONV_ALGO'] = 'native'      

import imageio
from PIL import Image

from pathlib import Path, sys
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


from TRELLIS.trellis.pipelines import BlipTrellisImageTo3DPipeline, TrellisTextTo3DPipeline
from TRELLIS.trellis.utils import render_utils, postprocessing_utils


def load_image_embedding_from_pt(path: str, device: str = "cuda"):
    """
    Expects a tensor or a dict containing 'image_embeds' saved with torch.save.
    Ensures batch dimension and moves to correct device / dtype.
    """
    ckpt = torch.load(path, map_location=device)

    # Handle either raw tensor or dict
    if isinstance(ckpt, torch.Tensor):
        image_embeds = ckpt
    
    # Move to device + dtype
    image_embeds = image_embeds.to(
        device=device,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )

    print(f"Loaded image_embeds from {path} with shape {image_embeds.shape}")
    return image_embeds.unsqueeze(0) if image_embeds.ndim == 2 else image_embeds


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


    base_dir = "run_outputs/embeddings"
    emb_path = "/path/to/precomputed_embeddings.pt"
    caption = "Claw hammer with bifurcated claw and blue grip."

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trellis_pipeline = BlipTrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    trellis_pipeline.cuda()

    
    print(f"Running Trellis for caption: {caption}")

    image_embeds = load_image_embedding_from_pt(emb_path, device=device)

    save_dir = f"{base_dir}"
    os.makedirs(save_dir, exist_ok=True)

    trellis_run_single_image_forward(
        pipeline=trellis_pipeline,
        image_embeds=image_embeds,
        save_dir=save_dir,
        seed=42,
        device=device,
    )

    # (optional) Save caption again if you like:
    with open(os.path.join(save_dir, "caption.txt"), "w") as f:
        f.write(caption)

a17c9c3f0dd9eb82cd2f417c39/0.pt"
        # caption = "Detailed smartphone model with a matte green body, realistic camera texture, and a screen showing a cloud wallpaper, reflecting a modern and sophisticated aesthetic."
    caption = "Claw hammer with bifurcated claw and blue grip."

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trellis_pipeline = BlipTrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    trellis_pipeline.cuda()

    
    print(f"Running Trellis for caption: {caption}")

    image_embeds = load_image_embedding_from_pt(emb_path, device=device)

    save_dir = f"{base_dir}"
    os.makedirs(save_dir, exist_ok=True)

    trellis_run_single_image_forward(
        pipeline=trellis_pipeline,
        image_embeds=image_embeds,
        save_dir=save_dir,
        seed=42,
        device=device,
    )

    # (optional) Save caption again if you like:
    with open(os.path.join(save_dir, "caption.txt"), "w") as f:
        f.write(caption)

