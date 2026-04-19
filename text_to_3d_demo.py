import argparse
import os
from pathlib import Path

import torch

# Ensure local packages are importable when run from repo root
repo_root = Path(__file__).resolve().parent
import sys
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from TRELLIS.trellis.pipelines import BlipTrellisImageTo3DPipeline
from run.run_combined import (
    blip_init,
    generate_image_embeddings,
    trellis_run_single_image_forward,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal text-to-3D demo using BLIP3-o + TRELLIS-image-large."
    )
    parser.add_argument(
        "--blip_ckpt",
        type=str,
        required=True,
        help="Path or Hugging Face id for the BLIP3-o checkpoint (e.g., BLIP3o/BLIP3o-Model-4B or a fine-tuned checkpoint).",
    )
    parser.add_argument(
        "--trellis_ckpt",
        type=str,
        default="microsoft/TRELLIS-image-large",
        help="Path or Hugging Face id for the TRELLIS image model (default: microsoft/TRELLIS-image-large).",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the reference image used for BLIP conditioning (same role as in run/run_combined.py).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt describing the 3D asset to generate.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/text_to_3d_demo",
        help="Output directory for videos and GLB/PLY files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for TRELLIS sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This demo requires a CUDA GPU for TRELLIS and BLIP3-o.")

    os.environ.setdefault("SPCONV_ALGO", "native")

    device = "cuda"

    out_dir = Path(args.out_dir)

    # 1) Initialize BLIP3-o + tokenizer + data_args using the same helper as run/run_combined.py
    model, tokenizer, data_args, device = blip_init(ckpt=args.blip_ckpt, device=device)

    # 2) Initialize TRELLIS image-to-3D pipeline that accepts BLIP embeddings
    print(f"[Demo] Loading TRELLIS image model from {args.trellis_ckpt}...")
    trellis_pipe = BlipTrellisImageTo3DPipeline.from_pretrained(args.trellis_ckpt)
    trellis_pipe.cuda()

    # 3) Use the same generate_image_embeddings helper as run/run_combined.py
    image_embeds = generate_image_embeddings(
        model=model,
        tokenizer=tokenizer,
        data_args=data_args,
        image_path=args.image,
        caption=args.prompt,
        device=device,
        seed=args.seed,
    )

    # 4) Run 3D generation and save outputs using the shared helper
    trellis_run_single_image_forward(
        pipeline=trellis_pipe,
        image_embeds=image_embeds,
        save_dir=str(out_dir),
        device=device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
