import torch
import torch.nn as nn
import torch.nn.functional as F

from .factory import get_model_config
from .dino_processor import DinoV2ImageTrainProcessor
import torch.hub

import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class DinoVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.vision_tower_pretrained = args.vision_tower_pretrained
        self.config = get_model_config(vision_tower)

        if not delay_load:
            print(f"Loading DINOv2 ViT: {self.vision_tower_name}")
            self.load_model()
        elif getattr(args, "unfreeze_mm_vision_tower", False):
            print("Checkpoint contains vision tower weights, unfreezing...")
            self.load_model()
        elif hasattr(args, "mm_tunable_parts") and "mm_vision_tower" in args.mm_tunable_parts:
            print("Checkpoint contains vision tower weights, mm_vision_tower tunable.")
            self.load_model()
        else:
            self.cfg_only = self.config

    def load_model(self, device_map=None):
        print(f"Pretrained: {self.vision_tower_pretrained}")
        self.image_processor = DinoV2ImageTrainProcessor(
            self.config["vision_cfg"]["image_size"]
        )

        # Load DINOv2 backbone from torch hub
        dist_ok = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if dist_ok else 0
        is_rank0 = (rank == 0)

        if is_rank0:
            _ = torch.hub.load(
            "facebookresearch/dinov2",
            self.vision_tower_name,
            pretrained=True,
            )
        if dist_ok:
            dist.barrier()
        
        self.vision_tower = torch.hub.load(
            "facebookresearch/dinov2",
            self.vision_tower_name,
            pretrained=True,
        )

        # Freeze by default
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

        print(f"Loaded image processor: {self.image_processor}")

    def forward(self, images):
        feats = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype),
            is_training=True
        )['x_prenorm']
        feats = F.layer_norm(feats, feats.shape[-1:])
        feats = feats.to(images.dtype)
        return feats

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        if hasattr(self.vision_tower, 'device'):
            return self.vision_tower.device
        else:
            return next(self.vision_tower.parameters()).device

    @property
    def hidden_size(self):
        return self.config["vision_cfg"]["width"]

    @property
    def num_patches(self):
        return (self.config["vision_cfg"]["image_size"] // self.config["vision_cfg"]["patch_size"]) ** 2

    @property
    def num_patches_per_side(self):
        return self.config["vision_cfg"]["image_size"] // self.config["vision_cfg"]["patch_size"]

    @property
    def image_size(self):
        return self.config["vision_cfg"]["image_size"]


