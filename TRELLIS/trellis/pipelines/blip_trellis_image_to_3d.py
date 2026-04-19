from typing import *
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import rembg
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp



from torchvision.transforms import InterpolationMode
from transformers.image_transforms import convert_to_rgb


class BlipTrellisImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        self._init_image_cond_model(image_cond_model)

    @staticmethod
    def from_pretrained(
        path: str, 
        sparse_structure_flow_ckpt: str = None,
        slat_flow_ckpt: str = None,
    ) -> "TrellisImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
            sparse_structure_flow_ckpt (str): Optional path to a finetuned sparse structure flow model checkpoint (.pt file).
            slat_flow_ckpt (str): Optional path to a finetuned structured latent flow model checkpoint (.pt file).
        """
        pipeline = super(BlipTrellisImageTo3DPipeline, BlipTrellisImageTo3DPipeline).from_pretrained(path)
        new_pipeline = BlipTrellisImageTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_image_cond_model(args['image_cond_model'])

        # Load custom finetuned checkpoints if provided
        if sparse_structure_flow_ckpt is not None:
            new_pipeline.load_sparse_structure_flow_ckpt(sparse_structure_flow_ckpt)
        if slat_flow_ckpt is not None:
            new_pipeline.load_slat_flow_ckpt(slat_flow_ckpt)

        return new_pipeline
    
    def load_sparse_structure_flow_ckpt(self, ckpt_path: str) -> None:
        """
        Load a finetuned sparse structure flow model checkpoint.
        
        Args:
            ckpt_path (str): Path to the .pt checkpoint file (e.g., denoiser_ema0.9999_step0100000.pt)
        """
        print(f"Loading finetuned sparse structure flow model from: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location='cpu')
        self.models['sparse_structure_flow_model'].load_state_dict(state_dict)
        print("Successfully loaded finetuned sparse structure flow model!")
    
    def load_slat_flow_ckpt(self, ckpt_path: str) -> None:
        """
        Load a finetuned structured latent (SLAT) flow model checkpoint.
        
        Args:
            ckpt_path (str): Path to the .pt checkpoint file (e.g., denoiser_ema0.9999_step0100000.pt)
        """
        print(f"Loading finetuned SLAT flow model from: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location='cpu')
        self.models['slat_flow_model'].load_state_dict(state_dict)
        print("Successfully loaded finetuned SLAT flow model!")
        
    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        dinov2_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
        dinov2_model.eval()
        self.models['image_cond_model'] = dinov2_model
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform
        
    def get_cond(self, image_embeds) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = image_embeds
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples
        
        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()

        return coords

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if 'mesh' in formats:
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
        if 'gaussian' in formats:
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)
        if 'radiance_field' in formats:
            ret['radiance_field'] = self.models['slat_decoder_rf'](slat)
        return ret
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    @torch.no_grad()
    def run(
        self,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        image_embeds: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Run the pipeline.

        Args:
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
            image_embeds (Optional[torch.Tensor]): Precomputed image embeddings to use as condition. If None, the image will be encoded.
        """
        print("Blip image embeds shape:", image_embeds.shape)
        cond = self.get_cond(image_embeds)
        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)

   