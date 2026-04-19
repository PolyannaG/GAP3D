import torch
import torch.nn as nn
import warnings
from PIL import Image
from torchvision import transforms
from torch.hub import load_state_dict_from_url
from timm.models import clean_state_dict

from .model.radio_model import RADIOModel, create_model_from_args
from .model.common import RESOURCE_MAP, DEFAULT_VERSION
from .model.adaptor_registry import adaptor_registry
from .model.enable_damp import configure_damp_from_args
from .model.enable_spectral_reparam import disable_spectral_reparam, configure_spectral_reparam_from_args
from .model.feature_normalizer import FeatureNormalizer, IntermediateFeatureNormalizer
from .model.input_conditioner import get_default_conditioner
from .radio_processor import RadioImageTrainProcessor
from .factory import get_model_config


class RadioVisionTower(nn.Module):
    def __init__(self, vision_tower, vision_tower_cfg, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.vision_tower_cfg = vision_tower_cfg
        self.config = get_model_config(vision_tower)
        
        # Extract RADIO model version from vision_tower string
        # Expected format: "radio_v2.5-h" or similar
        if vision_tower in RESOURCE_MAP:
            self.radio_version = vision_tower
        else:
            # Default to a specific version if not found
            self.radio_version = DEFAULT_VERSION
            
        self.resource = RESOURCE_MAP[self.radio_version]

        if not delay_load:
            print(f"Loading RADIO Vision Tower: {self.radio_version}")
            self.load_model()
        else:
            self.cfg_only = True

    def load_model(self, device_map=None):
        print(f"Loading RADIO model: {self.radio_version}")
        
        # Create RADIO model using manual loading approach
        self.vision_tower = self._load_radio_model_manually(
            version=self.radio_version,
            device=device_map,
            progress=True
        )
        
        # Setup image processor
        print(f"Setting up image processor for RADIO model: {self.radio_version}")
        self.image_processor = RadioImageTrainProcessor(self.config["vision_cfg"]["image_size"])
        
        
        print(f"Loaded RADIO vision tower: {self.radio_version}")

        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def get_prefix_state_dict(self, state_dict, prefix):
        """Extract state dict entries with a specific prefix."""
        return {
            k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)
        }

    def _load_radio_model_manually(self, version, device="cuda", progress=True, make_preprocessor_external=True):
        """
        Manually load a RADIO model from scratch.
        """
        print(f"Manual loading RADIO model: {version}")
        
        # Get model resource information
        if version not in RESOURCE_MAP:
            raise ValueError(f"Unknown version: {version}. Available versions: {list(RESOURCE_MAP.keys())}")
        
        resource = RESOURCE_MAP[version]
        print(f"Model info - Patch size: {resource.patch_size}, Max resolution: {resource.max_resolution}")
        
        # Download checkpoint
        print("Downloading checkpoint...")
        chk = load_state_dict_from_url(
            resource.url, 
            progress=progress, 
            map_location="cpu", 
            weights_only=False
        )
        
        # Extract state dict and args
        if "state_dict_ema" in chk:
            print("Using EMA state dict for inference.")
            state_dict = chk["state_dict_ema"]
            chk['args'].spectral_reparam = False
        else:
            print("Using standard state dict for inference.")
            state_dict = chk["state_dict"]
        
        args = chk["args"]
        print(f"Loaded args: {args}")
        print(f"Model architecture: {args.model}")
        
        # Create the base model architecture
        print("Creating model architecture...")
        model = create_model_from_args(args)
        
        # Get model state dict (without prefix)
        model_state_dict = self.get_prefix_state_dict(state_dict, "base_model.")
        
        # Configure spectral reparametrization if needed
        if args.spectral_reparam:
            print("Configuring spectral reparametrization...")
            configure_spectral_reparam_from_args(model, args, state_dict_guidance=model_state_dict)
        
        # Configure DAMP if needed
        if getattr(args, 'damp', None):
            print("Configuring DAMP...")
            configure_damp_from_args(model, args)
        
        # Clean and load state dict
        print("Loading model state dict...")
        state_dict = clean_state_dict(state_dict)
        key_warn = model.load_state_dict(model_state_dict, strict=False)
        
        if key_warn.missing_keys:
            warnings.warn(f'Missing keys in state dict: {key_warn.missing_keys}')
        if key_warn.unexpected_keys:
            warnings.warn(f'Unexpected keys in state dict: {key_warn.unexpected_keys}')
        print("Model architecture created and state dict loaded.")
        
        # Disable spectral reparametrization for inference
        if chk['args'].spectral_reparam:
            print("Disabling spectral reparametrization for inference...")
            disable_spectral_reparam(model)
            chk['args'].spectral_reparam = False
        
        # Create input conditioner
        print("Setting up input conditioning...")
        conditioner = get_default_conditioner()
        conditioner.load_state_dict(self.get_prefix_state_dict(state_dict, "input_conditioner."))
        
        # Set model dtype
        dtype = getattr(chk['args'], 'dtype', torch.float32)
        print(f"Model dtype: {dtype}")
        # print(f"Setting model dtype to: {dtype}")
        # model.to(dtype=dtype, device=device)
        # conditioner.dtype = dtype
        # conditioner.to(device=device)
        
        # Handle class tokens and summary indices
        print("Determining class tokens and summary indices...")
        cls_token_per_teacher = getattr(chk['args'], 'cls_token_per_teacher', True)
        if cls_token_per_teacher:
            name_to_idx_map = dict()
            for i, t in enumerate(chk['args'].teachers):
                if t.get('use_summary', True):
                    name = t['name']
                    if name not in name_to_idx_map:
                        name_to_idx_map[name] = i
            print(f"Name to index map: {name_to_idx_map}")
            summary_idxs = torch.tensor(sorted(name_to_idx_map.values()), dtype=torch.int64)
        else:
            summary_idxs = torch.tensor([0], dtype=torch.int64)
        
        # Load feature normalizers
        feat_norm_sd = self.get_prefix_state_dict(state_dict, '_feature_normalizer.')
        feature_normalizer = None
        if feat_norm_sd:
            feature_normalizer = FeatureNormalizer(feat_norm_sd['mean'].shape[0], dtype=dtype)
            feature_normalizer.load_state_dict(feat_norm_sd)
        
        inter_feat_norm_sd = self.get_prefix_state_dict(state_dict, '_intermediate_feature_normalizer.')
        inter_feature_normalizer = None
        if inter_feat_norm_sd:
            inter_feature_normalizer = IntermediateFeatureNormalizer(
                *inter_feat_norm_sd['means'].shape[:2],
                rot_per_layer=inter_feat_norm_sd['rotation'].ndim == 3,
                dtype=dtype
            )
            inter_feature_normalizer.load_state_dict(inter_feat_norm_sd)
        
        # Create the final RADIO model
        print("Assembling RADIO model...")
        radio = RADIOModel(
            model,
            conditioner,
            summary_idxs=summary_idxs,
            patch_size=resource.patch_size,
            max_resolution=resource.max_resolution,
            window_size=None,  # Can be set for VitDet
            preferred_resolution=resource.preferred_resolution,
            adaptors=dict(),  # No adaptors for now, can be added later if needed
            feature_normalizer=feature_normalizer,
            inter_feature_normalizer=inter_feature_normalizer,
        )

        if make_preprocessor_external:
            print("Making preprocessor external...")
            # Images will be normalized to [0, 1] range during preprocessing
            radio.make_preprocessor_external()
        
        print(f"RADIO model loaded successfully!")
        print(f"- Embed dim: {radio.embed_dim}")
        print(f"- Summary dim: {radio.summary_dim}")
        print(f"- Patch size: {radio.patch_size}")
        print(f"- Preferred resolution: {radio.preferred_resolution}")
        
        return radio


    def forward(self, images, return_summary=True):
        if type(images) is list:
            raise ValueError("Input should be a single tensor, not a list of images.")
        else:
            with torch.no_grad():
                # RADIO expects images in [0, 1] range
                input_images = images.to(device=self.device, dtype=self.dtype)
                
                # Get nearest supported resolution and resize if needed
                nearest_res = self.vision_tower.get_nearest_supported_resolution(*input_images.shape[-2:])
                if input_images.shape[-2:] != nearest_res:
                    raise ValueError(
                        f"Input image size {input_images.shape[-2:]} does not match expected size {nearest_res}. "
                        "Please resize your images to the nearest supported resolution.")
                
                # RADIO returns RadioOutput namedtuple with .features and .summary
                radio_output = self.vision_tower(input_images, feature_fmt='NCHW')
                if hasattr(radio_output, 'features') and hasattr(radio_output, 'summary'):
                    image_features = radio_output.features
                    image_summary = radio_output.summary
                    # print(f"RADIO model output: features shape {image_features.shape}, summary shape {image_summary.shape}")
                    # features shape torch.Size([16, 1024, 32, 32]), summary shape torch.Size([16, 3072])
                    image_features = image_features.to(dtype=self.dtype)
                    image_summary = image_summary.to(dtype=self.dtype)
                    # print(f"After cast: Device: {self.device}, Dtype: {self.dtype}") torch.bfloat16
                else:
                    raise ValueError("RADIO model output does not contain expected features and summary.")

        if return_summary:
            return image_features, image_summary
        else:
            return image_features

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
        # Use RADIO's embed_dim property if available
        if hasattr(self.vision_tower, 'embed_dim'):
            return self.vision_tower.embed_dim
        elif hasattr(self.vision_tower, 'model') and hasattr(self.vision_tower.model, 'embed_dim'):
            return self.vision_tower.model.embed_dim
        else:
            print("WARNING: RADIO model does not have embed_dim property. "
                  "Falling back to common RADIO hidden sizes based on version.")
            # Fallback: Common RADIO hidden sizes based on version
            if 'h' in self.radio_version:
                return 1280  # RADIO-H
            elif 'l' in self.radio_version:
                return 1024  # RADIO-L
            elif 'b' in self.radio_version:
                return 768   # RADIO-B
            else:
                return 1280  # Default

    @property
    def num_patches(self):
        image_size = self.config['vision_cfg']['image_size']
        return (image_size // self.resource.patch_size) ** 2


    @property
    def image_size(self):
        return self.config['vision_cfg']['image_size']
