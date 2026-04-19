# Copyright 2024 Alpha-VLLM Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import LuminaFeedForward
from diffusers.models.attention_processor import Attention, LuminaAttnProcessor2_0
from diffusers.models.embeddings import LuminaCombinedTimestepCaptionEmbedding, LuminaPatchEmbed, PixArtAlphaTextProjection

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import LuminaLayerNormContinuous, LuminaRMSNormZero, RMSNorm
from diffusers.utils import is_torch_version, logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# flexible_lumina_attn_processor.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb

class FlexibleLuminaAttnProcessor2_0(nn.Module):
    """
    A drop-in replacement for LuminaAttnProcessor2_0 that accepts:
      - key padding masks: [B, K] (bool or float)
      - pairwise masks: [B, 1, Q, K] or [B, H, Q, K] (bool or float)
    """

    def __init__(self):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "This processor requires PyTorch >= 2.0 for scaled_dot_product_attention."
            )

    def _prepare_attn_mask(
        self,
        attention_mask: torch.Tensor | None,
        batch_size: int,
        heads: int,
        q_len: int,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None

        # BOOL path: True means "keep"
        if attention_mask.dtype == torch.bool:
            if attention_mask.ndim == 2:         # [B, K]
                m = attention_mask.view(batch_size, 1, 1, -1)
                m = m.expand(-1, heads, q_len, -1)  # [B, H, Q, K]
            elif attention_mask.ndim == 4:       # [B, 1|H, Q, K]
                m = attention_mask
                # Cast to head dim
                if m.shape[1] == 1:
                    m = m.expand(-1, heads, -1, -1)
            else:
                raise ValueError(f"Unsupported bool mask shape: {attention_mask.shape}")
            # PyTorch SDPA expects boolean "attn_mask" with True=mask out → pass through
            return m

        # FLOAT path: additive mask, 0=allow, -inf=block
        elif attention_mask.dtype in (torch.float16, torch.float32, torch.float64):
            raise NotImplementedError("Implementation not verified for float mask")
            # if attention_mask.ndim == 2:         # [B, K]
            #     m = attention_mask.view(batch_size, 1, 1, -1)
            #     m = m.expand(-1, heads, q_len, -1)
            # elif attention_mask.ndim == 4:       # [B, 1|H, Q, K]
            #     m = attention_mask
            #     if m.shape[1] == 1:
            #         m = m.expand(-1, heads, -1, -1)
            # else:
            #     raise ValueError(f"Unsupported float mask shape: {attention_mask.shape}")
            # return m

        else:
            raise TypeError(f"Unsupported mask dtype: {attention_mask.dtype}")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        query_rotary_emb: torch.Tensor | None = None,
        key_rotary_emb: torch.Tensor | None = None,
        base_sequence_length: int | None = None,
    ) -> torch.Tensor:
        # allow [B, C, H, W] by flattening to [B, L, C]
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape

        # Projections
        query = attn.to_q(hidden_states)                    
        key   = attn.to_k(encoder_hidden_states)           
        value = attn.to_v(encoder_hidden_states)         

        query_dim = query.shape[-1]
        inner_dim = key.shape[-1]
        head_dim  = query_dim // attn.heads
        kv_heads  = inner_dim // head_dim  

        # Norms if present
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Reshape to heads
        query = query.view(batch_size, -1, attn.heads, head_dim)   
        key   = key.view(batch_size, -1, kv_heads, head_dim)       
        value = value.view(batch_size, -1, kv_heads, head_dim)     

        # RoPE
        if query_rotary_emb is not None:
            query = apply_rotary_emb(query, query_rotary_emb, use_real=False)
        if key_rotary_emb is not None:
            key = apply_rotary_emb(key, key_rotary_emb, use_real=False)

        # Type consistency
        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        # Proportional attention scaling (same as original)
        if key_rotary_emb is None:
            softmax_scale = None
        else:
            if base_sequence_length is not None:
                softmax_scale = math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale
            else:
                softmax_scale = attn.scale

        # GQA: expand KV to H by repeating across groups
        n_rep = attn.heads // kv_heads
        if n_rep >= 1:
            key   = key.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)   
            value = value.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)  

        # [B, H, L, Hd]
        query = query.transpose(1, 2)
        key   = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # ---- Change to official processor: Custom attention mask ----
        attn_mask = self._prepare_attn_mask(attention_mask, batch_size, attn.heads, query.shape[2])

        # SDPA
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, scale=softmax_scale
        )  

        hidden_states = hidden_states.transpose(1, 2).to(dtype) 

        if input_ndim == 4:
            raise ValueError(f"Expected input_ndim to not be 4")
            # hidden_states = hidden_states.transpose(1, 2).view(b, c, h, w)

        return hidden_states



class LuminaNextDiTBlock(nn.Module):
    """
    A LuminaNextDiTBlock for LuminaNextDiT2DModel.

    Parameters:
        dim (`int`): Embedding dimension of the input features.
        num_attention_heads (`int`): Number of attention heads.
        num_kv_heads (`int`):
            Number of attention heads in key and value features (if using GQA), or set to None for the same as query.
        multiple_of (`int`): The number of multiple of ffn layer.
        ffn_dim_multiplier (`float`): The multipier factor of ffn layer dimension.
        norm_eps (`float`): The eps for norm layer.
        qk_norm (`bool`): normalization for query and key.
        cross_attention_dim (`int`): Cross attention embedding dimension of the input text prompt hidden_states.
        norm_elementwise_affine (`bool`, *optional*, defaults to True),
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float,
        norm_eps: float,
        qk_norm: bool,
        cross_attention_dim: int,
        norm_elementwise_affine: bool = True,
        custom_processor = True
    ) -> None:
        super().__init__()
        self.head_dim = dim // num_attention_heads

        self.gate = nn.Parameter(torch.zeros([num_attention_heads]))

        # Self-attention
        if custom_processor:
            print("Initializing attention inside DiT with custom processor.")
            self.attn1 = Attention(
                query_dim=dim,
                cross_attention_dim=None,
                dim_head=dim // num_attention_heads,
                qk_norm="layer_norm_across_heads" if qk_norm else None,
                heads=num_attention_heads,
                kv_heads=num_kv_heads,
                eps=1e-5,
                bias=False,
                out_bias=False,
                processor=FlexibleLuminaAttnProcessor2_0(),
            )
        else:
            print("Initializing attention inside DiT without custom processor.")
            self.attn1 = Attention(
                query_dim=dim,
                cross_attention_dim=None,
                dim_head=dim // num_attention_heads,
                qk_norm="layer_norm_across_heads" if qk_norm else None,
                heads=num_attention_heads,
                kv_heads=num_kv_heads,
                eps=1e-5,
                bias=False,
                out_bias=False,
                processor=LuminaAttnProcessor2_0(),
            )
            
        self.attn1.to_out = nn.Identity()

        # Cross-attention
        if custom_processor:
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                dim_head=dim // num_attention_heads,
                qk_norm="layer_norm_across_heads" if qk_norm else None,
                heads=num_attention_heads,
                kv_heads=num_kv_heads,
                eps=1e-5,
                bias=False,
                out_bias=False,
                processor=FlexibleLuminaAttnProcessor2_0(),
            )
        else:
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                dim_head=dim // num_attention_heads,
                qk_norm="layer_norm_across_heads" if qk_norm else None,
                heads=num_attention_heads,
                kv_heads=num_kv_heads,
                eps=1e-5,
                bias=False,
                out_bias=False,
                processor=LuminaAttnProcessor2_0(),
            )

        self.feed_forward = LuminaFeedForward(
            dim=dim,
            inner_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )

        self.norm1 = LuminaRMSNormZero(
            embedding_dim=dim,
            norm_eps=norm_eps,
            norm_elementwise_affine=norm_elementwise_affine,
        )
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)

        self.norm2 = RMSNorm(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)

        self.norm1_context = RMSNorm(cross_attention_dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_mask: torch.Tensor,
        temb: torch.Tensor,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Perform a forward pass through the LuminaNextDiTBlock.

        Parameters:
            hidden_states (`torch.Tensor`): The input of hidden_states for LuminaNextDiTBlock.
            attention_mask (`torch.Tensor): The input of hidden_states corresponse attention mask.
            image_rotary_emb (`torch.Tensor`): Precomputed cosine and sine frequencies.
            encoder_hidden_states: (`torch.Tensor`): The hidden_states of text prompt are processed by Gemma encoder.
            encoder_mask (`torch.Tensor`): The hidden_states of text prompt attention mask.
            temb (`torch.Tensor`): Timestep embedding with text prompt embedding.
            cross_attention_kwargs (`Dict[str, Any]`): kwargs for cross attention.
        """
        residual = hidden_states

        # Self-attention
        norm_hidden_states, gate_msa, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
        self_attn_output = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_hidden_states,
            attention_mask=attention_mask,
            query_rotary_emb=image_rotary_emb,
            key_rotary_emb=image_rotary_emb,
            **cross_attention_kwargs,
        )

        # Cross-attention
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        cross_attn_output = self.attn2(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=encoder_mask,
            query_rotary_emb=image_rotary_emb,
            key_rotary_emb=None,
            **cross_attention_kwargs,
        )
        cross_attn_output = cross_attn_output * self.gate.tanh().view(1, 1, -1, 1)
        mixed_attn_output = self_attn_output + cross_attn_output
        mixed_attn_output = mixed_attn_output.flatten(-2)
        # linear proj
        hidden_states = self.attn2.to_out[0](mixed_attn_output)

        hidden_states = residual + gate_msa.unsqueeze(1).tanh() * self.norm2(hidden_states)

        mlp_output = self.feed_forward(self.ffn_norm1(hidden_states) * (1 + scale_mlp.unsqueeze(1)))

        hidden_states = hidden_states + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(mlp_output)

        return hidden_states


class LuminaNextDiT2DModel(ModelMixin, ConfigMixin):
    """
    LuminaNextDiT: Diffusion model with a Transformer backbone.

    Inherit ModelMixin and ConfigMixin to be compatible with the sampler StableDiffusionPipeline of diffusers.

    Parameters:
        sample_size (`int`): The width of the latent images. This is fixed during training since
            it is used to learn a number of position embeddings.
        patch_size (`int`, *optional*, (`int`, *optional*, defaults to 2):
            The size of each patch in the image. This parameter defines the resolution of patches fed into the model.
        in_channels (`int`, *optional*, defaults to 4):
            The number of input channels for the model. Typically, this matches the number of channels in the input
            images.
        hidden_size (`int`, *optional*, defaults to 4096):
            The dimensionality of the hidden layers in the model. This parameter determines the width of the model's
            hidden representations.
        num_layers (`int`, *optional*, default to 32):
            The number of layers in the model. This defines the depth of the neural network.
        num_attention_heads (`int`, *optional*, defaults to 32):
            The number of attention heads in each attention layer. This parameter specifies how many separate attention
            mechanisms are used.
        num_kv_heads (`int`, *optional*, defaults to 8):
            The number of key-value heads in the attention mechanism, if different from the number of attention heads.
            If None, it defaults to num_attention_heads.
        multiple_of (`int`, *optional*, defaults to 256):
            A factor that the hidden size should be a multiple of. This can help optimize certain hardware
            configurations.
        ffn_dim_multiplier (`float`, *optional*):
            A multiplier for the dimensionality of the feed-forward network. If None, it uses a default value based on
            the model configuration.
        norm_eps (`float`, *optional*, defaults to 1e-5):
            A small value added to the denominator for numerical stability in normalization layers.
        learn_sigma (`bool`, *optional*, defaults to True):
            Whether the model should learn the sigma parameter, which might be related to uncertainty or variance in
            predictions.
        qk_norm (`bool`, *optional*, defaults to True):
            Indicates if the queries and keys in the attention mechanism should be normalized.
        cross_attention_dim (`int`, *optional*, defaults to 2048):
            The dimensionality of the text embeddings. This parameter defines the size of the text representations used
            in the model.
        scaling_factor (`float`, *optional*, defaults to 1.0):
            A scaling factor applied to certain parameters or layers in the model. This can be used for adjusting the
            overall scale of the model's operations.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["LuminaNextDiTBlock"]

    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: Optional[int] = 2,
        in_channels: Optional[int] = 4,
        hidden_size: Optional[int] = 2304,
        num_layers: Optional[int] = 32,  # 32
        num_attention_heads: Optional[int] = 32,  # 32
        num_kv_heads: Optional[int] = None,
        multiple_of: Optional[int] = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: Optional[float] = 1e-5,
        learn_sigma: Optional[bool] = True,
        qk_norm: Optional[bool] = True,
        cross_attention_dim: Optional[int] = 2048,
        scaling_factor: Optional[float] = 1.0,
        summary_token_dim: Optional[int] = 3072,
        num_register_tokens: Optional[int] = 4,
    ) -> None:
        super().__init__()
        self.sample_size = sample_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scaling_factor = scaling_factor
        self.gradient_checkpointing = False

        self.caption_projection = PixArtAlphaTextProjection(in_features=cross_attention_dim, hidden_size=hidden_size)
        self.patch_embedder = LuminaPatchEmbed(patch_size=patch_size, in_channels=in_channels, embed_dim=hidden_size, bias=True)

        self.time_caption_embed = LuminaCombinedTimestepCaptionEmbedding(hidden_size=min(hidden_size, 1024), cross_attention_dim=hidden_size)

        if summary_token_dim is None and num_register_tokens is None:
            self.layers = nn.ModuleList(
                [
                    LuminaNextDiTBlock(
                        hidden_size,
                        num_attention_heads,
                        num_kv_heads,
                        multiple_of,
                        ffn_dim_multiplier,
                        norm_eps,
                        qk_norm,
                        hidden_size,
                        custom_processor=False
                    )
                    for _ in range(num_layers)
                ]
            )
        elif summary_token_dim is None and num_register_tokens is not None:
            self.layers = nn.ModuleList(
                [
                    LuminaNextDiTBlock(
                        hidden_size,
                        num_attention_heads,
                        num_kv_heads,
                        multiple_of,
                        ffn_dim_multiplier,
                        norm_eps,
                        qk_norm,
                        hidden_size,
                        custom_processor=False
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    LuminaNextDiTBlock(
                        hidden_size,
                        num_attention_heads,
                        num_kv_heads,
                        multiple_of,
                        ffn_dim_multiplier,
                        norm_eps,
                        qk_norm,
                        hidden_size,
                        custom_processor=True
                    )
                    for _ in range(num_layers)
                ]
            )
        self.norm_out = LuminaLayerNormContinuous(
            embedding_dim=hidden_size,
            conditioning_embedding_dim=min(hidden_size, 1024),
            elementwise_affine=False,
            eps=1e-6,
            bias=True,
            out_dim=patch_size * patch_size * self.out_channels,
        )
        
        self.num_register_tokens = num_register_tokens
        if self.num_register_tokens is not None:
            self.cls_embed = nn.Linear(self.in_channels, self.hidden_size, bias=True)
            self.reg_embed = nn.Linear(self.in_channels, self.hidden_size, bias=True)

            self.cls_role = nn.Parameter(torch.zeros(1, self.hidden_size))         
            self.reg_roles = nn.Parameter(torch.zeros(1, self.num_register_tokens, self.hidden_size))
            nn.init.trunc_normal_(self.cls_role, std=0.02)
            nn.init.trunc_normal_(self.reg_roles, std=0.02)


            
        # self.final_layer = LuminaFinalLayer(hidden_size, patch_size, self.out_channels)

        # learnable CLS corner token
        # self.cls2d_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        # nn.init.trunc_normal_(self.cls2d_token, std=0.02)

        
        temb_dim = min(self.hidden_size, 1024)
        
        self.summary_token_dim = summary_token_dim
        if self.summary_token_dim is not None:
            # Projection to summary embedding dimension
            # self.sum_out = nn.Sequential(
            #     nn.Linear(self.hidden_size + temb_dim, 4 * self.summary_token_dim),
            #     nn.SiLU(),
            #     nn.Linear(4 * self.summary_token_dim, self.summary_token_dim),
            # )

                        # init
            self.sum_out = LuminaLayerNormContinuous(
                embedding_dim=hidden_size,
                conditioning_embedding_dim=min(hidden_size, 1024),
                elementwise_affine=False,
                eps=1e-6,
                bias=True,
                out_dim=summary_token_dim,
            )
            # Initialize noise projection weights
            # proj_W = torch.randn(self.in_channels, self.summary_token_dim) / (self.in_channels ** 0.5)
            # self.register_buffer("noise_pool_W", proj_W, persistent=True)

            # pooled noise input
            # self.sum_ln   = nn.LayerNorm(self.summary_token_dim)
            # self.sum_inj  = nn.Linear(self.summary_token_dim, self.hidden_size, bias=False)
            # self.cls_ln   = nn.LayerNorm(self.hidden_size)
            # self.sum_gate = nn.Parameter(torch.tensor(1.0))  # learnable gate
            # self.sum_embed = nn.Sequential(
            #     nn.LayerNorm(self.summary_token_dim),  # optional but stabilizes scale
            #     nn.Linear(self.summary_token_dim, self.hidden_size, bias=False),
            # )
            self.sum_embed = nn.Linear(self.summary_token_dim, self.hidden_size, bias=True)


        assert (hidden_size // num_attention_heads) % 4 == 0, "2d rope needs head dim to be divisible by 4"

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_mask: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        cross_attention_kwargs: Dict[str, Any] = None,
        return_dict=True,
        noisy_sum: Optional[torch.Tensor] = None,
        noisy_regs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of LuminaNextDiT.

        Parameters:
            hidden_states (torch.Tensor): Input tensor of shape (N, C, H, W).
            timestep (torch.Tensor): Tensor of diffusion timesteps of shape (N,).
            encoder_hidden_states (torch.Tensor): Tensor of caption features of shape (N, D).
            encoder_mask (torch.Tensor): Tensor of caption masks of shape (N, L).
        """
        
        hidden_states, key_mask, img_size, rope = self.patch_embedder(hidden_states, image_rotary_emb)
        B, L, D = hidden_states.shape
        H = W = int(L ** 0.5)
        assert H * W == L, f"Expected square grid, got L={L}"

        rope = rope.to(hidden_states.device) 
        # breakpoint()
        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        temb = self.time_caption_embed(timestep, encoder_hidden_states, encoder_mask)
        encoder_mask = encoder_mask.bool()

       
        if self.summary_token_dim is not None:
            cls_17, real_17, dummy_17, L17 = self._idx_17(H, W, hidden_states.device)

            # Inflate sequence to 17×17 and insert CLS
            seq17 = hidden_states.new_zeros(B, L17, D)
            seq17[:, real_17, :] = hidden_states
            # after computing seq17[:, real_17, :] = hidden_states
            
            
           
            if noisy_sum is None:
                raise ValueError("noisy_sum (the current s_sigma) must be provided for FM.")
            if noisy_sum.ndim == 4:
                noisy_sum = noisy_sum.squeeze(-1).squeeze(-1)  # [B, S]
            cls_act = self.sum_embed(noisy_sum)                 # [B, hidden_size]
            cls_act = cls_act.unsqueeze(1)                      # [B, 1, H]
            seq17[:, cls_17, :] = cls_act  


            # Inflate RoPE: real tokens get RoPE; CLS/dummies get zeros
            rope17 = rope.new_zeros(1, L17, rope.shape[-1])
            rope17[:, real_17, :] = rope

            # Key padding mask at K=17×17
            if key_mask.dtype != torch.bool:
                key_mask = key_mask.bool()
            kp_17 = torch.ones(B, L17, dtype=torch.bool, device=hidden_states.device)
            kp_17[:, real_17] = key_mask

            # Pairwise bool mask [B,1,Q,K]: True = keep
            A = self._build_pairwise_mask_bool(kp_17, Q=L17, K=L17).clone()
            # Dummies inert: mask rows & cols
            A[:, :, dummy_17, :] = False
            A[:, :, :, dummy_17] = False
            # Patches must NOT attend to CLS (but CLS may attend to patches)
            A[:, :, real_17, cls_17] = False
            A[:, :, cls_17, cls_17] = True
            A[:, :, dummy_17, cls_17] = True

            work_hidden = seq17
            work_rope   = rope17
            work_mask   = A
        
        elif self.summary_token_dim is None and self.num_register_tokens is not None:
            # image size is [(height, width)] * batch_size, increase all by 1
            assert (H,W) == (37,37), f"Dimensions must be 37x37 after patchify when using register tokens, got {H}x{W}"


            new_seq = hidden_states.new_zeros(B, L+self.num_register_tokens+1, D)
            assert new_seq.shape[1] == 37*37+5, f"Expected sequence shape to be 38*38=1444, got {new_seq.shape[1]} vs {38*38}"
            new_seq[:, self.num_register_tokens+1:, :] = hidden_states

            # # noisy_regs: (B, 4, C) -> (B, 4, C)
            # reg_proj = self.reg_embed(noisy_regs)
            # # reg_proj = self.reg_embed(noisy_regs.squeeze(-1).transpose(1,2))
            # # noisy_sum: (B, C) → (B, C)
            # # cls_proj = self.cls_embed(noisy_sum.squeeze(-1).squeeze(-1))
            # cls_proj = self.cls_embed(noisy_sum)
            
            cls_proj = self.cls_embed(noisy_sum) + self.cls_role            
            reg_proj = self.reg_embed(noisy_regs) + self.reg_roles  

            assert reg_proj.shape == (B, self.num_register_tokens, D), f"Expected reg_proj shape to be {[B, self.num_register_tokens, D]}, got {reg_proj.shape}"
            assert cls_proj.shape == (B, D), f"Expected cls_proj shape to be {[B, D]}, got {cls_proj.shape}"
            assert hidden_states.shape == (B, 37*37, D), f"Expected hidden_states shape to be {[B, 37*37, D]}, got {hidden_states.shape}"

            new_seq[:, 0, :] = cls_proj
            new_seq[:, 1:1+self.num_register_tokens, :] = reg_proj

            #   Real tokens get RoPE; CLS/dummies get zeros
            # new_rope = rope.new_zeros(1, L+self.num_register_tokens+1, rope.shape[-1])
            # new_rope[:, 1+self.num_register_tokens:, :] = rope
            # print("Rope shape is:", rope.shape)
            # print("Dtype is:", rope.dtype)
            rope_cis = rope.squeeze(0)  # -> [S, D_h/2]
            ident = torch.ones(
                (self.num_register_tokens + 1, rope_cis.size(1)),
                dtype=rope_cis.dtype,
                device=rope_cis.device,
                )  
            # print("Ident is:", ident)
            rope_cis = torch.cat([ident, rope_cis], dim=0)
            rope_cis = rope_cis.unsqueeze(0) 
            # print("Rope_cis shape is:", rope_cis.shape)

            B, Lp, D = new_seq.shape  # Lp = L + 1 + Nreg
            new_key_mask = torch.ones(B, Lp, dtype=torch.int32, device=new_seq.device)
            new_key_mask[:, 1 + self.num_register_tokens:] = key_mask  # patches keep their mask

            work_hidden = new_seq
            work_rope   = rope_cis
            work_mask   = new_key_mask
   
        else:
            # Plain 16×16 path
            # if key_mask.dtype != torch.bool:
            #     key_mask = key_mask.bool()
            work_hidden = hidden_states
            work_rope   = rope
            # work_mask   = self._build_pairwise_mask_bool(key_mask, Q=L, K=L)
            work_mask = key_mask

       
        # self.print_mask_bool(work_mask, b=0, h=0, rows=slice(0,1000), cols=slice(0,10000))
        assert (work_mask.any(dim=-1)).all(), "some query rows have no allowed keys"

        for layer in self.layers:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                work_hidden = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    work_hidden,
                    work_mask,
                    work_rope,
                    encoder_hidden_states,
                    encoder_mask,
                    temb,
                    cross_attention_kwargs,
                    **ckpt_kwargs,
                )
            else:
                work_hidden = layer(
                    work_hidden,
                    work_mask,            # <<—— pairwise bool mask [B,1,Q,K]
                    work_rope,            # positions aligned to work_hidden
                    encoder_hidden_states,
                    encoder_mask,         # text key-padding mask [B, L_text]
                    temb=temb,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
        if self.summary_token_dim is not None:
            # cls_17 is the flat index of the 2D-CLS corner in the 17x17 sequence you built earlier
            # (reuse the same `cls_17` tensor you computed when inflating to 17x17)
            cls_tok = work_hidden.index_select(1, cls_17).squeeze(1)  # [B, hidden_size]
     
            cls_radio = self.sum_out(cls_tok.unsqueeze(1), temb).squeeze(1)  # [B, S]

            # # condition with temb (shape [B, temb_dim]) then project to RaDiO dim 3072
            # cls_in   = torch.cat([cls_tok, temb], dim=-1)            # [B, hidden_size + temb_dim]
            # cls_radio = self.sum_out(cls_in)                    # [B, 3072]
            patch_tokens = work_hidden.index_select(1, real_17)
        # elif self.summary_token_dim is None and self.num_register_tokens is not None:
        #     cls_tok = work_hidden[:, 0, :]  # [B, C]
        #     cls_dino = self.cls_out(cls_tok.unsqueeze(1), temb).squeeze(1)  # [B, C]

        #     reg_toks = work_hidden[:, 1:1+self.num_register_tokens, :]  # [B, num_register_tokens, C]
        #     reg_dino = self.regs_out(reg_toks, temb)                    # [B, num_register_tokens, C]

        #     patch_tokens = work_hidden[:, 1+self.num_register_tokens:, :]

        #     assert patch_tokens.shape[1] == 37*37, f"Expected patch tokens to be {H*W}, got {patch_tokens.shape[1]}"
        #     assert patch_tokens.shape[2] == 1024, f"Expected patch tokens dim to be {1024}, got {patch_tokens.shape[2]}"
        #     assert cls_dino.shape == (B, 1024), f"Expected cls_dino shape to be {[B, 1024]}, got {cls_dino.shape}"
        #     assert reg_dino.shape == (B, 4, 1024), f"Expected reg_dino shape to be {[B, 4, 1024]}, got {reg_dino.shape}"
        else:
            patch_tokens = work_hidden

        if self.summary_token_dim is None and self.num_register_tokens is not None:

            out = self.norm_out(work_hidden, temb)


            cls_dino = out[:, 0, :]  # [B, C]
            reg_dino = out[:, 1:1+self.num_register_tokens, :]  # [B, num_register_tokens, C]
            hidden_states = out[:, 1+self.num_register_tokens:, :]

            assert hidden_states.shape[1] == 37*37, f"Expected hidden_states to be {H*W}, got {hidden_states.shape[1]}"
            assert hidden_states.shape[2] == 1024, f"Expected hidden_states dim to be {1024}, got {hidden_states.shape[2]}"
            assert cls_dino.shape == (B, 1024), f"Expected cls_dino shape to be {[B, 1024]}, got {cls_dino.shape}"
            assert reg_dino.shape == (B, 4, 1024), f"Expected reg_dino shape to be {[B, 4, 1024]}, got {reg_dino.shape}"

        else:
            hidden_states = self.norm_out(patch_tokens, temb)


        
        # unpatchify
        height_tokens = width_tokens = self.patch_size
        height, width = img_size[0]
        batch_size = hidden_states.size(0)
        sequence_length = (height // height_tokens) * (width // width_tokens)
        hidden_states = hidden_states[:, :sequence_length].view(
            batch_size, height // height_tokens, width // width_tokens, height_tokens, width_tokens, self.out_channels
        )
        output = hidden_states.permute(0, 5, 1, 3, 2, 4).flatten(4, 5).flatten(2, 3)

        if self.summary_token_dim is not None:
            if not return_dict:
                return (output, cls_radio)
            return Transformer2DModelOutput(sample=output), cls_radio
        elif self.summary_token_dim is None and self.num_register_tokens is not None:
            if not return_dict:
                return (output, cls_dino, reg_dino)
            return Transformer2DModelOutput(sample=output), cls_dino, reg_dino
        else:
            if not return_dict:
                return (output,)
            return Transformer2DModelOutput(sample=output)
            
    def _idx_17(self, H: int, W: int, device):
        # map (r,c) on a (H+1)x(W+1) grid to flat index
        def rc2i(r, c): return r * (W + 1) + c
        cls = torch.tensor([rc2i(0, 0)], device=device)
        real = torch.tensor([rc2i(r, c) for r in range(1, H + 1) for c in range(1, W + 1)], device=device)
        dummy = torch.tensor([rc2i(0, c) for c in range(1, W + 1)] + [rc2i(r, 0) for r in range(1, H + 1)], device=device)
        L17 = (H + 1) * (W + 1)
        return cls, real, dummy, L17

    def _build_pairwise_mask_bool(self, keypad_1d: torch.Tensor, Q: int, K: int) -> torch.Tensor:
        """
        keypad_1d: [B, K] with True=keep, False=pad (or 1/0). Returns bool mask [B,1,Q,K] with True=mask-out.
        """
        if keypad_1d.dtype != torch.bool:
            keypad_1d = keypad_1d.bool()
        # mask-out padded keys (columns)
        return keypad_1d.unsqueeze(1).unsqueeze(2).expand(-1, 1, Q, -1)

    def print_mask_bool(self, mask, b=0, h=0, rows=slice(0, 1000), cols=slice(0, 1000),
                    true_char='█', false_char='·'):
        """
        mask: [B,H,Lq,Lk] bool (True=keep/allow, False=block)
        """
        m = mask[b, h, rows, cols].detach().cpu().numpy()
        for r in range(m.shape[0]):
            line = ''.join(true_char if m[r, c] else false_char for c in range(m.shape[1]))
            print(line)

    def _idx_dino_grid(self, H: int, W: int, N_reg: int, device):
        """
        Make a (H+1) x W1 canvas:
        (0,0)      : CLS
        (0,1..Nreg): REGs
        (1..H,1..W): image patches
        left col (r>0,0) and any extra top row cells after regs are dummies.
        """
        H1 = H + 1
        W1 = max(W + 1, 1 + N_reg)

        def rc2i(r, c): return r * W1 + c

        idx_cls     = torch.tensor([rc2i(0, 0)], device=device, dtype=torch.long)
        idx_regs    = torch.tensor([rc2i(0, 1 + k) for k in range(N_reg)],
                                device=device, dtype=torch.long) if N_reg > 0 else torch.empty(0, dtype=torch.long, device=device)
        idx_patches = torch.tensor([rc2i(1 + r, 1 + c) for r in range(H) for c in range(W)],
                                device=device, dtype=torch.long)

        # dummies: left column under CLS + top row after regs (if any)
        dummy_left       = [rc2i(r, 0) for r in range(1, H1)]
        dummy_top_excess = [rc2i(0, c) for c in range(1 + N_reg, W1)]
        idx_dummy = torch.tensor(dummy_left + dummy_top_excess, device=device, dtype=torch.long) \
                    if (dummy_left or dummy_top_excess) else torch.empty(0, dtype=torch.long, device=device)

        L1 = H1 * W1
        return H1, W1, idx_cls, idx_regs, idx_patches, idx_dummy, L1


