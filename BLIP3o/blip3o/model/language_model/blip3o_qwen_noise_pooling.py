from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import copy
import math
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from blip3o.model.blip3o_arch import blip3oMetaModel, blip3oMetaForCausalLM

from transformers import Qwen2_5_VLConfig, Qwen2_5_VLModel, Qwen2_5_VLForConditionalGeneration

from blip3o.constants import UND_IMAGE_TOKEN_IDX



from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import numpy_to_pil
import numpy as np
from diffusers.models import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler


class blip3oQwenConfig(Qwen2_5_VLConfig):
    model_type = "blip3o_qwen"


class blip3oQwenModel(blip3oMetaModel, Qwen2_5_VLModel):
    config_class = blip3oQwenConfig

    def __init__(self, config: Qwen2_5_VLConfig):
        super(blip3oQwenModel, self).__init__(config)


class blip3oQwenForCausalLM(Qwen2_5_VLForConditionalGeneration, blip3oMetaForCausalLM):
    config_class = blip3oQwenConfig

    def __init__(self, config):
        Qwen2_5_VLForConditionalGeneration.__init__(self, config)
        config.model_type = "blip3o_qwen"

        self.model = blip3oQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        ids: Optional[list] = None,
        i_s_pos: Optional[list] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        gen_image: Optional[torch.FloatTensor] = None,
        und_image: Optional[torch.FloatTensor] = None,
        grid_thw: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

       
        
        if inputs_embeds is None:
            if self.predict_summary_token:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                    latents,
                    summary_latent
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    gen_image,
                    und_image,
                    grid_thw,
                    i_s_pos,
                    image_sizes
                )
            else:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                    latents,
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    gen_image,
                    und_image,
                    grid_thw,
                    i_s_pos,
                    image_sizes
                )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        total_loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)


            # compute image loss
            # target_img_embeds = torch.clone(inputs_embeds.detach())[:,1:,:] # get target image emb
            img_loss_funct = torch.nn.MSELoss()
            # img_hidden_states = self.get_model().down_projector(hidden_states[:,-self.get_n_query():,:])
            img_hidden_states = []
            N_QUERY = self.get_n_query() 
            for b in range(hidden_states.shape[0]):
                img_hidden_states.append(hidden_states[b,i_s_pos[b]:i_s_pos[b]+N_QUERY,:])
            img_hidden_states = torch.stack(img_hidden_states,dim=0)
            img_hidden_states = self.get_model().down_projector(img_hidden_states)
            # img_loss = 0.0
            img_loss = None
            sum_loss = None
            if latents is None:
                img_loss = img_loss_funct(img_hidden_states, torch.clone(img_hidden_states.detach()))
            else:
                bsz = latents.shape[0]
                # device = latents.device
                dtype = latents.dtype
                noise = torch.randn_like(latents, device=latents.device)
                u = torch.rand(size=(bsz,), device="cpu")
                indices = (u * self.get_model().noise_scheduler.config.num_train_timesteps).long()
                timesteps = self.get_model().noise_scheduler.timesteps[indices].to(device=latents.device)
                sigmas = self.get_sigmas(timesteps, latents.device, n_dim=latents.ndim, dtype=dtype)
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
                # print(f"noisy_latents shape {noisy_latents.shape}, img_hidden_states shape {img_hidden_states.shape}")
                
                if self.predict_summary_token:
                    B, C, H, W = latents.shape
                    summary_latent = summary_latent.unsqueeze(-1).unsqueeze(-1)
                    # project channel-wise pooled noise to summary dim with fixed W
                    if hasattr(self.get_model().dit.model, "noise_pool_W"):
                        Wsum = self.get_model().dit.model.noise_pool_W    # [C, S]
                        eps_chan = noise.mean(dim=(2, 3)) * (H * W) ** 0.5          # [B, C]
                        eps_sum = eps_chan @ Wsum   
                        eps_sum = eps_sum.unsqueeze(-1).unsqueeze(-1)       
                        sig_sum = self.get_sigmas(timesteps, latents.device, n_dim=eps_sum.ndim, dtype=latents.dtype)  # (B,1,1,1)
                        noisy_sum = (1.0 - sig_sum) * summary_latent + sig_sum * eps_sum
                    else:
                        raise ValueError("No noise_pool_W found in model, please implement a learnable buffer for pooling.")

                    
                    # noise_sum = torch.randn_like(summary_latent, device=latents.device)
                    # sig_sum = self.get_sigmas(timesteps, latents.device, n_dim=noise_sum.ndim, dtype=latents.dtype)  # (B,1,1,1)
                    # noisy_sum = (1.0 - sig_sum) * summary_latent + sig_sum * noise_sum

                    # v_sum_target = (noise_sum - summary_latent).detach()  

                    dit_out = self.get_model().dit(
                        x=noisy_latents,
                        timestep=timesteps,
                        z_latents=self.mask_drop(img_hidden_states),
                        noisy_sum=noisy_sum,
                    )                     
                else:
                    dit_out = self.get_model().dit(
                        x=noisy_latents,
                        timestep=timesteps,
                        z_latents=self.mask_drop(img_hidden_states),
                    )
                target = noise - latents
                v_sum_target = (eps_sum - summary_latent).detach()  
                
                if isinstance(dit_out, (tuple, list)):
                    img_vel_pred = dit_out[0]
                    cls_vel_pred = dit_out[1].unsqueeze(-1).unsqueeze(-1)
                    assert img_vel_pred.shape == (B, 1024, 16, 16), f"Unexpected img_vel_pred shape: {img_vel_pred.shape}"
                    assert cls_vel_pred.shape == (B, 3072, 1, 1), f"Unexpected cls_vel_pred shape: {cls_vel_pred.shape}"
                else:
                    img_vel_pred = dit_out
                    cls_vel_pred = None

                img_loss = F.mse_loss(img_vel_pred.float(), target.float(), reduction="mean")

                if self.predict_summary_token:
                    sum_loss = torch.nn.functional.mse_loss(
                        cls_vel_pred.float(), v_sum_target.float(), reduction="mean"
                    )
                
                lambda_sum = getattr(self.get_model(), "lambda_summary", 0.5)
                total_loss = img_loss + (lambda_sum * sum_loss if sum_loss is not None else 0.0)
                
                if self.predict_summary_token:
                    print(f"img loss {img_loss}, sum loss {sum_loss}, total loss {total_loss}")
                else:   
                    print(f"img loss {img_loss}")

                # pred_norm = torch.norm(noise_pred.float(), p=2)
                # target_norm = torch.norm(target.float(), p=2)
                # error = noise_pred.float() - target.float()
                # error_norm = torch.norm(error, p=2)
                # print(f"Pred norm: {pred_norm.item():.4f}, Target norm: {target_norm.item():.4f}, Error norm: {error_norm.item():.4f}")
                # print(f"Error norm: {error_norm.item():.4f}")

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                img_indicator,
                _
            ) = self.prepare_inputs_labels_for_understanding(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    @torch.no_grad()
    def generate_image(
        self,
        text: List[str],
        tokenizer: AutoTokenizer,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        max_var: Optional[float] = None,
        # placeholder: str = DEFAULT_IMG_PLACEHOLDER,
    ):  
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained("Alpha-VLLM/Lumina-Next-SFT-diffusers", subfolder="scheduler")


        N_QUERY = self.get_n_query()            
        inputs = tokenizer(text, padding="longest", return_tensors="pt")
        device = self.get_model().device
        attention_mask = inputs.attention_mask.to(device)
        input_ids = inputs.input_ids.to(device)  # B x N
        input_ids = torch.cat([input_ids, torch.tensor([[151665]]).to(device)], dim=1)
        # breakpoint()


        text_embeds = self.get_model().embed_tokens(input_ids)
        latent_queries = self.get_model().latent_queries.repeat(text_embeds.shape[0], 1, 1)


        if pixel_values is not None:
            und_image_idx = (input_ids == UND_IMAGE_TOKEN_IDX)
            pixel_values = pixel_values.type(self.visual.dtype)
            und_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            text_embeds[und_image_idx] = und_image_embeds.to(text_embeds.device)[:und_image_idx.sum(), :]


        text_embeds = torch.cat([text_embeds, latent_queries], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(latent_queries[:, :, 0])], dim=1)


        outputs = self.model(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1][:,-N_QUERY:,:]
        img_hidden_states = hidden_states 
        output_img = self.sample_images(img_hidden_states, scheduler)
        output_img = output_img.view(1, 1792, -1).permute(0,2,1).contiguous()

        return output_img

    @torch.no_grad()
    def sample_images(
        self,
        img_hidden_states,
        scheduler,
        guidance_scale: float = 3.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 30,
        num_images_per_prompt: int = 1,
        return_tensor=False,
        **kwargs,
    ):
        
        device = img_hidden_states.device
        dtype = img_hidden_states.dtype


        img_hidden_states_null = torch.zeros_like(img_hidden_states, device=device, dtype=dtype)
        img_hidden_states_input = torch.cat([img_hidden_states_null, img_hidden_states], 0)

        batch_size = img_hidden_states.shape[0]
        latent_size = self.get_model().dit.config.input_size
        latent_channels = self.get_model().dit.config.in_channels

        latents = randn_tensor(
            shape=(batch_size * num_images_per_prompt, latent_channels, latent_size, latent_size),
            generator=generator,
            device=device,
            dtype=dtype,
        )

        # set step values
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        scheduler.set_timesteps(num_inference_steps, sigmas=sigmas)

        # Repeat z_latents and conditions for each image per prompt
        img_hidden_states_input = img_hidden_states_input.repeat_interleave(num_images_per_prompt, dim=0)

        for t in scheduler.timesteps:
            latent_model_input = latents.repeat(2, 1, 1, 1)
            if hasattr(scheduler, "scale_model_input"):
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # predict noise model_output
            noise_pred = self.get_model().dit(
                x=latent_model_input,
                timestep=t.unsqueeze(0).expand(latent_model_input.shape[0]).to(latent_model_input.device, torch.long),
                z_latents=img_hidden_states_input,
            )

            # perform guidance
            noise_pred_uncond, noise_pred = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            # compute previous image: x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # samples = self.decode_latents(latents, return_tensor=return_tensor)
        # breakpoint()
        return latents

    def decode_latents(self, latents, normalize=True, return_tensor=False):
        if isinstance(self.get_model().vae, AutoencoderKL):
            latents = latents / self.get_model().vae.config.scaling_factor
            if self.get_model().vae.config.shift_factor is not None:
                latents = latents + self.get_model().vae.config.shift_factor
            latents = latents.to(dtype=torch.float32)
            samples = self.get_model().vae.decode(latents).sample
        else:
            samples = self.get_model().vae.decode(latents)
        if normalize:
            samples = (samples / 2 + 0.5).clamp(0, 1)
        else:
            samples = samples.clamp(-1, 1)
        if return_tensor:
            return samples
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()
        samples = numpy_to_pil(samples)
        return samples

    def prepare_and_encode_inputs(
        self,
        inputs: List[str | Image.Image],
        tokenizer: AutoTokenizer,
        do_classifier_free_guidance: bool = False,
    ):
        # pdb.set_trace()
        device = self.get_model().device
        dtype = self.get_model().dtype

        has_image, has_text = False, False
        text_prompt, image_prompt = "", []
        img_processor = self.get_vision_tower().image_processor
        negative_prompt = {}

        for x in inputs:
            if isinstance(x, str):
                has_text = True
                text_prompt += x
            else:
                has_image = True
                text_prompt += DEFAULT_IMAGE_TOKEN
                image_prompt.append(img_processor.preprocess(x, return_tensors='pt')['pixel_values'])
        # pdb.set_trace()
        if len(image_prompt) == 0:
            image_prompt = None
        else:
            image_prompt = torch.cat(image_prompt)
            image_prompt = image_prompt.type(dtype).to(device)

        if has_image and not has_text:
            prompt = self.encode_images(image_prompt)
            # pdb.set_trace()
            if do_classifier_free_guidance:
                key = "[NULL_IMAGE]"
                if key not in negative_prompt:
                    negative_image = torch.zeros_like(image_prompt)
                    negative_prompt[key] = self.encode_images(negative_image)
                prompt = torch.cat([prompt, negative_prompt[key]], dim=0)
        else:
            prompt = self.generate_image(text=[text_prompt], image=image_prompt, tokenizer=tokenizer)
            if do_classifier_free_guidance:
                key = ""
                if key not in negative_prompt:
                    negative_prompt[key] = self.generate_image(text=[""], tokenizer=tokenizer)
                prompt = torch.cat([prompt, negative_prompt[key]], dim=0)
        
        gen_pooling = self.get_gen_pooling()
        n_query = self.get_n_query()
        num_img, _, c = prompt.shape
        if 'pool2d' in gen_pooling and has_text and not 'early' in gen_pooling:
            stride = int(gen_pooling.split('_')[1])
            sqrt_n = int(n_query**0.5)
            prompt = prompt.permute(0, 2, 1).reshape(num_img, -1, sqrt_n, sqrt_n)
            prompt = F.avg_pool2d(prompt, kernel_size=(stride, stride), stride=stride)
            prompt = prompt.reshape(num_img, c, -1).permute(0,2,1)
        return prompt


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

    @torch.no_grad()
    def sample_images_no_cfg(
        self,
        img_hidden_states,
        scheduler,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 30,
        num_images_per_prompt: int = 1,
        return_tensor=False,
        **kwargs,
    ):
        
        device = img_hidden_states.device
        dtype = img_hidden_states.dtype


        batch_size = img_hidden_states.shape[0]
        latent_size = self.get_model().dit.config.input_size
        latent_channels = self.get_model().dit.config.in_channels

        latents = randn_tensor(
            shape=(batch_size * num_images_per_prompt, latent_channels, latent_size, latent_size),
            generator=generator,
            device=device,
            dtype=dtype,
        )

        # set step values
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        scheduler.set_timesteps(num_inference_steps, sigmas=sigmas)

        # Repeat z_latents and conditions for each image per prompt
        img_hidden_states_input = img_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)

        for t in scheduler.timesteps:
            # latent_model_input = latents.repeat(2, 1, 1, 1)
            latent_model_input = latents
            if hasattr(scheduler, "scale_model_input"):
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # predict noise model_output
            noise_pred = self.get_model().dit(
                x=latent_model_input,
                timestep=t.unsqueeze(0).expand(latent_model_input.shape[0]).to(latent_model_input.device, torch.long),
                z_latents=img_hidden_states_input,
            )

            # perform guidance
            # noise_pred_uncond, noise_pred = noise_pred.chunk(2)
            # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            # compute previous image: x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # samples = self.decode_latents(latents, return_tensor=return_tensor)
        # breakpoint()
        return latents

    @torch.no_grad()
    def evaluate_mapper(
        self,                    
        dataloader,                
        device=None,              
        num_inference_steps: int = 30,
        k_neigh: int = 10,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        max_eval_samples: Optional[int] = 1000
    ):
       
        self.get_model().dit.eval()
        self.get_model().eval()

        print(f"Evaluating mapper on device {device}")
        self.get_model().to(device)

        cos_means, cos_stds, mse_vals = [], [], []
        pred_norm_means, pred_norm_stds = [], []
        tgt_norm_means,  tgt_norm_stds  = [], []
        ratio_means, ratio_stds = [], []
        all_overlap_fracs, all_overlap_stds = [], []

        sample_count = 0
        from tqdm import tqdm
        for batch in tqdm(dataloader, desc="Eval mapper"):
            if max_eval_samples is not None and sample_count >= max_eval_samples:
                break
            sample_count += len(batch['input_ids'])
            
            batch_on_dev: Dict[str, Any] = {}
            for k, v in batch.items():
                # print(f"Batch {k} type: {type(v)}")
                batch_on_dev[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            
            # for k,v in batch_on_dev.items():
            #     if isinstance(v, torch.Tensor):
            #         print(f"Batch {k} shape: {v.shape}, dtype: {v.dtype}, device: {v.device}")

            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            
            input_ids = batch_on_dev.get("input_ids", None)
            position_ids = batch_on_dev.get("position_ids", None)
            attention_mask = batch_on_dev.get("attention_mask", None)
            past_key_values = batch_on_dev.get("past_key_values", None) 
            labels = batch_on_dev.get("labels", None)
            gen_image = batch_on_dev.get("gen_image", None)
            und_image = batch_on_dev.get("und_image", None)
            grid_thw = batch_on_dev.get("grid_thw", None)
            i_s_pos = batch_on_dev.get("i_s_pos", None)
            image_sizes = batch_on_dev.get("image_sizes", None)
            
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, latents) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                gen_image,
                und_image,
                grid_thw,
                i_s_pos,
                image_sizes
             ) 
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = outputs[0]
            
            if labels is not None:
        
                # compute image loss
                # target_img_embeds = torch.clone(inputs_embeds.detach())[:,1:,:] # get target image emb
                img_loss_funct = torch.nn.MSELoss()
                # img_hidden_states = self.get_model().down_projector(hidden_states[:,-self.get_n_query():,:])
                img_hidden_states = []
                
                for b in range(hidden_states.shape[0]):
                    img_hidden_states.append(hidden_states[b,i_s_pos[b]:i_s_pos[b]+self.get_n_query(),:])
                img_hidden_states = torch.stack(img_hidden_states,dim=0)
                img_hidden_states = self.get_model().down_projector(img_hidden_states)

            if not hasattr(self, "_inference_scheduler"):
                self._inference_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                    "Alpha-VLLM/Lumina-Next-SFT-diffusers", subfolder="scheduler"
            )
            scheduler = self._inference_scheduler
            pred = self.sample_images_no_cfg(
                img_hidden_states,
                scheduler=scheduler,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=1,
                generator=None,
            )                                                            # [B, 1024, H, W]

            pred_f = pred.float()
            target_f = latents.float()

            # 4) compute per‐patch cosine‐map
            cos_map = F.cosine_similarity(pred_f, target_f, dim=1, eps=1e-8)  # [B,H,W]
            cos_means.append(cos_map.mean(dim=(1,2)).mean().item())  # avg of per-image means
            cos_stds.append(cos_map.std(dim=(1,2)).mean().item())    # avg of per-image stds

            mse_vals.append(F.mse_loss(pred_f, target_f, reduction="mean").item())

            # norms + ratio (per-image)
            pred_norm_map = pred_f.norm(dim=1)     # [B,H,W]
            target_norm_map = target_f.norm(dim=1) # [B,H,W]
            ratio_map = pred_norm_map / (target_norm_map + 1e-8)

            pred_norm_means.append(pred_norm_map.mean(dim=(1,2)).mean().item())
            pred_norm_stds.append(pred_norm_map.std(dim=(1,2)).mean().item())
            tgt_norm_means.append(target_norm_map.mean(dim=(1,2)).mean().item())
            tgt_norm_stds.append(target_norm_map.std(dim=(1,2)).mean().item())
            ratio_means.append(ratio_map.mean(dim=(1,2)).mean().item())
            ratio_stds.append(ratio_map.std(dim=(1,2)).mean().item())


            # 6) Neighborhood preservation
            B, C, H, W = target_f.shape
            N = H * W
            # flatten patches: [B, N, C]
            tgt_flat = target_f.view(B, C, N).permute(0, 2, 1)
            prd_flat = pred_f.view(B, C, N).permute(0, 2, 1)
            # normalize so dot produces cosine
            tgt_normed = F.normalize(tgt_flat, dim=2)
            prd_normed = F.normalize(prd_flat, dim=2)
            # compute pairwise cosine-sims: [B, N, N]
            sim_T = torch.bmm(tgt_normed, tgt_normed.transpose(1, 2))
            sim_P = torch.bmm(prd_normed, prd_normed.transpose(1, 2))
            # mask out self-similarity
            diag_mask = torch.eye(N, device=device).bool().unsqueeze(0)  # [1, N, N]
            sim_T = sim_T.masked_fill(diag_mask, -1e9)
            sim_P = sim_P.masked_fill(diag_mask, -1e9)
            # top-k neighbor indices: [B, N, k_neigh]
            neigh_T = torch.topk(sim_T, k=k_neigh, dim=2).indices
            neigh_P = torch.topk(sim_P, k=k_neigh, dim=2).indices
            neigh_T_set = torch.zeros(B, N, N, dtype=torch.bool, device=sim_T.device)
            neigh_P_set = torch.zeros(B, N, N, dtype=torch.bool, device=sim_T.device)
            neigh_T_set.scatter_(2, neigh_T, True)
            neigh_P_set.scatter_(2, neigh_P, True)
            intersect = (neigh_T_set & neigh_P_set).sum(dim=2).float()  # [B,N]

            overlap_fracs = intersect / k_neigh # [B,N]
            # per-image mean/std
            overlap_mean_per_image = overlap_fracs.mean(dim=1)  # [B]
            overlap_std_per_image  = overlap_fracs.std(dim=1)   # [B]
            all_overlap_fracs.append(overlap_mean_per_image.detach().cpu())
            all_overlap_stds.append(overlap_std_per_image.detach().cpu())

        all_overlap_fracs = torch.cat(all_overlap_fracs)  # per-image means
        all_overlap_stds  = torch.cat(all_overlap_stds)   # per-image stds
        # aggregate across all batches
        metrics = {
            "cos_mean":           sum(cos_means) / len(cos_means),
            "mse":                sum(mse_vals)  / len(mse_vals),
            "norm_ratio_mean":    sum(ratio_means) / len(ratio_means),
            "pred_norm_mean":     sum(pred_norm_means) / len(pred_norm_means),
            "target_norm_mean":   sum(tgt_norm_means)   / len(tgt_norm_means),
            "neigh_overlap_mean": all_overlap_fracs.mean().item(),
            "cos_std":            sum(cos_stds)  / len(cos_stds),
            "norm_ratio_std":     sum(ratio_stds) / len(ratio_stds),
            "pred_norm_std":      sum(pred_norm_stds)  / len(pred_norm_stds),
            "target_norm_std":    sum(tgt_norm_stds)    / len(tgt_norm_stds),
            "neigh_overlap_std":  all_overlap_stds.std().item(),
        }

     
        return metrics


    @torch.no_grad()
    def sample_images_cfg_cls(
        self,
        img_hidden_states,
        scheduler,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 30,
        num_images_per_prompt: int = 1,
        return_tensor=False,
        guidance_scale: float = 3.0,
        **kwargs,
    ):
        
        device = img_hidden_states.device
        dtype = img_hidden_states.dtype

        img_hidden_states_null = torch.zeros_like(img_hidden_states, device=device, dtype=dtype)
        img_hidden_states_input = torch.cat([img_hidden_states_null, img_hidden_states], 0)

        batch_size = img_hidden_states.shape[0]
        latent_size = self.get_model().dit.config.input_size
        latent_channels = self.get_model().dit.config.in_channels
        D_sum  = self.get_model().dit.model.summary_token_dim

        sched_img = copy.deepcopy(scheduler)
        sched_sum = copy.deepcopy(scheduler)

        latents = randn_tensor(
            shape=(batch_size * num_images_per_prompt, latent_channels, latent_size, latent_size),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        # s_state  = randn_tensor((batch_size * num_images_per_prompt, D_sum, 1, 1), device=device, dtype=dtype)
        if not hasattr(self.get_model().dit.model, "noise_pool_W"):
            raise ValueError("Noise projection matrix not initialized.")
                    # 1) global average pool noise over H,W — re-scale to keep unit variance
                    #    mean over HW has var 1/(HW); multiply by sqrt(HW) to restore ~1.
        eps_chan = latents.mean(dim=(2, 3)) * math.sqrt(latent_size * latent_size)     # (B, C)
        eps_sum = eps_chan @ self.get_model().dit.model.noise_pool_W   # (B, summary_embedding_size)
        s_state = eps_sum.unsqueeze(-1).unsqueeze(-1)
        # s_state = randn_tensor(
        #     shape=(batch_size * num_images_per_prompt, D_sum, 1, 1), 
        #     generator=generator,
        #     device=device, 
        #     dtype=dtype
        # )

        # set step values
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        sched_img.set_timesteps(num_inference_steps, sigmas=sigmas)
        sched_sum.set_timesteps(num_inference_steps, sigmas=sigmas)

        # Repeat z_latents and conditions for each image per prompt
        img_hidden_states_input = img_hidden_states_input.repeat_interleave(num_images_per_prompt, dim=0)

        for t in sched_img.timesteps:
            # latent_model_input = latents.repeat(2, 1, 1, 1)
            noisy_sum = s_state.repeat(2,1,1,1)
            latent_model_input = latents.repeat(2, 1, 1, 1)

            if hasattr(sched_img, "scale_model_input"):
                latent_model_input = sched_img.scale_model_input(latent_model_input, t)
            if hasattr(sched_sum, "scale_model_input"):
                noisy_sum = sched_sum.scale_model_input(noisy_sum, t)

            # predict noise model_output
            noise_pred, summary_pred = self.get_model().dit(
                x=latent_model_input,
                timestep=t.unsqueeze(0).expand(latent_model_input.shape[0]).to(latent_model_input.device, torch.long),
                z_latents=img_hidden_states_input,
                noisy_sum=noisy_sum,
            )

            noise_pred_uncond, noise_pred = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            summary_pred = summary_pred.unsqueeze(-1).unsqueeze(-1)
            summary_pred_uncond, summary_pred = summary_pred.chunk(2)
            summary_pred = summary_pred_uncond + guidance_scale * (summary_pred - summary_pred_uncond)

            

            # perform guidance
            # noise_pred_uncond, noise_pred = noise_pred.chunk(2)
            # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            # compute previous image: x_t -> x_t-1
            latents = sched_img.step(noise_pred, t, latents).prev_sample
            s_state = sched_sum.step(summary_pred, t, s_state).prev_sample
        # samples = self.decode_latents(latents, return_tensor=return_tensor)
        # breakpoint()
        return s_state, latents

    @torch.no_grad()
    def evaluate_mapper_cls(
        self,                    
        dataloader,                
        device=None,              
        num_inference_steps: int = 30,
        k_neigh: int = 10,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        max_eval_samples: Optional[int] = 500
    ):
       
        self.get_model().dit.eval()
        self.get_model().eval()

        print(f"Evaluating mapper on device {device}")
        self.get_model().to(device)

        cos_means, cos_stds, mse_vals = [], [], []
        pred_norm_means, pred_norm_stds = [], []
        tgt_norm_means,  tgt_norm_stds  = [], []
        ratio_means, ratio_stds = [], []
        all_overlap_fracs, all_overlap_stds = [], []
        
        sum_cos_means, sum_mse_vals = [], []
        sum_pred_norm_means = []
        sum_tgt_norm_means = []
        sum_ratio_means = []

        sample_count = 0
        from tqdm import tqdm
        for batch in tqdm(dataloader, desc="Eval mapper"):
            if max_eval_samples is not None and sample_count >= max_eval_samples:
                break
            sample_count += len(batch['input_ids'])
            
            batch_on_dev: Dict[str, Any] = {}
            for k, v in batch.items():
                # print(f"Batch {k} type: {type(v)}")
                batch_on_dev[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            
            # for k,v in batch_on_dev.items():
            #     if isinstance(v, torch.Tensor):
            #         print(f"Batch {k} shape: {v.shape}, dtype: {v.dtype}, device: {v.device}")

            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            
            input_ids = batch_on_dev.get("input_ids", None)
            position_ids = batch_on_dev.get("position_ids", None)
            attention_mask = batch_on_dev.get("attention_mask", None)
            past_key_values = batch_on_dev.get("past_key_values", None) 
            labels = batch_on_dev.get("labels", None)
            gen_image = batch_on_dev.get("gen_image", None)
            und_image = batch_on_dev.get("und_image", None)
            grid_thw = batch_on_dev.get("grid_thw", None)
            i_s_pos = batch_on_dev.get("i_s_pos", None)
            image_sizes = batch_on_dev.get("image_sizes", None)
            
            
            
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, latents, sum_latents) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                gen_image,
                und_image,
                grid_thw,
                i_s_pos,
                image_sizes
            )
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = outputs[0]
            
            if labels is not None:
        
                # compute image loss
                # target_img_embeds = torch.clone(inputs_embeds.detach())[:,1:,:] # get target image emb
                img_loss_funct = torch.nn.MSELoss()
                # img_hidden_states = self.get_model().down_projector(hidden_states[:,-self.get_n_query():,:])
                img_hidden_states = []
                
                for b in range(hidden_states.shape[0]):
                    img_hidden_states.append(hidden_states[b,i_s_pos[b]:i_s_pos[b]+self.get_n_query(),:])
                img_hidden_states = torch.stack(img_hidden_states,dim=0)
                img_hidden_states = self.get_model().down_projector(img_hidden_states)

            if not hasattr(self, "_inference_scheduler"):
                self._inference_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                    "Alpha-VLLM/Lumina-Next-SFT-diffusers", subfolder="scheduler"
            )
            scheduler = self._inference_scheduler
            summary_pred, pred = self.sample_images_cfg_cls(
                    img_hidden_states,
                    scheduler=scheduler,
                    num_inference_steps=num_inference_steps,
                    num_images_per_prompt=1,
                    generator=None,
                ) 

            pred_f = pred.float()
            target_f = latents.float()
           
            summary_pred = summary_pred.squeeze(-1).squeeze(-1)  # [B,3072]
            assert summary_pred.shape[1] == 3072, f"Expected summary_pred to have shape [B, 3072], but got {summary_pred.shape}"
            summary_pred = summary_pred.float()
            sum_target_f = sum_latents.float()

            # 4) compute per‐patch cosine‐map
            cos_map = F.cosine_similarity(pred_f, target_f, dim=1, eps=1e-8)  # [B,H,W]
            cos_means.append(cos_map.mean(dim=(1,2)).mean().item())  # avg of per-image means
            cos_stds.append(cos_map.std(dim=(1,2)).mean().item())    # avg of per-image stds

            mse_vals.append(F.mse_loss(pred_f, target_f, reduction="mean").item())

            # norms + ratio (per-image)
            pred_norm_map = pred_f.norm(dim=1)     # [B,H,W]
            target_norm_map = target_f.norm(dim=1) # [B,H,W]
            ratio_map = pred_norm_map / (target_norm_map + 1e-8)

            pred_norm_means.append(pred_norm_map.mean(dim=(1,2)).mean().item())
            pred_norm_stds.append(pred_norm_map.std(dim=(1,2)).mean().item())
            tgt_norm_means.append(target_norm_map.mean(dim=(1,2)).mean().item())
            tgt_norm_stds.append(target_norm_map.std(dim=(1,2)).mean().item())
            ratio_means.append(ratio_map.mean(dim=(1,2)).mean().item())
            ratio_stds.append(ratio_map.std(dim=(1,2)).mean().item())

            
            sum_cos_map = F.cosine_similarity(summary_pred, sum_target_f, dim=1, eps=1e-8)  # [B,1,1]
            sum_cos_means.append(sum_cos_map.mean().item())  # avg of per-image means
            sum_mse_vals.append(F.mse_loss(summary_pred, sum_target_f, reduction="mean").mean().item())
            sum_pred_norm = summary_pred.norm(dim=1)     # [B,1,1]
            sum_target_norm = sum_target_f.norm(dim=1) # [B,1,1]
            sum_ratio = sum_pred_norm / (sum_target_norm + 1e-8)
            sum_pred_norm_means.append(sum_pred_norm.mean().item())
            sum_tgt_norm_means.append(sum_target_norm.mean().item())
            sum_ratio_means.append(sum_ratio.mean().item())


            # 6) Neighborhood preservation
            B, C, H, W = target_f.shape
            N = H * W
            # flatten patches: [B, N, C]
            tgt_flat = target_f.view(B, C, N).permute(0, 2, 1)
            prd_flat = pred_f.view(B, C, N).permute(0, 2, 1)
            # normalize so dot produces cosine
            tgt_normed = F.normalize(tgt_flat, dim=2)
            prd_normed = F.normalize(prd_flat, dim=2)
            # compute pairwise cosine-sims: [B, N, N]
            sim_T = torch.bmm(tgt_normed, tgt_normed.transpose(1, 2))
            sim_P = torch.bmm(prd_normed, prd_normed.transpose(1, 2))
            # mask out self-similarity
            diag_mask = torch.eye(N, device=device).bool().unsqueeze(0)  # [1, N, N]
            sim_T = sim_T.masked_fill(diag_mask, -1e9)
            sim_P = sim_P.masked_fill(diag_mask, -1e9)
            # top-k neighbor indices: [B, N, k_neigh]
            neigh_T = torch.topk(sim_T, k=k_neigh, dim=2).indices
            neigh_P = torch.topk(sim_P, k=k_neigh, dim=2).indices
            neigh_T_set = torch.zeros(B, N, N, dtype=torch.bool, device=sim_T.device)
            neigh_P_set = torch.zeros(B, N, N, dtype=torch.bool, device=sim_T.device)
            neigh_T_set.scatter_(2, neigh_T, True)
            neigh_P_set.scatter_(2, neigh_P, True)
            intersect = (neigh_T_set & neigh_P_set).sum(dim=2).float()  # [B,N]

            overlap_fracs = intersect / k_neigh # [B,N]
            # per-image mean/std
            overlap_mean_per_image = overlap_fracs.mean(dim=1)  # [B]
            overlap_std_per_image  = overlap_fracs.std(dim=1)   # [B]
            all_overlap_fracs.append(overlap_mean_per_image.detach().cpu())
            all_overlap_stds.append(overlap_std_per_image.detach().cpu())

        all_overlap_fracs = torch.cat(all_overlap_fracs)  # per-image means
        all_overlap_stds  = torch.cat(all_overlap_stds)   # per-image stds
        # aggregate across all batches
        metrics = {
            "cos_mean":           sum(cos_means) / len(cos_means),
            "mse":                sum(mse_vals)  / len(mse_vals),
            "norm_ratio_mean":    sum(ratio_means) / len(ratio_means),
            "pred_norm_mean":     sum(pred_norm_means) / len(pred_norm_means),
            "target_norm_mean":   sum(tgt_norm_means)   / len(tgt_norm_means),
            "neigh_overlap_mean": all_overlap_fracs.mean().item(),
            "cos_std":            sum(cos_stds)  / len(cos_stds),
            "norm_ratio_std":     sum(ratio_stds) / len(ratio_stds),
            "pred_norm_std":      sum(pred_norm_stds)  / len(pred_norm_stds),
            "target_norm_std":    sum(tgt_norm_stds)    / len(tgt_norm_stds),
            "neigh_overlap_std":  all_overlap_stds.std().item(),
            "sum_cos_mean":       sum(sum_cos_means) / len(sum_cos_means),
            "sum_mse":            sum(sum_mse_vals)  / len(sum_mse_vals),
            "sum_norm_ratio_mean":sum(sum_ratio_means) / len(sum_ratio_means),
            "sum_pred_norm_mean": sum(sum_pred_norm_means) / len(sum_pred_norm_means),
            "sum_target_norm_mean":sum(sum_tgt_norm_means)   / len(sum_tgt_norm_means),
        }

     
        return metrics

AutoConfig.register("blip3o_qwen", blip3oQwenConfig)
AutoModelForCausalLM.register(blip3oQwenConfig, blip3oQwenForCausalLM)




















