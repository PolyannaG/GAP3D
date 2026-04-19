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



# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import imageio
from PIL import Image

from pathlib import Path, sys
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


from TRELLIS.trellis.pipelines import BlipTrellisImageTo3DPipeline, TrellisTextTo3DPipeline
from TRELLIS.trellis.utils import render_utils, postprocessing_utils

from eval.model_helpers import load_weights_exact_blip, ModelArguments, DataArguments


class SingleT2IDataset(Dataset):
    def __init__(self, image_path: str, caption: str, tokenizer, data_args: DataArguments):
        self.image_path = image_path
        self.caption = caption
        self.tokenizer = tokenizer
        self.da = data_args

    def __len__(self): 
        return 1

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

        return {
            "input_ids": tokd["input_ids"][0],
            "labels":    tokd["labels"][0],
            "gen_image": gen_px,     # (B=1, C, H, W)
            "ids":       "single",   # mimic dataset ID
        }



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

    model_args = ModelArguments(model_name_or_path=ckpt)
    data_args = DataArguments()
   
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

def generate_image_embeddings(
    model,
    tokenizer,
    data_args,
    image_path: str,
    caption: str,
    device: str = "cuda",
    seed: int = 42,
):

    # 4) Make single-image dataset + collator + loader (exactly like train/eval)
    ds = SingleT2IDataset(image_path=image_path, caption=caption, tokenizer=tokenizer, data_args=data_args)
    collator = DataCollatorForSupervisedDataset(n_query=data_args.n_query, tokenizer=tokenizer)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collator, num_workers=0)

    # get batch from loader (only 1 item)
    batch = next(iter(dl))

   
    with torch.no_grad():
        (summary_embed, reg_embeds, patch_embeds), metrics = model.generate_embeddings_dino(
            batch=batch,
            device=device,
            num_inference_steps=50,
            cfg=True,  
        )

    # Convert patch embeds from B,S,H,W  to B,H*W,S
    B, S, H, W = patch_embeds.shape
    patch_embeds = patch_embeds.permute(0, 2, 3, 1).reshape(B, H*W, S)
    # Bring B,D sum embed, B,4,D reg embeds and B,H*W,D patch embeds in common embedding array
    summary_pred = summary_embed.unsqueeze(1)          # B,1,D
    image_embeds = torch.cat([summary_pred, reg_embeds, patch_embeds], dim=1)  # B,1+4+H*W,D
    print("Image embeds shape:", image_embeds.shape)

    print("=== EVAL (single image) METRICS ===")
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
    base_dir = "run_outputs/test"
    
    use_original_trellis = False

    # Load a pipeline from a model folder or a Hugging Face model hub.
    
    # captions = [
    #     "Vintage camera with leather case.",
    #     "Two-story brick house with red roof and fence.",
    #     "A rustic log cabin with a stone chimney and a wooden porch.",
    #     "Portable transistor radio, dark cover, speaker grille, brand logo on front.",
    #     "Futuristic red toy blaster with transparent magazine.",
    #     "Blocky, orange and teal robot with articulated limbs.",
    #     "Metallic dog-like robot with articulated legs and futuristic design elements.",
    #     "Yellow and black bulldozer with movable front blade.",
    #     "Wooden horse cart with wheels and a handle, a medieval transport vehicle.",
    #     "A sleek, futuristic silver and blue spaceship model.",
    #     "Ship with copper and brown hues, intricate deck details.",
    #     "A stylized, cartoonish rocket with a red dome top and black antenna, teal cylindrical middle section with red bands and black connectors.",
    #     "A weather-worn vintage delivery van with a boxy shape, a rusted faded green finish, square windows, rusty roof rack.",
    #     "A Victorian mansion made of stone bricks with ornate trim, bay windows, and a wraparound porch.",
    #     "A wooden bookshelf with carved details and adjustable shelves.",
    #     "A wooden rocking chair with a woven seat and back.",
    #     "Vintage green computer monitor.",
    #     "Sci-fi inspired silver and blue toy gun with intricate design.",
    #     "The tree has stylized, rounded canopies made up of layered, scale-like leaves in shades of green. Its trunk is twisted.",
    #     "The train carriage has a classic, vintage design with a dark, rounded roof, teal exterior, detailed windows, and red wheels.",
    #     "Carved wooden chess piece. (queen)",
    #     "Ceramic mug with a crack.",
    #     "Dark leather suitcase with brass latches.",
    #     "Geometric metal sculpture with angular edges.",
    #     "Rustic lantern with a flickering flame.",
    # ]

    # captions = [
    #     "A vintage black mechanical typewriter with chrome accents sits on a wooden desk, its round keys slightly faded from years of use. The metal arms and platen roller show subtle wear, and the ribbon spool housing has small scratches that catch the light. The typewriter's carriage return lever curves gracefully over the top, and the engraved manufacturer's nameplate gleams with a dull metallic sheen. Dust and patina give it a sense of history, as though countless letters and manuscripts were once typed on it.",
    #     "A futuristic electric motorcycle with a sleek, aerodynamic body painted in metallic graphite and cobalt blue. Its wheels are hubless, glowing faintly along the rim edges, and the frame has seamless carbon fiber panels with embedded LED strips that pulse in sync with acceleration. The digital dashboard floats holographically above the handlebars, displaying speed, power output, and navigation data. Small vents and fins suggest advanced cooling systems, while the seat is sculpted for a single rider in a racing posture.",
    #     "A tall wooden grandfather clock crafted from dark mahogany with ornate gold details on its clock face. The pendulum swings behind a glass pane etched with floral patterns, and the chimes are housed in a brass casing that gleams softly. The body has carved embellishments along the edges, and the base features clawed feet resting on a stone floor. Dusty and slightly aged, the clock's hands are frozen at five past seven, hinting at an old, forgotten history.",
    #     "A large, industrial-style cargo drone designed for off-world transport. Its rectangular body is constructed from reinforced alloy panels with visible rivets and caution markings. Four rotors extend from articulated arms that fold inward for storage, each tipped with blue glowing propulsion rings. The cargo bay features mechanical clamps and magnetic locks for holding crates. Scratches and burn marks suggest heavy use in rugged environments, while its front camera sensor glows with a soft red tracking light.",
    #     "A polished steel knight's helmet from the medieval era, complete with a movable visor and ornate engravings. The surface reflects ambient light unevenly, showing faint battle scars and patches of tarnish. Gold trim runs along the edges, and the visor slits are narrow and intimidating. A red plume made of dyed horsehair flows from the top, slightly frayed at the ends. Resting on a stone pedestal, the helmet evokes a sense of nobility, valor, and long-forgotten battles.",
    #     "A retro-style arcade cabinet painted in neon pink and electric blue with stylized pixel art characters on its sides. The screen displays a looping 8-bit title animation, and the joystick has a worn red ball top from years of gameplay. Two yellow buttons glow faintly, while the coin slot still has a small \"Insert Coin\" light. The cabinet edges are slightly scuffed, revealing wood beneath the paint, and the back panel hums faintly with the sound of old CRT electronics.",
    #     "A grand crystal chandelier with cascading tiers of faceted glass droplets hanging from a golden frame. Each droplet refracts the light into tiny rainbows across nearby walls, and the central column is made of sculpted brass with floral motifs. Candle-like bulbs sit on curling arms that extend outward in a spiral pattern. Dust has settled on the upper surfaces, and the entire fixture sways gently as if stirred by an unseen breeze in an opulent ballroom.",
    #     "A strange, bioluminescent alien plant contained within a transparent terrarium made of hexagonal glass panels. Its tendrils curl upward, covered in small, pulsating orbs that emit a greenish glow. The central stem is translucent, revealing inner fluid movement like veins of light. The soil beneath it appears crystalline, with scattered metallic fragments embedded throughout. Condensation dots the inner glass surface, distorting reflections of the glowing leaves, giving the entire structure a surreal, otherworldly aura.",
    #     "A majestic steampunk airship constructed from brass, copper, and dark oak planks. Large propellers spin slowly at its sides, powered by exposed mechanical gears and pistons. The balloon canopy is made of reinforced canvas stitched with leather bands and patched in places. Brass pipes vent small puffs of steam near the helm, where a large wooden wheel controls direction. Lanterns hang from the underside, flickering warmly against the backdrop of clouds tinged with sunset colors.",
    #     "An ancient humanoid statue carved from weathered limestone, standing partially buried in sand. Its surface is eroded but still shows faint markings resembling inscriptions or runes. Moss and vines have begun to overtake its base, and cracks run across the torso and arms. The statue's face is stoic, with hollow eyes that seem to gaze into the distance. The lighting emphasizes rough textures, casting long shadows that evoke a feeling of forgotten civilizations and timeless silence."
    # ]

    captions = [
        "A vintage mechanical typewriter made of cast metal with a black enamel finish. The frame includes a return lever, roller platen, and two ribbon spools mounted on top. Each round key has a metal rim and a glass cap with faded lettering. The space bar is slightly worn. The machine rests on four rubber feet and includes a carriage release knob, adjustable paper guide, and visible linkage arms beneath the key bed. Surface oxidation is visible around screw heads and levers.",
        "An electric motorcycle with a streamlined carbon-fiber frame and hubless wheels. The tires have a thin illuminated rim strip. The chassis includes side vents, aerodynamic fairings, and integrated LED indicators. The seat is contoured synthetic leather with a low backrest. Handlebar controls include digital throttle sensors, brake levers, and a transparent holographic dashboard displaying telemetry. The front suspension is an inverted fork design with visible dampers and sensor modules near the wheel hub.",
        "A freestanding wooden grandfather clock made from dark-stained oak. The upper section houses a circular clock face with brass Roman numerals and rotating hands. A glass panel covers the pendulum chamber, revealing a polished brass pendulum and two cylindrical counterweights suspended by chains. The base includes molded trim and decorative feet. The back panel is removable for maintenance, exposing the internal mechanical gear assembly. The clock surface shows fine scratches and dust accumulation.",
        "A heavy-duty quadcopter cargo drone with a rectangular metal chassis and retractable landing struts. The frame supports four motor arms with enclosed ducted fans for vertical lift. The top surface features two antennae, a GPS module, and a signal light. The underside includes a detachable cargo bay with mechanical clamps and magnetic retention locks. Power connectors and cooling vents are visible on the side panels. The housing has minor abrasion marks and heat discoloration near the exhaust ports.",
        "A medieval-style steel helmet with a movable front visor and hinged cheek guards. The visor has narrow horizontal slits for vision and small perforations for airflow. Rivets secure the joint plates, and the surface shows hammering texture from manual forging. The lower rim includes attachment loops for a neck guard. The metal has uneven tarnish and micro-scratches from polishing. The interior is lined with aged leather padding fixed by brass rivets along the crown edge.",
        "An upright arcade cabinet constructed from laminated plywood with a black and red color scheme. The front panel includes a joystick, two convex push buttons, and a coin slot assembly. The display area houses a 19-inch CRT monitor protected by a tinted acrylic cover. Speaker grilles are located above the marquee light box. The rear panel has a removable maintenance door for access to wiring and PCB boards. Surface edges are slightly chipped, revealing the wood core material.",
        "A suspended chandelier composed of multiple glass crystal strands attached to a circular brass frame. Each crystal prism is faceted and connected with small metal rings. The central column is hollow brass tubing carrying electrical wiring to eight candle-shaped bulb sockets. Mounting hardware includes a threaded rod, canopy plate, and chain link suspension. Minor dust and fingerprints are visible on the glass surfaces. The structure weighs approximately 12 kilograms and measures about one meter in diameter.",
        "A contained bioluminescent plant specimen inside a transparent glass terrarium with a hexagonal metal frame. The plant has multiple semi-translucent stems with nodules emitting faint green light. The base substrate is granular mineral soil with metallic particles. A humidity sensor and small ventilation port are integrated into the lid. Internal condensation partially obscures visibility near the top. Power cables for lighting and sensor control exit through a sealed grommet on the rear panel.",
        "A rigid-frame airship with brass and copper exterior plating and a cylindrical gas envelope on top. The propulsion system consists of twin rear propellers powered by external piston engines with exposed connecting rods. The gondola features glass windows, control levers, and a steering wheel linked to rudder assemblies via mechanical cables. Exhaust stacks are mounted along the upper hull, releasing steam vapor. Rivet lines, copper piping, and pressure gauges are visible throughout the superstructure.",
        "A humanoid stone statue carved from limestone, approximately two meters tall. The surface texture is rough with visible chisel marks and erosion on the extremities. The head features simplified facial details, including hollow eyes and a flat nose ridge. Moss and lichen growth cover the lower torso. The statue stands on an integrated square base slab with minor cracks. Material density and color variations indicate partial weathering, typical of outdoor exposure over extended time."
    ]


    if not use_original_trellis:

        model, tokenizer, data_args, device = blip_init(ckpt=ckpt_dir)

        trellis_pipeline = BlipTrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
        trellis_pipeline.cuda()
        image = "assets/avocado_chair.png"

       
    

        for i, caption in enumerate(captions):
            print(f"Generating for caption {i}: {caption}")
            image_embeds = generate_image_embeddings(
                model=model,
                tokenizer=tokenizer,
                data_args=data_args,
                image_path=image,
                caption=caption,
                device=device,
                seed = 42 + i,
            )

            save_dir = f"{base_dir}/asset_{i}"
            os.makedirs(save_dir, exist_ok=True)
            trellis_run_single_image_forward(pipeline=trellis_pipeline, image_embeds=image_embeds, save_dir=save_dir, seed = 42 + i)
            # Save caption used
            with open(f"{save_dir}/caption.txt", "w") as f:
                f.write(caption)

    else:
        trellis_pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
        trellis_pipeline.cuda()

        for i, caption in enumerate(captions):
            save_dir = f"{base_dir}/asset_{i}"
            os.makedirs(save_dir, exist_ok=True)
            trellis_run_single_text_forward(pipeline=trellis_pipeline, caption=caption, save_dir=save_dir, seed = 42 + i)
            # Save caption used
            with open(f"{save_dir}/caption.txt", "w") as f:
                f.write(caption)
    

    # Additional examples can be run by changing the image path and caption above.
