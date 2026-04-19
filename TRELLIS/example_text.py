import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import imageio
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
pipeline.cuda()


# caption  = "a chair looking like an avocado"
# caption = "A modern chair with a light wood frame, visible grain, four tapered legs, a flat rectangular wood seat, and a white plastic backrest and armrests integrated into one piece, connected by metal fasteners."
# caption = "A vertically standing, rectangular refrigerator in metallic silver with two full-height doors, central black handles, a water/ice dispenser, black detailing on edges, paneled sides, and geometric black designs suggesting an industrial style."
caption = "A modern rectangular three-seater sofa with clean lines, light gray fabric upholstery, thick armrests, uniform backrest, three equal-sized seat cushions, six pillows (four dark gray, two light gray), and small wooden legs, featuring a soft woven texture."

# Run the pipeline
outputs = pipeline.run(
    caption,
    seed=1,
    # Optional parameters
    # sparse_structure_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 7.5,
    # },
    # slat_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 7.5,
    # },
)
# outputs is a dictionary containing generated 3D assets in different formats:
# - outputs['gaussian']: a list of 3D Gaussians
# - outputs['radiance_field']: a list of radiance fields
# - outputs['mesh']: a list of meshes

# Render the outputs
video = render_utils.render_video(outputs['gaussian'][0])['color']
imageio.mimsave("text_gs.mp4", video, fps=30)
video = render_utils.render_video(outputs['radiance_field'][0])['color']
imageio.mimsave("text_rf.mp4", video, fps=30)
video = render_utils.render_video(outputs['mesh'][0])['normal']
imageio.mimsave("text_mesh.mp4", video, fps=30)

# GLB files can be extracted from the outputs
glb = postprocessing_utils.to_glb(
    outputs['gaussian'][0],
    outputs['mesh'][0],
    # Optional parameters
    simplify=0.95,          # Ratio of triangles to remove in the simplification process
    texture_size=1024,      # Size of the texture used for the GLB
)
glb.export("text.glb")

# Save Gaussians as PLY files
outputs['gaussian'][0].save_ply("text.ply")
