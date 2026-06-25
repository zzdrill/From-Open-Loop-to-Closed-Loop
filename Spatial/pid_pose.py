import numpy as np
import torch
from PIL import Image
from diffusers.utils import load_image
from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler
from controlnet_aux import OpenposeDetector
from controlnet_aux.open_pose import util
from controlnet_aux.open_pose.body import Keypoint

# ControlNet Union pipeline/model comes from the ControlNetPlus repo; change to your local path as needed
import sys
sys.path.insert(0, "/home1/zzdrill/Projects/ControlNetPlus")
from models.controlnet_union import ControlNetModel_Union
from pipeline.pipeline_controlnet_union_sd_xl import StableDiffusionXLControlNetUnionPipeline

image = ""
output_name = ""
device = "cuda"
seed = 42
width = height = 1024

openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

# Reference pose (open-loop setpoint): extracted once from the reference image as the PID setpoint
ref_image = load_image(image)
ref_pose_img, ref_poses = openpose(ref_image, detect_resolution=1024, image_resolution=1024,
                                   include_hand=False, include_face=False)
orig_kp = np.array([[k.x, k.y] if k is not None else [0, 0]
                    for k in ref_poses[0].body.keypoints], dtype=np.float32)

controlnet = ControlNetModel_Union.from_pretrained(
    "xinsir/controlnet-union-sdxl-1.0", torch_dtype=torch.float16, use_safetensors=True
).to(device)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler"
)
pipe = StableDiffusionXLControlNetUnionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae,
    torch_dtype=torch.float16, scheduler=scheduler,
).to(device)

prompt = "A side portrait of young man with casual shirt"
negative_prompt = "longbody, lowres, bad anatomy, worst quality, low quality"

max_iter = 20
kp, ki, kd = 0.05, 0.02, 0.01
sum_delta, delta0 = 0, 0
current_pose_img = np.array(ref_pose_img)

for i in range(max_iter):
    generate_image = pipe(
        prompt=[prompt], image_list=[Image.fromarray(current_pose_img), 0, 0, 0, 0, 0],
        negative_prompt=[negative_prompt], width=width, height=height,
        num_inference_steps=25, guidance_scale=5, controlnet_conditioning_scale=0.8,
        union_control=True, union_control_type=torch.Tensor([1, 0, 0, 0, 0, 0]),
        generator=seed,
    ).images[0]
    generate_image.save(output_name, lossless=True, quality=100)
    _, gen_poses = openpose(generate_image, detect_resolution=1024, image_resolution=1024,
                            include_hand=False, include_face=False)
    gen_kp = np.array([[k.x, k.y] if k is not None else [0, 0]
                       for k in gen_poses[0].body.keypoints], dtype=np.float32)
    delta = orig_kp - gen_kp
    sum_delta += delta
    corrected = orig_kp + kp * delta + ki * sum_delta + kd * (delta - delta0)
    delta0 = delta
    corrected_kps = [Keypoint(x=float(x), y=float(y), score=1.0) for x, y in corrected]
    current_pose_img = util.draw_bodypose(np.zeros((height, width, 3), dtype=np.uint8), corrected_kps)
