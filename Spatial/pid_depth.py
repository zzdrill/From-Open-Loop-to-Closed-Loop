import cv2
import numpy as np
import torch
from PIL import Image
from diffusers.utils import load_image
from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler
from controlnet_aux import MidasDetector

# ControlNet Union pipeline/model comes from the ControlNetPlus repo; change to your local path as needed
import sys
sys.path.insert(0, "/home1/zzdrill/Projects/ControlNetPlus")
from models.controlnet_union import ControlNetModel_Union
from pipeline.pipeline_controlnet_union_sd_xl import StableDiffusionXLControlNetUnionPipeline

image = ""
output_name = ""
device = "cuda"
seed = 57

midas = MidasDetector.from_pretrained("lllyasviel/Annotators")

# Reference depth (open-loop setpoint): extracted once from the reference image as the PID setpoint
ref_image = load_image(image)
ref_depth, _ = midas(ref_image, output_type='cv2', return_float_01=True)
ref_depth = ref_depth.astype(np.float32)
h, w = ref_depth.shape[:2]
ratio = np.sqrt(1024. * 1024. / (w * h))
target_w, target_h = int(w * ratio), int(h * ratio)
ref_depth = cv2.resize(ref_depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

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

prompt = "A portrait of a young woman standing in a garden"
negative_prompt = "longbody, lowres, bad anatomy, worst quality, low quality, blurry"

max_iter = 20
kp, ki, kd = 0.01, 0.005, 0.005
sum_delta, delta0 = 0, 0
current_depth = ref_depth.copy()

for i in range(max_iter):
    depth_pil = Image.fromarray((np.clip(current_depth, 0, 1) * 255).astype(np.uint8))
    generate_image = pipe(
        prompt=[prompt], image_list=[0, depth_pil, 0, 0, 0, 0],
        negative_prompt=[negative_prompt], width=target_w, height=target_h,
        num_inference_steps=30, guidance_scale=5, generator=seed,
        union_control=True, union_control_type=torch.Tensor([0, 1, 0, 0, 0, 0]),
    ).images[0]
    generate_image.save(output_name, lossless=True, quality=100)
    gen_depth, _ = midas(generate_image, output_type='cv2', return_float_01=True)
    gen_depth = cv2.resize(gen_depth.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    delta = ref_depth - gen_depth
    sum_delta += delta
    current_depth = ref_depth + kp * delta + ki * sum_delta + kd * (delta - delta0)
    delta0 = delta
