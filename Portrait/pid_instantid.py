import cv2
import numpy as np
import torch
from PIL import Image
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_instantid_full import StableDiffusionXLInstantIDPipeline, draw_kps

image = ""
output_name = ""
device = "cuda"
seed = 42


def resize_img(img, size=1024):
    """Resize keeping aspect ratio to ~size, with sides aligned to 64 (required by InstantID ControlNet)."""
    w, h = img.size
    r = size / max(w, h)
    w, h = (round(r * w) // 64) * 64, (round(r * h) // 64) * 64
    return img.resize((w, h), Image.BILINEAR)


face_app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(512, 512))

controlnet = ControlNetModel.from_pretrained('./checkpoints/ControlNetModel', torch_dtype=torch.float16)
pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0', controlnet=controlnet, torch_dtype=torch.float16,
).to(device)
pipe.load_ip_adapter_instantid('./checkpoints/ip-adapter.bin')

# Reference identity (open-loop setpoint): antelopev2 face embedding + facial landmark map
face_image = resize_img(load_image(image))
img_np = cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR)
face = face_app.get(img_np)[0]
origin_face_emb = face['embedding']
face_kps = draw_kps(face_image, face['kps'])

prompt = "A side portrait of young man with casual shirt"
negative_prompt = "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, glitch, deformed, ugly, disfigured"

max_iter = 20
kp, ki, kd = 0.3, 0.09, 0.01
sum_delta, delta0 = 0, 0
current_face_emb = origin_face_emb.copy()

for i in range(max_iter):
    torch.manual_seed(seed)
    generate_image = pipe(
        prompt=prompt, negative_prompt=negative_prompt, image_embeds=current_face_emb, image=face_kps,
        control_mask=None, controlnet_conditioning_scale=0.8, ip_adapter_scale=0.8,
        num_inference_steps=25, guidance_scale=5,
    ).images[0]
    generate_image.save(output_name, quality=95)
    gen_img_np = cv2.cvtColor(np.array(generate_image), cv2.COLOR_RGB2BGR)
    gen_face_emb = face_app.get(gen_img_np)[0]['embedding']
    delta = origin_face_emb - gen_face_emb
    sum_delta += delta
    current_face_emb = origin_face_emb + kp * delta + ki * sum_delta + kd * (delta - delta0)
    delta0 = delta
