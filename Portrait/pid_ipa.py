import cv2
import numpy as np
from insightface.app import FaceAnalysis
import torch
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from ip_adapter.ip_adapter_faceid_separate import IPAdapterFaceIDXL

base_model_path = "SG161222/RealVisXL_V4.0"
ip_ckpt = "ip-adapter-faceid-portrait_sdxl.bin"
device = "cuda"

app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(512, 512))

image = ""
output_name = ""

img = cv2.imread(image)
faces = app.get(img)
faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0).unsqueeze(0)
origin_faceid_embeds = faceid_embeds.clone()

n_cond = faceid_embeds.shape[1]

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    feature_extractor=None,
    safety_checker=None
)

# load ip-adapter
ip_model = IPAdapterFaceIDXL(pipe, ip_ckpt, device, num_tokens=16, n_cond=n_cond)

# generate image
prompt = "A side portrait of young man with casual shirt"
negative_prompt = "beard, bald, monochrome, lowres, bad anatomy, worst quality, low quality, blurry"

max_iter = 20
kp,ki,kd = 0.3, 0.05, 0.03
sum_delta,delta0 = 0, 0

for i in range(max_iter):
    generate_image = ip_model.generate(
        prompt=prompt, negative_prompt=negative_prompt, faceid_embeds=faceid_embeds,
        num_samples=1, width=1024, height=1024,
        guidance_scale=5, num_inference_steps=30, seed=1234,
    )[0]
    generate_image.save(output_name, lossless=True, quality=100)
    generate_image = np.array(generate_image)
    generate_faces = app.get(generate_image)
    gen_faceid_embeds = torch.from_numpy(generate_faces[0].normed_embedding).unsqueeze(0).unsqueeze(0)
    delta = origin_faceid_embeds - gen_faceid_embeds
    sum_delta += delta
    faceid_embeds = origin_faceid_embeds + kp*delta + ki*sum_delta + kd*(delta - delta0)
    delta0 = delta






