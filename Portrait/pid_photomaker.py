import os
import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from diffusers import EulerDiscreteScheduler
from photomaker import PhotoMakerStableDiffusionXLPipeline, FaceAnalysis2, analyze_faces

image = ""
output_name = ""
device = "cuda"
seed = 42

face_detector = FaceAnalysis2(providers=['CUDAExecutionProvider'], allowed_modules=['detection', 'recognition'])
face_detector.prepare(ctx_id=0, det_size=(640, 640))

photomaker_ckpt = hf_hub_download(repo_id="TencentARC/PhotoMaker-V2", filename="photomaker-v2.bin", repo_type="model")
pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0", torch_dtype=torch.float16
).to(device)
pipe.load_photomaker_adapter(
    os.path.dirname(photomaker_ckpt), subfolder="",
    weight_name=os.path.basename(photomaker_ckpt), trigger_word="img"
)
pipe.fuse_lora()
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

# Reference identity (open-loop setpoint): 512-d id embedding from FaceAnalysis2; "img" is the PhotoMaker trigger word
input_id_images = [Image.open(image).convert("RGB")]
img_np = np.array(input_id_images[0])[:, :, ::-1]  # RGB -> BGR
origin_id_embed = torch.from_numpy(analyze_faces(face_detector, img_np)[0]['embedding'])

prompt = "A young man img, casual shirt"
negative_prompt = "(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth"

max_iter = 20
kp, ki, kd = 0.3, 0.09, 0.01
sum_delta, delta0 = 0, 0
current_id_embed = origin_id_embed.clone()

for i in range(max_iter):
    torch.manual_seed(seed)
    generate_image = pipe(
        prompt, negative_prompt=negative_prompt, input_id_images=input_id_images,
        id_embeds=current_id_embed.unsqueeze(0), num_images_per_prompt=1, start_merge_step=10,
    ).images[0]
    generate_image.save(output_name, quality=95)
    gen_np = np.array(generate_image)[:, :, ::-1]
    gen_id_embed = torch.from_numpy(analyze_faces(face_detector, gen_np)[0]['embedding'])
    delta = origin_id_embed - gen_id_embed
    sum_delta += delta
    current_id_embed = origin_id_embed + kp * delta + ki * sum_delta + kd * (delta - delta0)
    delta0 = delta
