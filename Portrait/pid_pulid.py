import numpy as np
import torch
from PIL import Image
from pulid import attention_processor as attention
from pulid.pipeline import PuLIDPipeline
from pulid.utils import resize_numpy_image_long, seed_everything

image = ""
output_name = ""
device = "cuda"
seed = 42

pipeline = PuLIDPipeline()
attention.NUM_ZERO = 8
attention.ORTHO = False
attention.ORTHO_v2 = True   # fidelity mode

# Reference identity (open-loop setpoint): PuLID id_embedding is [2, num_tokens, dim] (0=uncond, 1=cond)
main_np = resize_numpy_image_long(np.array(Image.open(image).convert("RGB")), 1024)
origin_id_embedding = pipeline.get_id_embedding(main_np)

prompt = "A portrait of a young woman, professional photo"
negative_prompt = "flaws in the eyes, flaws in the face, lowres, low quality, worst quality, deformed, ugly, disfigured"

max_iter = 20
kp, ki, kd = 0.3, 0.09, 0.01
sum_delta, delta0 = 0, 0
current_id_embedding = origin_id_embedding.clone()

for i in range(max_iter):
    seed_everything(seed)
    generate_image = pipeline.inference(
        prompt, (1, 1024, 1024), negative_prompt, current_id_embedding, 0.8, 1.2, 4,
    )[0]
    generate_image.save(output_name, quality=95)
    gen_id_embedding = pipeline.get_id_embedding(np.array(generate_image))
    delta = origin_id_embedding[1:2] - gen_id_embedding[1:2]
    sum_delta += delta
    current_id_embedding[1:2] = origin_id_embedding[1:2] + kp * delta + ki * sum_delta + kd * (delta - delta0)
    delta0 = delta
