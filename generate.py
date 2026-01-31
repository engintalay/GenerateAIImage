import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

import os
os.makedirs("outputs", exist_ok=True)

if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype
)

pipe = pipe.to(device)
pipe.enable_attention_slicing()

# referans yüz
face_image = Image.open("refs/face.jpg").convert("RGB")

prompt = (
    "photo of the same person, sitting at an office desk, "
    "drinking coffee, natural light, realistic photography"
)

image = pipe(
    prompt=prompt,
    image=face_image,
    strength=0.6,
    guidance_scale=7.5,
    num_inference_steps=30
).images[0]

image.save("outputs/test_1.png")
print("✔ output saved to outputs/test_1.png")
