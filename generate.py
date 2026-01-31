import torch
from diffusers import StableDiffusionImg2ImgPipeline, AutoencoderKL
from PIL import Image

import os
os.makedirs("outputs", exist_ok=True)

if torch.cuda.is_available():
    device = "cuda"
    # Force float32 for GTX 16xx series to avoid NaNs
    dtype = torch.float32
    print("✔ CUDA detected – using NVIDIA GPU (forcing float32)")
else:
    device = "cpu"
    dtype = torch.float32

# Load better VAE
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",
    torch_dtype=dtype
)

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype,
    vae=vae,
)
# Disable NSFW checker
pipe.safety_checker = None
pipe.feature_extractor = None
pipe.requires_safety_checker = False

if device == "cuda":
    pipe.enable_sequential_cpu_offload()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()


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
    num_inference_steps=5
).images[0]

image.save("outputs/test_1.png")
print("✔ output saved to outputs/test_1.png")
