import os
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from diffusers.models import IPAdapter

# -------------------------------------------------
# Device detection (CUDA varsa otomatik kullan)
# -------------------------------------------------
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
    print("✔ CUDA detected – using NVIDIA GPU")
else:
    device = "cpu"
    dtype = torch.float32
    print("ℹ CUDA not available – using CPU")

os.makedirs("outputs", exist_ok=True)

# -------------------------------------------------
# Stable Diffusion pipeline
# -------------------------------------------------
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype
)

pipe = pipe.to(device)

# Memory optimizations
pipe.enable_attention_slicing()
if device == "cuda":
    pipe.enable_xformers_memory_efficient_attention()

# -------------------------------------------------
# IP-Adapter Face
# -------------------------------------------------
ip_adapter = IPAdapter(
    pipe,
    "h94/IP-Adapter",
    subfolder="models",
    weight_name="ip-adapter-face_sd15.bin",
    device=device
)

# -------------------------------------------------
# Reference image
# -------------------------------------------------
face_image = Image.open("refs/face.jpg").convert("RGB")

prompt = (
    "photo of a woman sitting at an office desk, "
    "drinking coffee, natural window light, "
    "realistic photography, shallow depth of field"
)

negative_prompt = (
    "different person, distorted face, deformed eyes, "
    "unrealistic, low quality"
)

# -------------------------------------------------
# Generation
# -------------------------------------------------
images = ip_adapter.generate(
    pil_image=face_image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_samples=1,
    num_inference_steps=30,
    guidance_scale=7.0,
    seed=42
)

images[0].save("outputs/ip_test_1.png")
print("✔ outputs/ip_test_1.png saved")
