import os
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline

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
# pipe.enable_attention_slicing()
if device == "cuda":
    pipe.enable_xformers_memory_efficient_attention()

# -------------------------------------------------
# IP-Adapter Face Setup (Native Diffusers)
# -------------------------------------------------
# Not: Modeli ilk kez indirirken internet gerekebilir.
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-full-face_sd15.bin")

# Etkisini ayarlamak için (0.0 - 1.0 arası)
pipe.set_ip_adapter_scale(0.6)

# -------------------------------------------------
# Reference image
# -------------------------------------------------
# Eğer dosya yoksa hata vermemesi için kontrol ekleyebiliriz veya kullanıcıya bırakırız.
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
# native diffusers kullanımında ip_adapter_image parametresi kullanılır
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    ip_adapter_image=face_image,
    num_inference_steps=30,
    guidance_scale=7.0,
    num_images_per_prompt=1,
    generator=torch.Generator(device).manual_seed(42) if device == "cpu" else torch.Generator(device).manual_seed(42)
).images[0]

image.save("outputs/ip_test_1.png")
print("✔ outputs/ip_test_1.png saved")
