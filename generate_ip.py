
import os
import sys
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

# -------------------------------------------------
# Configuration
# -------------------------------------------------
REF_IMAGE_PATH = "refs/face.jpg"
OUTPUT_DIR = "outputs"
OUTPUT_FILENAME = "ip_sdxl_test_1.png"

PROMPT = (
    "photo of a person sitting at an office desk, "
    "drinking coffee, natural window light, "
    "realistic photography, shallow depth of field, 8k, high quality"
)

NEGATIVE_PROMPT = (
    "different person, distorted face, deformed eyes, "
    "unrealistic, low quality, cartoon, anime, scalar, blurry"
)

# -------------------------------------------------
# Setup & Validation
# -------------------------------------------------
if not os.path.exists(REF_IMAGE_PATH):
    print(f"❌ Error: Reference image not found at '{REF_IMAGE_PATH}'")
    print("   Please place a face image in the 'refs' folder.")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------
# Device detection
# -------------------------------------------------
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
    print("✔ CUDA detected – using NVIDIA GPU (float16)")
else:
    device = "cpu"
    dtype = torch.float32
    print("ℹ CUDA not available – using CPU")

# -------------------------------------------------
# Pipeline Initialization (SDXL)
# -------------------------------------------------
print("Loading Stable Diffusion XL pipeline...")

# Load SDXL
# We use the official base model. 
# VAE: SDXL VAE is usually good, but we can fix fp16 issues if they arise. 
# For now, standard loading.
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=dtype,
    use_safetensors=True,
    variant="fp16"
)

# -------------------------------------------------
# IP-Adapter Setup for SDXL
# -------------------------------------------------
print("Loading IP-Adapter for SDXL...")

# For SDXL IP-Adapter Plus Face, we need the correct image encoder
# The model uses CLIP-ViT-H-14-laion2B-s32B-b79K
image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
# Usually Diffusers handles the image encoder if we load the ip-adapter, 
# but for best results/control we often load it. 
# However, let's try the simplified API first which is robust.

# Load IP Adapter
# repo: h94/IP-Adapter
# subfolder: sdxl_models
# filename: ip-adapter-plus-face_sdxl_vit-h.safetensors
pipe.load_ip_adapter(
    "h94/IP-Adapter", 
    subfolder="sdxl_models", 
    weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors",
    image_encoder_folder="models/image_encoder" # This might be needed if not auto-downloaded
)
# Note: if the image encoder isn't found, we might need to load it manually. 
# But let's trust the library to fetch the default for the adapter or fail with a clear message.
# "h94/IP-Adapter" repo has "models/image_encoder" which matches the vit-h usually.

# Scale: 0.0 to 1.0
pipe.set_ip_adapter_scale(0.7)

# Move to GPU
pipe.to(device)

# -------------------------------------------------
# Optimization
# -------------------------------------------------
# With 24GB VRAM, we don't need aggressive offloading for a single image.
# We can enable some optimizations for speed.
# pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True) # Optional, can be slow to compile first time

# -------------------------------------------------
# Generation
# -------------------------------------------------
print("Generating image (SDXL 1024x1024)...")
face_image = Image.open(REF_IMAGE_PATH).convert("RGB")

# Callback to save intermediate steps
intermediate_dir = os.path.join(OUTPUT_DIR, "intermediate_sdxl")
os.makedirs(intermediate_dir, exist_ok=True)

def save_intermediate_step(step, timestep, latents):
    # Save fewer intermedates for SDXL as it's heavier
    if step % 10 == 0: 
        print(f"  [Step {step}] Processing...")
        # Decoding SDXL latents mid-loop is expensive and complex due to VAE size.
        # Skipping visual preview for speed, just logging.

image = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    ip_adapter_image=face_image,
    num_inference_steps=30, # SDXL needs fewer steps usually
    guidance_scale=7.0,
    num_images_per_prompt=1,
    height=1024,
    width=1024,
    callback=save_intermediate_step,
    callback_steps=1,
    # generator=torch.Generator(device).manual_seed(42) 
).images[0]

output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
image.save(output_path)
print(f"✔ Output saved to {output_path}")
