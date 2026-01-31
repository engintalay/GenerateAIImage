
import os
import sys
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, AutoencoderKL

# -------------------------------------------------
# Configuration
# -------------------------------------------------
REF_IMAGE_PATH = "refs/face.jpg"
OUTPUT_DIR = "outputs"
OUTPUT_FILENAME = "ip_test_1.png"

PROMPT = (
    "photo of a person sitting at an office desk, "
    "drinking coffee, natural window light, "
    "realistic photography, shallow depth of field"
)

NEGATIVE_PROMPT = (
    "different person, distorted face, deformed eyes, "
    "unrealistic, low quality"
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
    print("✔ CUDA detected – using NVIDIA GPU")
else:
    device = "cpu"
    dtype = torch.float32
    print("ℹ CUDA not available – using CPU")

# -------------------------------------------------
# Pipeline Initialization
# -------------------------------------------------
print("Loading Stable Diffusion pipeline...")

# Load better VAE to fix black image/NaN issues
vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",
    torch_dtype=dtype
)

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype,
    vae=vae,
)
# Disable NSFW checker
pipe.safety_checker = None
pipe.feature_extractor = None

# pipe = pipe.to(device) # CPU offload handles device placement


# Fix for "black image" issue:
# explicit cast removed to test if sd-vae-ft-mse works in float16
# if device == "cuda":
#    pipe.vae = pipe.vae.to(dtype=torch.float32)


# -------------------------------------------------
# IP-Adapter Setup
# -------------------------------------------------
print("Loading IP-Adapter...")
# Using 'ip-adapter-plus-face_sd15.bin' for better detail (requires internet to download first time)
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus-face_sd15.bin")

# Scale: 0.0 to 1.0 (Higher means more resemblance to the reference image)
pipe.set_ip_adapter_scale(0.8)

if device == "cuda":
    # Enable CPU offload for 4GB VRAM - Must be called AFTER loading all components (like IP-Adapter)
    pipe.enable_model_cpu_offload()

# -------------------------------------------------
# Generation
# -------------------------------------------------
print("Generating image...")
face_image = Image.open(REF_IMAGE_PATH).convert("RGB")

image = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    ip_adapter_image=face_image,
    num_inference_steps=50,
    guidance_scale=7.0,
    num_images_per_prompt=1,
    generator=torch.Generator(device).manual_seed(42)
).images[0]

output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
image.save(output_path)
print(f"✔ Output saved to {output_path}")
