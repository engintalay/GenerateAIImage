
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
    # GTX 16xx series has poor float16 support causing NaNs (black images).
    # We force float32 to guarantee numerical stability.
    dtype = torch.float32  
    print("✔ CUDA detected – using NVIDIA GPU (forcing float32)")
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
# Disable NSFW checker explicitly
pipe.safety_checker = None
pipe.feature_extractor = None
pipe.requires_safety_checker = False

# pipe = pipe.to(device) # CPU offload handles device placement

# -------------------------------------------------
# IP-Adapter Setup
# -------------------------------------------------
print("Loading IP-Adapter...")
# Using 'ip-adapter-plus-face_sd15.bin' for better detail (requires internet to download first time)
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus-face_sd15.bin")

# Scale: 0.0 to 1.0 (Higher means more resemblance to the reference image)
pipe.set_ip_adapter_scale(0.8)

if device == "cuda":
    # Enable Sequential CPU offload for 4GB VRAM - Stronger than model_cpu_offload
    # Required because float32 takes 2x memory
    pipe.enable_sequential_cpu_offload()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()


# -------------------------------------------------
# Generation
# -------------------------------------------------
# -------------------------------------------------
# Generation
# -------------------------------------------------
print("Generating image...")
face_image = Image.open(REF_IMAGE_PATH).convert("RGB")

# Callback to save intermediate steps
intermediate_dir = os.path.join(OUTPUT_DIR, "intermediate")
os.makedirs(intermediate_dir, exist_ok=True)

def save_intermediate_step(step, timestep, latents):
    # Save every 5 steps
    if step % 5 == 0:
        with torch.no_grad():
            # Latents need to be scaled before decoding
            latents_scaled = latents / vae.config.scaling_factor
            
            # Use the VAE to decode
            # Note: With cpu_offload, we rely on hooks or manual movement. 
            # Trying to use vae as is (accelerate should handle hooks if accessed via pipe.vae)
            image = pipe.vae.decode(latents_scaled, return_dict=False)[0]
            
            # Post-process
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = (image * 255).round().astype("uint8")
            image = Image.fromarray(image[0])
            
            image_path = os.path.join(intermediate_dir, f"step_{step:02d}.png")
            image.save(image_path)
            print(f"  [Step {step}] Saved intermediate image: {image_path}")

image = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    ip_adapter_image=face_image,
    num_inference_steps=50,
    guidance_scale=7.0,
    num_images_per_prompt=1,
    height=768,
    width=512,
    callback=save_intermediate_step,
    callback_steps=1,
    generator=torch.Generator(device).manual_seed(42)
).images[0]

output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
image.save(output_path)
print(f"✔ Output saved to {output_path}")

