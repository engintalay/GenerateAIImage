
import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from transformers import CLIPVisionModelWithProjection
from insightface.app import FaceAnalysis

# -------------------------------------------------
# Configuration
# -------------------------------------------------
REF_IMAGE_PATH = "refs/face.jpg"
OUTPUT_DIR = "outputs"
OUTPUT_FILENAME = "ip_canny_face_test_1.png"

PROMPT = (
    "professional headshot photo of a person, "
    "natural lighting, high quality, realistic, detailed face"
)

NEGATIVE_PROMPT = (
    "different person, wrong face, distorted face, deformed eyes, "
    "unrealistic, low quality, cartoon, anime, painting, drawing, "
    "multiple faces, blurry face"
)

# -------------------------------------------------
# Setup & Validation
# -------------------------------------------------
if not os.path.exists(REF_IMAGE_PATH):
    print(f"❌ Error: Reference image not found at '{REF_IMAGE_PATH}'")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float32
    print("✔ CUDA detected – using NVIDIA GPU (float32)")
else:
    print("❌ GPU required for this configuration.")
    sys.exit(1)

# -------------------------------------------------
# Face Detection
# -------------------------------------------------
print("Initializing Face Analysis...")
app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

image_bgr = cv2.imread(REF_IMAGE_PATH)
faces = app.get(image_bgr)

if len(faces) == 0:
    print("❌ No face detected!")
    sys.exit(1)

# Sort by size
faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
face_info = faces[0]
bbox = face_info.bbox
x1, y1, x2, y2 = bbox
h, w, c = image_bgr.shape

# Crop with smaller margin for tighter face focus
margin_x = (x2 - x1) * 0.2  # Reduced from 0.3 to focus more on face
margin_y = (y2 - y1) * 0.2
x1 = max(0, int(x1 - margin_x))
y1 = max(0, int(y1 - margin_y))
x2 = min(w, int(x2 + margin_x))
y2 = min(h, int(y2 + margin_y))

face_crop_bgr = image_bgr[y1:y2, x1:x2]
face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
face_image_pil = Image.fromarray(face_crop_rgb)

# -------------------------------------------------
# Prepare Canny Control Image
# -------------------------------------------------
# Canny edge detection on the cropped face
# This forces the generated face to have the exact same lines/structure
canny_image = cv2.Canny(face_crop_bgr, 100, 200)
canny_image = canny_image[:, :, None]
canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
canny_image_pil = Image.fromarray(canny_image)

# -------------------------------------------------
# Pipeline Setup (ControlNet + IP-Adapter)
# -------------------------------------------------
print("Loading Models...")

# 1. ControlNet Canny
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=dtype
)

# 2. image encoder
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    torch_dtype=dtype
)

# 3. Pipeline
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    image_encoder=image_encoder,
    torch_dtype=dtype,
    use_safetensors=True
)

# 4. IP-Adapter
pipe.load_ip_adapter(
    "h94/IP-Adapter", 
    subfolder="sdxl_models", 
    weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors"
)

# 5. Memory Optimizations for 4GB GPU
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
# pipe.to(device)  # Removed: offloading handles placement. Direct .to(device) would move everything to GPU at once.

# -------------------------------------------------
# Generation
# -------------------------------------------------
print("Generating image (ControlNet + IP-Adapter)...")
torch.cuda.empty_cache()

# Lower scale for stability and better generation
pipe.set_ip_adapter_scale(0.7) 

image = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    image=canny_image_pil, # ControlNet input
    ip_adapter_image=face_image_pil, # IP-Adapter input
    num_inference_steps=2,  # Adjusted for verification speed
    guidance_scale=5.0,      # Lower guidance for more IP-Adapter influence
    controlnet_conditioning_scale=0.3,  # Reduced from 0.5 for less structural control
    num_images_per_prompt=1,
    height=768,
    width=768,
).images[0]

output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
image.save(output_path)
print(f"✔ Output saved to {output_path}")
