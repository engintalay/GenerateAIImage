import os
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from transformers import CLIPVisionModelWithProjection
from insightface.app import FaceAnalysis

# Configuration
REF_IMAGE_PATH = "refs/face.jpg"
OUTPUT_DIR = "outputs"
OUTPUT_FILENAME = "face_preserve_test.png"

PROMPT = "photo of a person, professional headshot, natural lighting, high quality, realistic"
NEGATIVE_PROMPT = "different person, distorted face, deformed, low quality, cartoon, anime"

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

# Face detection
app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

image_bgr = cv2.imread(REF_IMAGE_PATH)
faces = app.get(image_bgr)

if len(faces) == 0:
    print("❌ No face detected!")
    exit(1)

# Get largest face with minimal cropping
face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)[0]
x1, y1, x2, y2 = face.bbox

# Minimal margin for maximum face detail
h, w, _ = image_bgr.shape
margin = min((x2-x1), (y2-y1)) * 0.1  # Very small margin
x1 = max(0, int(x1 - margin))
y1 = max(0, int(y1 - margin))
x2 = min(w, int(x2 + margin))
y2 = min(h, int(y2 + margin))

face_crop = cv2.cvtColor(image_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
face_image = Image.fromarray(face_crop)

# Pipeline setup - IP-Adapter only (no ControlNet for maximum face preservation)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    torch_dtype=dtype
)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    image_encoder=image_encoder,
    torch_dtype=dtype,
    variant="fp16"
)

pipe.load_ip_adapter(
    "h94/IP-Adapter", 
    subfolder="sdxl_models", 
    weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors"
)

pipe.to(device)

# Maximum face preservation settings
pipe.set_ip_adapter_scale(2.0)  # Very high for maximum similarity

# Generate multiple images with different seeds
os.makedirs(OUTPUT_DIR, exist_ok=True)

for i in range(3):
    seed = 42 + i
    generator = torch.Generator(device).manual_seed(seed)
    
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        ip_adapter_image=face_image,
        num_inference_steps=50,  # More steps for better quality
        guidance_scale=5.0,      # Lower guidance for more face influence
        generator=generator,
        height=1024,
        width=1024,
    ).images[0]
    
    output_path = os.path.join(OUTPUT_DIR, f"face_preserve_{i+1}.png")
    image.save(output_path)
    print(f"✔ Generated: {output_path}")
