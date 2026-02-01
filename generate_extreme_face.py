import os
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from transformers import CLIPVisionModelWithProjection
from insightface.app import FaceAnalysis

REF_IMAGE_PATH = "refs/face.jpg"
OUTPUT_DIR = "outputs"

device = "cuda"
dtype = torch.float16

# Face analysis
app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Load and process reference
ref_img = cv2.imread(REF_IMAGE_PATH)
faces = app.get(ref_img)
face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)[0]

# Extract face embedding
face_embedding = face.embedding  # 512-dim vector

# Very tight face crop - only the face
x1, y1, x2, y2 = face.bbox
face_only = ref_img[int(y1):int(y2), int(x1):int(x2)]
face_pil = Image.fromarray(cv2.cvtColor(face_only, cv2.COLOR_BGR2RGB))

# Pipeline with custom image encoder
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", torch_dtype=dtype
)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    image_encoder=image_encoder,
    torch_dtype=dtype,
    variant="fp16"
)

pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors")
pipe.to(device)

# Extreme settings for face preservation
pipe.set_ip_adapter_scale(3.0)  # Maximum possible

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generate with very specific prompt
for i in range(3):
    seed = 1000 + i
    generator = torch.Generator(device).manual_seed(seed)
    
    image = pipe(
        prompt=f"exact same person, identical facial features, same face, professional photo",
        negative_prompt="different person, changed face, wrong identity, multiple faces",
        ip_adapter_image=face_pil,
        num_inference_steps=80,  # More steps
        guidance_scale=3.0,      # Very low guidance
        generator=generator,
        height=1024,
        width=1024,
    ).images[0]
    
    image.save(f"{OUTPUT_DIR}/extreme_face_{i+1}.png")
    print(f"✔ Generated extreme_face_{i+1}.png")

print("Face embedding shape:", face_embedding.shape)
print("Try the generated images - one should be closer to your reference")
