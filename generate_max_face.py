import os
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from transformers import CLIPVisionModelWithProjection
from insightface.app import FaceAnalysis

REF_IMAGE_PATH = "refs/face.jpg"
OUTPUT_DIR = "outputs"

# Face-focused prompt
PROMPT = "close-up portrait photo of the same person, identical face, realistic, high quality"
NEGATIVE_PROMPT = "different person, wrong identity, changed face, multiple people, distorted"

device = "cuda"
dtype = torch.float16

# Face detection
app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

image_bgr = cv2.imread(REF_IMAGE_PATH)
faces = app.get(image_bgr)
face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)[0]

# Tight face crop
x1, y1, x2, y2 = face.bbox
h, w, _ = image_bgr.shape
margin = min((x2-x1), (y2-y1)) * 0.15
x1, y1 = max(0, int(x1-margin)), max(0, int(y1-margin))
x2, y2 = min(w, int(x2+margin)), min(h, int(y2+margin))

face_crop = cv2.cvtColor(image_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
face_pil = Image.fromarray(face_crop)

# Create multiple face variations for stronger identity
face_images = [face_pil, face_pil, face_pil]  # Triple the same face

# Canny for structure
canny = cv2.Canny(image_bgr[y1:y2, x1:x2], 50, 150)
canny = np.stack([canny]*3, axis=-1)
canny_pil = Image.fromarray(canny)

# Pipeline
controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=dtype)
image_encoder = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", torch_dtype=dtype)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    image_encoder=image_encoder,
    torch_dtype=dtype,
    variant="fp16"
)

pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors")
pipe.to(device)

# Maximum face preservation
pipe.set_ip_adapter_scale(2.2)

os.makedirs(OUTPUT_DIR, exist_ok=True)

image = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    image=canny_pil,
    ip_adapter_image=[face_images],  # Multiple same faces
    num_inference_steps=60,
    guidance_scale=4.0,  # Very low for maximum IP-Adapter influence
    controlnet_conditioning_scale=0.2,  # Minimal structure control
    height=1024,
    width=1024,
).images[0]

image.save(f"{OUTPUT_DIR}/max_face_preserve.png")
print("✔ Generated with maximum face preservation")
