import os
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLInpaintPipeline
from transformers import CLIPVisionModelWithProjection
from insightface.app import FaceAnalysis
import insightface

REF_IMAGE_PATH = "refs/face.jpg"
OUTPUT_DIR = "outputs"

# Setup
device = "cuda"
dtype = torch.float16

# Face analysis
app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Face swapper
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)

# Load reference face
ref_img = cv2.imread(REF_IMAGE_PATH)
ref_faces = app.get(ref_img)
if not ref_faces:
    print("❌ No face in reference image")
    exit(1)
ref_face = ref_faces[0]

# Generate base image with SDXL
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=dtype,
    variant="fp16"
)
pipe.to(device)

# Create base portrait
base_image = pipe(
    prompt="professional headshot photo of a person, neutral expression, studio lighting",
    negative_prompt="distorted face, low quality",
    height=1024,
    width=1024,
    num_inference_steps=30
).images[0]

# Convert to cv2 format
base_cv2 = cv2.cvtColor(np.array(base_image), cv2.COLOR_RGB2BGR)

# Detect face in generated image
base_faces = app.get(base_cv2)
if not base_faces:
    print("❌ No face in generated image")
    exit(1)

# Swap face
result = swapper.get(base_cv2, base_faces[0], ref_face, paste_back=True)

# Save result
os.makedirs(OUTPUT_DIR, exist_ok=True)
result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
result_pil.save(f"{OUTPUT_DIR}/faceswap_result.png")
print("✔ Face swap completed")
