
import os
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from transformers import CLIPVisionModelWithProjection
from insightface.app import FaceAnalysis

class ImageGenerator:
    def __init__(self, device="cuda", dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        self.pipe = None
        self.face_app = None
        self._load_models()

    def _load_models(self):
        print("⏳ Loading models...")
        
        # 1. Face Analysis (InsightFace)
        self.face_app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

        # 2. ControlNet
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            torch_dtype=self.dtype
        )

        # 3. Image Encoder
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            torch_dtype=self.dtype
        )

        # 4. Pipeline
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            image_encoder=image_encoder,
            torch_dtype=self.dtype,
            variant="fp16",
            use_safetensors=True
        )

        # 5. IP-Adapter
        self.pipe.load_ip_adapter(
            "h94/IP-Adapter", 
            subfolder="sdxl_models", 
            weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors"
        )
        
        self.pipe.to(self.device)
        print("✔ Models loaded successfully.")

    def process_face(self, image_pil):
        """
        Detects, crops face and returns (face_crop_pil, canny_edge_pil)
        """
        image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        faces = self.face_app.get(image_bgr)
        
        if len(faces) == 0:
            print("⚠ No face detected, using full image")
            return image_pil, self._make_canny(image_bgr)

        # Get largest face
        faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
        face = faces[0]
        bbox = face.bbox
        
        # Crop with context
        h, w, _ = image_bgr.shape
        x1, y1, x2, y2 = bbox
        margin_x = (x2 - x1) * 0.3
        margin_y = (y2 - y1) * 0.3
        
        x1 = max(0, int(x1 - margin_x))
        y1 = max(0, int(y1 - margin_y))
        x2 = min(w, int(x2 + margin_x))
        y2 = min(h, int(y2 + margin_y))
        
        face_crop = image_bgr[y1:y2, x1:x2]
        canny_map = self._make_canny(face_crop)
        
        face_crop_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        
        return face_crop_pil, canny_map

    def _make_canny(self, image_bgr):
        canny = cv2.Canny(image_bgr, 100, 200)
        canny = canny[:, :, None]
        canny = np.concatenate([canny, canny, canny], axis=2)
        return Image.fromarray(canny)

    def generate(self, 
                 prompt: str, 
                 negative_prompt: str, 
                 input_images: list[Image.Image], 
                 num_steps: int = 30, 
                 guidance_scale: float = 7.0,
                 ip_scale: float = 1.0,
                 control_scale: float = 0.5,
                 seed: int = 42):
        
        if not input_images:
            raise ValueError("At least one input image is required.")
        
        # Strategy:
        # Image 1: Main reference (Structure + Identity)
        # Image 2+: Identity mixing
        
        main_image = input_images[0]
        face_crop_main, canny_map = self.process_face(main_image)
        
        # Collect identity embeddings
        # We start with the main image's face
        identity_images = [face_crop_main]
        
        # Process other images just for identity (simple crop or full use)
        # For simplicity, we try to crop faces from them too, if fails use full.
        for img in input_images[1:]:
            face_crop, _ = self.process_face(img)
            identity_images.append(face_crop)
            
        print(f"🎨 Generating with {len(identity_images)} identity inputs...")
        
        self.pipe.set_ip_adapter_scale(ip_scale)
        
        generator = torch.Generator(self.device).manual_seed(seed)
        
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=canny_map, # ControlNet
            ip_adapter_image=identity_images, # List of images for mixing
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=control_scale,
            num_images_per_prompt=1,
            height=1024,
            width=1024,
            generator=generator
        ).images[0]
        
        return image
