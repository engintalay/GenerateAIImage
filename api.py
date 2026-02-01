
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from typing import List
import io
import base64
from PIL import Image
from engine import ImageGenerator
from ui import create_ui
import gradio as gr

# Initialize App
app = FastAPI(title="GenAI Image Service", version="1.0")

# Initialize Engine (Global)
print("🚀 Initializing Generator Engine...")
generator = ImageGenerator()

# Mount Gradio UI
print("🖥 Mounting Gradio UI at /ui ...")
gradio_app = create_ui(generator)
app = gr.mount_gradio_app(app, gradio_app, path="/ui")

@app.get("/")
def read_root():
    return {"message": "Service is running. Go to /ui for interface or /docs for API."}

@app.post("/generate")
async def generate_api(
    prompt: str = Form(...),
    negative_prompt: str = Form("blurry, low quality"),
    num_steps: int = Form(30),
    guidance_scale: float = Form(7.0),
    ip_scale: float = Form(1.0),
    control_scale: float = Form(0.5),
    files: List[UploadFile] = File(...)
):
    """
    Generate image from prompt and uploaded images.
    Returns PNG image bytes.
    """
    try:
        # Load images
        input_images = []
        for file in files:
            content = await file.read()
            image = Image.open(io.BytesIO(content)).convert("RGB")
            input_images.append(image)
        
        # Generate
        output_image = generator.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            input_images=input_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            ip_scale=ip_scale,
            control_scale=control_scale,
            seed=42 # Or random
        )
        
        # Return as PNG
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        return Response(content=img_byte_arr, media_type="image/png")

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7002)
