
import gradio as gr
from PIL import Image
import numpy as np
import random

def create_ui(generator):
    """
    Creates a Gradio Interface using the provided generator instance.
    """
    
    def generate_wrapper(prompt, negative_prompt, files, steps, scale, guidance, control_scale, seed):
        if not files:
            raise gr.Error("Please upload at least one image.")
        
        # Convert gallery/file inputs to PIL
        images = []
        for file_path in files:
            images.append(Image.open(file_path).convert("RGB"))
            
        for output, face_crops in generator.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            input_images=images,
            num_steps=steps,
            guidance_scale=guidance,
            ip_scale=scale,
            control_scale=control_scale,
            seed=seed,
            return_intermediates=True
        ):
            yield output, face_crops

    with gr.Blocks(title="GenAI Image Service") as demo:
        gr.Markdown("# 🎨 AI Image Generator (SDXL + ControlNet + IP-Adapter)")
        
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="Prompt", value="photo of a person, masterpiece, 8k, high quality")
                neg_prompt = gr.Textbox(label="Negative Prompt", value="blurry, low quality, distorted")
                
                # File uploader
                # Using 'filepath' type
                input_files = gr.File(label="Input Images (First is Structure+Identity, others Identity)", file_count="multiple", type="filepath")
                
                with gr.Accordion("Advanced Settings", open=True):
                    steps = gr.Slider(label="Inference Steps", minimum=10, maximum=100, step=1, value=30)
                    guidance = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=20.0, step=0.5, value=7.5)
                    scale = gr.Slider(label="Identity Strength (IP-Adapter)", minimum=0.0, maximum=2.0, step=0.1, value=0.7)
                    control_scale = gr.Slider(label="Structure Strength (ControlNet)", minimum=0.0, maximum=2.0, step=0.1, value=0.4)
                    
                    with gr.Row():
                        seed_num = gr.Number(label="Seed", value=42, precision=0)
                        randomize = gr.Checkbox(label="Randomize Seed", value=True)
                
                btn = gr.Button("Generate", variant="primary")
            
            with gr.Column(scale=1):
                output_image = gr.Image(label="Generated Image")
                face_gallery = gr.Gallery(label="Detected Faces (Used for Identity)", columns=3, height="auto")
        
        def ui_fn(p, np, f, s, sc, g, cs, sn, r):
            if r:
                sn = random.randint(0, 2**32 - 1)
            for out in generate_wrapper(p, np, f, s, sc, g, cs, sn):
                yield out

        btn.click(
            ui_fn,
            inputs=[prompt, neg_prompt, input_files, steps, scale, guidance, control_scale, seed_num, randomize],
            outputs=[output_image, face_gallery]
        )
    
    return demo

if __name__ == "__main__":
    from engine import ImageGenerator
    print("🚀 Initializing Standalone UI with Share Mode...")
    generator = ImageGenerator()
    demo = create_ui(generator)
    print("🌍 Launching Gradio with Public Link...")
    demo.launch(share=True, server_port=7002)
