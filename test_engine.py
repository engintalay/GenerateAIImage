
from engine import ImageGenerator
from PIL import Image

def test():
    print("Initializing Engine...")
    gen = ImageGenerator()
    
    print("Loading Ref Image...")
    # Load face.jpg
    ref_img = Image.open("refs/face.jpg").convert("RGB")
    
    print("Generating...")
    output = gen.generate(
        prompt="photo of a person, masterpiece, 8k",
        negative_prompt="blurry, low quality",
        input_images=[ref_img],
        num_steps=20
    )
    
    output.save("outputs/engine_test.png")
    print("Done! Saved to outputs/engine_test.png")

if __name__ == "__main__":
    test()
