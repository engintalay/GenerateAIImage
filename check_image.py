
from PIL import Image
import numpy as np
import os
import sys

output_path = "outputs/ip_test_1.png"

if not os.path.exists(output_path):
    print(f"File not found: {output_path}")
    sys.exit(1)

img = Image.open(output_path).convert("RGB")
data = np.array(img)

# Check if image is all black
if np.all(data == 0):
    print("FAILURE: Image is all black.")
    sys.exit(1)

# Check if image is mostly black (mean pixel value < 5 for example)
mean_val = np.mean(data)
print(f"Image mean pixel value: {mean_val:.2f}")

if mean_val < 1.0:
    print("FAILURE: Image is almost completely black.")
    sys.exit(1)

print("SUCCESS: Image has content.")
