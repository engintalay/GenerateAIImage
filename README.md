python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

pip install \
  torch torchvision torchaudio \
  diffusers transformers accelerate \
  safetensors pillow opencv-python

