# GenerateAIImage

Bu proje, Stable Diffusion kullanarak verilen bir referans fotoğraftan yeni görseller üretir.

## Özellikler

- **Image-to-Image Üretim**: Referans bir yüz fotoğrafı kullanarak, belirtilen prompt'a uygun yeni görseller oluşturur.
- **Otomatik Cihaz Seçimi**: NVIDIA GPU (CUDA) varsa kullanır, yoksa otomatik olarak CPU'ya geçer.

## Kurulum

1. Proje dizinine gidin ve bir sanal ortam oluşturun:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   # venv\Scripts\activate  # Windows
   ```

2. Gerekli paketleri yükleyin:
   ```bash
   pip install --upgrade pip
   pip install torch torchvision torchaudio
   pip install diffusers transformers accelerate safetensors pillow opencv-python
   ```

## Kullanım

1. **Referans Görsel**: `refs` klasörü içine `face.jpg` adında bir referans fotoğraf koyun.
   ```
   refs/
   └── face.jpg
   ```

2. **Kodu Çalıştırın**:
   ```bash
   python generate.py
   ```

3. **Çıktı**: Üretilen görsel `outputs/` klasörüne `test_1.png` olarak kaydedilecektir.

## Gereksinimler

- Python 3.8+
- (Opsiyonel) NVIDIA GPU ve CUDA (daha hızlı üretim için)
