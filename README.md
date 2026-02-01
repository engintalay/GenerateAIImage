# GenerateAIImage

Bu proje, Stable Diffusion XL (SDXL) kullanarak verilen bir referans fotoğraftan yüksek kaliteli (1024x1024) yeni görseller üretir. IP-Adapter (Face) teknolojisi ile yüz benzerliğini korur.

## Özellikler

- **SDXL 1.0 Kalitesi**: 1024x1024 çözünürlükte gerçekçi üretim.
- **Yüz Transferi**: `IP-Adapter Plus Face (ViT-H)` kullanarak referans yüzü korur.
- **Yüksek Performans**: RTX A5000 gibi yüksek VRAM'li kartlar için optimize edilmiştir (FP16).

## Kurulum

1. Proje dizinine gidin ve bir sanal ortam oluşturun:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   # venv\Scripts\activate  # Windows
   ```

2. Gerekli paketleri `requirements.txt` üzerinden yükleyin:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Kullanım

1. **Referans Görsel**: `refs` klasörü içine `face.jpg` adında bir referans fotoğraf koyun.
   ```
   refs/
   └── face.jpg
   ```

2. **Kodu Çalıştırın**:
   ```bash
   python generate_ip.py
   ```

3. **Çıktı**: Üretilen görsel `outputs/` klasörüne `ip_sdxl_test_1.png` olarak kaydedilecektir.

## Sistem Gereksinimleri

- **Python**: 3.8+
- **GPU**: NVIDIA GPU (En az 16GB VRAM önerilir, 24GB+ ile tam performans çalışır).
- **RAM**: 16GB+ Sistem RAM.

## Notlar

- İlk çalıştırmada SDXL modelleri ve IP-Adapter ağırlıkları indirilecektir (~10GB).
