# 🔬 ISIC Cilt Lezyonu Analizi - Sayısal Görüntü İşleme Projesi

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/engindalgic86-cpu/UU_DIP_ISIC_SKIN_LESION_ANALYSIS/blob/main/isic_analysis_colab.ipynb)

## Proje Sahibi: Mustafa Engin Dalgıç
## Öğrenci No: 254309502
## Üniversite: Üsküdar Üniversitesi - Bilgisayar Mühendisliği YL
## Email: engindalgic86@gmail.com
## Üsküdar Üniversitesi Sayısal Görüntü İşleme dersi kapsamında hazırlanmış ISIC cilt lezyonu analizi projesidir.

---

## 📚 Proje Raporu

📄 **[Proje Raporunu İnceleyin (PDF)](https://github.com/engindalgic86-cpu/UU_DIP_ISIC_SKIN_LESION_ANALYSIS/blob/main/docs/Sayisal%20G%C3%B6r%C3%BCnt%C3%BC%20%C4%B0%C5%9Fleme%20Proje.pdf)** - Detaylı analiz ve sonuçlar

---

## 🚀 2 FARKLI ÇALIŞTIRMA YÖNTEMİ

### 🌐 YÖNTEM 1: Google Colab (TEK TIKLA - ÖNERİLEN) ⭐

**Kurulum gerektirmez! Tarayıcıda direkt çalışır.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/engindalgic86-cpu/UU_DIP_ISIC_SKIN_LESION_ANALYSIS/blob/main/isic_analysis_colab.ipynb)

**Nasıl Kullanılır:**
1. Yukarıdaki **"Open in Colab"** butonuna tıklayın
2. Notebook açılınca: `Runtime > Run all` (veya `Ctrl+F9`)
3. 10-15 dakika bekleyin
4. Grafikler otomatik görünecek! ✅

**Avantajları:**
- ✅ Hiçbir şey kurmanıza gerek yok
- ✅ Tarayıcıda çalışır
- ✅ Ücretsiz GPU var
- ✅ Tek tık ile başlar

---

### 💻 YÖNTEM 2: Lokal Bilgisayarınızda

```bash
# 1. Repoyu klonlayın
git clone https://github.com/engindalgic86-cpu/UU_DIP_ISIC_SKIN_LESION_ANALYSIS.git
cd UU_DIP_ISIC_SKIN_LESION_ANALYSIS

# 2. Gereksinimleri yükleyin
pip install -r requirements.txt

# 3. Kurulumu test edin (opsiyonel)
python test_kurulum.py

# 4. Programı çalıştırın
python isic_project.py
```

**Gereksinimler:**
- Python 3.8+
- 8 GB RAM
- 10-15 dakika işlem süresi

---

## 📊 Program Ne Yapar?

Bu program cilt lezyonu görüntüleri üzerinde **7 ana bölümde** analiz yapar:

### 🎯 Uygulanan Teknikler

1. ✅ **Veri Analizi** - Görüntü özellikleri ve dağılımları
2. ✅ **Görselleştirme** - RGB vs Grayscale karşılaştırma  
3. ✅ **Histogram Analizi** - Renk dağılımları
4. ✅ **Kontrast İyileştirme** - CLAHE, Gamma Correction
5. ✅ **Gürültü Azaltma** - Median, Gaussian filtreleme
6. ✅ **Geometrik Dönüşümler** - Döndürme, ayna çevirme
7. ✅ **Frekans Filtreleme** - FFT, keskinleştirme

**Çıktı:** Toplamda **~45 grafik** `.png` formatında oluşturulur.

---

## 📁 Proje Yapısı

```
UU_DIP_ISIC_SKIN_LESION_ANALYSIS/
│
├── 📓 isic_analysis_colab.ipynb  # Google Colab notebook
├── 🐍 isic_project.py            # Ana Python programı
├── 🧪 test_kurulum.py            # Kurulum test scripti
├── 📄 requirements.txt           # Python gereksinimleri
├── 📖 README.md                  # Bu dosya
│
├── 📁 docs/                      # Dökümanlar
│   └── 📄 Sayisal Görüntü İşleme Proje.pdf  # Proje raporu
│
├── 📁 ISIC/                      # Veri seti
│   ├── melanoma/
│   ├── nevus/
│   └── ...
│
└── 📊 results/                   # Program çıktıları
    ├── 01_veri_analizi.png
    ├── 02_rgb_vs_grayscale.png
    └── ... (~45 grafik)
```

---

## 🔬 Görüntü İşleme Yöntemleri

### Kontrast İyileştirme
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization)
- **Gamma Correction**
- **Histogram Equalization**

### Filtreleme
- **Median Filtering** - Gürültü azaltma
- **Gaussian Filtering** - Yumuşatma
- **FFT** (Fast Fourier Transform) - Frekans alanı

### İleri Teknikler
- **Unsharp Masking** - Keskinleştirme
- **Bicubic Interpolation** - Görüntü büyütme
- **Low-pass / High-pass Filters**

---

## 📊 Veri Seti

- **Kaynak**: [ISIC Archive](https://www.isic-archive.com/)
- **Sınıf Sayısı**: 9 farklı cilt kanseri türü
- **Format**: RGB görüntüler

### Cilt Kanseri Türleri

- Melanoma
- Nevus
- Basal cell carcinoma
- Actinic keratosis
- Dermatofibroma
- Pigmented benign keratosis
- Seborrheic keratosis
- Squamous cell carcinoma
- Vascular lesion

---

## 📄 Dökümanlar

- 📊 **[Proje Raporu (PDF)](https://github.com/engindalgic86-cpu/UU_DIP_ISIC_SKIN_LESION_ANALYSIS/blob/main/docs/Sayisal%20G%C3%B6r%C3%BCnt%C3%BC%20%C4%B0%C5%9Fleme%20Proje.pdf)** - Detaylı analiz ve bulgular
- 📓 **[Google Colab Notebook](https://colab.research.google.com/github/engindalgic86-cpu/UU_DIP_ISIC_SKIN_LESION_ANALYSIS/blob/main/isic_analysis_colab.ipynb)** - İnteraktif çalışma ortamı

---

## 🆘 Sorun Giderme

### Colab'da Sorun Yaşıyorsanız

1. Sayfayı yenileyin (F5)
2. `Runtime > Restart runtime`
3. Tekrar `Run all`

### Lokal Kurulumda Sorun Yaşıyorsanız

**"ISIC klasörü bulunamadı"**
```bash
git clone https://github.com/engindalgic86-cpu/UU_DIP_ISIC_SKIN_LESION_ANALYSIS.git
cd UU_DIP_ISIC_SKIN_LESION_ANALYSIS
ls ISIC/  # Kontrol edin
```

**"ModuleNotFoundError"**
```bash
pip install -r requirements.txt
```

**Bellek Hatası**
- Daha az RAM için koddaki `max_size=800` → `max_size=400` yapın

---

## 👨‍🎓 Proje Sahibi

**Mustafa Engin Dalgıç**
- 🎓 Üsküdar Üniversitesi - Bilgisayar Mühendisliği Yüksek Lisans
- 📧 engindalgic86@gmail.com
- 🆔 Öğrenci No: 254309502

---

## 📚 Kaynaklar

- [ISIC Archive](https://www.isic-archive.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Kaggle ISIC Dataset](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic)

---

## 📄 Lisans

MIT License - Detaylar için `LICENSE` dosyasına bakınız.

---

## 💬 Geri Bildirim

Sorularınız veya önerileriniz için:
- 📧 Email: engindalgic86@gmail.com
- 🐛 GitHub Issues: [Sorun bildirin](https://github.com/engindalgic86-cpu/UU_DIP_ISIC_SKIN_LESION_ANALYSIS/issues)

---

## 🌟 Hızlı Başlangıç

### Colab İçin (Önerilen):
1. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/engindalgic86-cpu/UU_DIP_ISIC_SKIN_LESION_ANALYSIS/blob/main/isic_analysis_colab.ipynb) ← Buraya tıkla
2. `Runtime > Run all`
3. Bitti! ✅

### Lokal İçin:
```bash
git clone https://github.com/engindalgic86-cpu/UU_DIP_ISIC_SKIN_LESION_ANALYSIS.git
cd UU_DIP_ISIC_SKIN_LESION_ANALYSIS
pip install -r requirements.txt
python isic_project.py
```

---

## 📸 Örnek Çıktılar

Programın ürettiği grafik örnekleri:

- 🎨 **RGB vs Grayscale** karşılaştırmaları
- 📊 **Histogram** analizleri
- ✨ **CLAHE** kontrast iyileştirme
- 🔇 **Gürültü azaltma** filtreleri
- 🔄 **Geometrik dönüşümler**
- 📐 **FFT frekans analizleri**

Tüm grafikleri görmek için programı çalıştırın!

---

⭐ **Projeyi beğendiyseniz yıldız vermeyi unutmayın!** ⭐

---

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-4.x-green.svg" alt="OpenCV">
  <img src="https://img.shields.io/badge/Status-Complete-success.svg" alt="Status">
  <img src="https://img.shields.io/github/stars/engindalgic86-cpu/UU_DIP_ISIC_SKIN_LESION_ANALYSIS?style=social" alt="Stars">
</div>

---

**Son Güncelleme:** Kasım 2024
