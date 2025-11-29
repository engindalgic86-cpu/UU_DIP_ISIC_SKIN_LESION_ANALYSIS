#  ISIC Cilt Lezyonu Analizi - Sayısal Görüntü İşleme Projesi

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/engindalgic86-cpu/UU_DIP_ISIC_SKIN_LESION_ANALYSIS/blob/main/isic_analysis_colab.ipynb)

- Proje Sahibi: Mustafa Engin Dalgıç
- Öğrenci No: 254309502
- Üniversite: Üsküdar Üniversitesi - Bilgisayar Mühendisliği YL
- Email: engindalgic86@gmail.com
- Üsküdar Üniversitesi Sayısal Görüntü İşleme dersi kapsamında hazırlanmış ISIC cilt lezyonu analizi projesidir.

---

##  Proje Raporu

📄 **[Proje Raporunu İnceleyin (PDF)](https://github.com/engindalgic86-cpu/UU_DIP_ISIC_SKIN_LESION_ANALYSIS/blob/main/docs/Sayisal%20G%C3%B6r%C3%BCnt%C3%BC%20%C4%B0%C5%9Fleme%20Proje.pdf)** - Detaylı analiz ve gözlemlerim

---

## ÇALIŞTIRMA YÖNTEMİ 

### Google Colab 

**Kurulum gerektirmeden,tarayıcıda direkt çalıştırılabilir.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/engindalgic86-cpu/UU_DIP_ISIC_SKIN_LESION_ANALYSIS/blob/main/isic_analysis_colab.ipynb)

---

###  Lokal Bilgisayarınızda

```bash
# 1. Repoyu klonlayın
git clone https://github.com/engindalgic86-cpu/UU_DIP_ISIC_SKIN_LESION_ANALYSIS.git
cd UU_DIP_ISIC_SKIN_LESION_ANALYSIS

# 2. Gereksinimleri yükleyin
pip install -r requirements.txt

# 3. Programı çalıştırın
python isic_project.py
```

**Gereksinimler:**
- Python 3.8+
- 8 GB RAM
- 1-2 dakika işlem süresi ( ISIC klasörü demo resimler için)

---

## Proje İçeriği

Bu projede, ISIC (International Skin Imaging Collaboration) deri lezyonu veri setinde kapsamlı görüntü işleme tekniklerinin uygulamasını içermektedir.Çalışma kapsamında, ISIC 2018 Skin Lesion Dataset kullanılarak, ISIC deri lezyonu görüntüleri üzerinde hem RGB (renkli) hem grayscale (gri tonlamalı) görüntü işleme tekniklerini uygulanmış, program çıktıları üzerinden adım adım tüm işlemler incelenmiştir.Paylaşılan sonuçlar üzerinden ve programın ürettiği ekran çıktıları üzerinden elde edilen sonuçlar yorumlanmıştır.Çalışma kapsamında, Python tabanlı bir geliştirme yapılmış ve çalışma github’a yüklenmiştir. Çalışma esnasında veri setleri yüklenip analiz edildikten sonra, kanal sayılarını ve dosya boyutu dağılımlarının kontrolü yapılmış, rastgele seçilen görüntüler üzerinden RGB-Grayscale dönüşümleri gerçekleştirilmiş, seçilen görüntülerin minimum piksel değeri, maksimum piksel değeri, ortalama ve standart sapmaları incelenmiştir. Histogram analizi yapılarak, yorumlanmıştır. Sonrasında görüntü işleme ve iyileştirme teknikleri uygulanıp, sonuçları incelenmiştir. (Stretching, Equalization, Gamma). Ardından gürültü azaltma(Median, Gaussian) ve döndürme ve ayna çevirme denenmiştir.FFT kapsamında Fourier dönüşümü yapılmış, Unsharp Masking ve Bicubic Enterpolasyon ile proje tamamlanmıştır.

###  Uygulanan Teknikler

1. ✅ **Veri Analizi** - Görüntü özellikleri ve dağılımları
2. ✅ **Görselleştirme** - RGB vs Grayscale karşılaştırma  
3. ✅ **Histogram Analizi** - Renk dağılımları
4. ✅ **Kontrast İyileştirme** -  Gamma Correction
5. ✅ **Gürültü Azaltma** - Median, Gaussian filtreleme
6. ✅ **Geometrik Dönüşümler** - Döndürme, ayna çevirme
7. ✅ **Frekans Filtreleme** - FFT, keskinleştirme

**Çıktı:** Toplamda **~45 grafik** `.png` formatında oluşturulur.

---

##  Proje Yapısı

```
UU_DIP_ISIC_SKIN_LESION_ANALYSIS/
│
├── 📓 isic_analysis_colab.ipynb  # Google Colab notebook
├── 🐍 isic_project.py            # Ana Python programı
├── 📄 requirements.txt           # Python gereksinimleri
├── 📖 README.md                  # Bu dosya
│
├── 📁 docs/                      # Dökümanlar
│   └── 📄 Sayisal Görüntü İşleme Proje.pdf  # Proje raporu
│
├── 📁 ISIC/                      # Veri seti
│   ├── 
│   ├── 
│   └── ...
│
└── 📊 results/                   # Program çıktıları
    ├── 01_veri_analizi.png
    ├── 02_rgb_vs_grayscale.png
    └── ... (~45 grafik)
```

---

##  Görüntü İşleme Yöntemleri

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

##  Veri Seti

- **Kaynak**: https://challenge.isic-archive.com/data/#2018 , https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic
- **Sınıf Sayısı**: 9 farklı cilt kanseri türü
- **Format**: RGB görüntüler
- **Demo**: ISIC klasöründe github üzerinde demo veri seti bulunmaktadır.

---

##  Dökümanlar

-  **[Proje Raporu (PDF)](https://github.com/engindalgic86-cpu/UU_DIP_ISIC_SKIN_LESION_ANALYSIS/blob/main/docs/Sayisal%20G%C3%B6r%C3%BCnt%C3%BC%20%C4%B0%C5%9Fleme%20Proje.pdf)** - Detaylı analiz ve bulgular
-  **[Google Colab Notebook](https://colab.research.google.com/github/engindalgic86-cpu/UU_DIP_ISIC_SKIN_LESION_ANALYSIS/blob/main/isic_analysis_colab.ipynb)** - İnteraktif çalışma ortamı

---


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

##  Proje Sahibi

**Mustafa Engin Dalgıç**
-  Üsküdar Üniversitesi - Bilgisayar Mühendisliği Yüksek Lisans
-  engindalgic86@gmail.com
-  Öğrenci No: 254309502

---

##  Kaynaklar

- [ISIC Archive](https://www.isic-archive.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Kaggle ISIC Dataset](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic)

---

##  Lisans

MIT License - Detaylar için `LICENSE` dosyasına bakınız.

---

##  Geri Bildirim

Sorularınız veya önerileriniz için:
-  Email: engindalgic86@gmail.com
-  GitHub Issues: [Sorun bildirin](https://github.com/engindalgic86-cpu/UU_DIP_ISIC_SKIN_LESION_ANALYSIS/issues)

---

##  Hızlı Başlangıç

### Colab İçin (Önerilen):
1. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/engindalgic86-cpu/UU_DIP_ISIC_SKIN_LESION_ANALYSIS/blob/main/isic_analysis_colab.ipynb) ← Buraya tıkla
2. `Runtime > Run all`


### Lokal İçin:
```bash
git clone https://github.com/engindalgic86-cpu/UU_DIP_ISIC_SKIN_LESION_ANALYSIS.git
cd UU_DIP_ISIC_SKIN_LESION_ANALYSIS
pip install -r requirements.txt
python isic_project.py
```

---

##  Örnek Çıktılar

Programın ürettiği grafik örnekleri:

-  **RGB vs Grayscale** karşılaştırmaları
-  **Histogram** analizleri
-  **CLAHE** kontrast iyileştirme
-  **Gürültü azaltma** filtreleri
-  **Geometrik dönüşümler**
-  **FFT frekans analizleri**

Tüm grafikleri görmek için programı çalıştırabilirsiniz.

---



---

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-4.x-green.svg" alt="OpenCV">
  <img src="https://img.shields.io/badge/Status-Complete-success.svg" alt="Status">
  <img src="https://img.shields.io/github/stars/engindalgic86-cpu/UU_DIP_ISIC_SKIN_LESION_ANALYSIS?style=social" alt="Stars">
</div>

---

**Son Güncelleme:** 29 Kasım 2025
