# 🔬 ISIC Deri Lezyonu Görüntü İşleme Projesi

Bu projede, ISIC (International Skin Imaging Collaboration) deri lezyonu veri setinde kapsamlı görüntü işleme tekniklerinin uygulamasını içermektedir.Çalışma kapsamında, ISIC 2018 Skin Lesion Dataset kullanılarak, ISIC deri lezyonu görüntüleri üzerinde 
hem RGB (renkli) hem grayscale (gri tonlamalı) görüntü işleme tekniklerini uygulanmış, program çıktıları üzerinden adım adım tüm işlemler incelenmiştir.Paylaşılan sonuçlar üzerinden ve programın ürettiği ekran çıktıları üzerinden elde edilen sonuçlar yorumlanmıştır.Çalışma kapsamında, Python tabanlı bir geliştirme yapılmış ve çalışma github’a yüklenmiştir.
Çalışma esnasında veri setleri yüklenip analiz edildikten sonra, kanal sayılarını ve dosya boyutu dağılımlarının kontrolü yapılmış, rastgele seçilen görüntüler üzerinden RGB-Grayscale dönüşümleri gerçekleştirilmiş, seçilen görüntülerin minimum piksel değeri, maksimum piksel değeri, ortalama ve standart sapmaları incelenmiştir. Histogram analizi yapılarak, yorumlanmıştır. Sonrasında görüntü işleme ve iyileştirme teknikleri uygulanıp, sonuçları incelenmiştir. (Stretching, Equalization, Gamma). Ardından gürültü azaltma(Median, Gaussian) ve döndürme ve ayna çevirme denenmiştir.FFT kapsamında Fourier dönüşümü yapılmış, Unsharp Masking ve Bicubic Enterpolasyon ile proje tamamlanmıştır. 

## 📊 Proje Özeti

- **Veri Seti:** ISIC Deri Lezyonu (1000 görüntü, 2.26 GB)
- **Uygulanan Teknik Sayısı:** 15+
- **Oluşturulan Görselleştirme:** ~46 grafik
- **Programlama Dili:** Python 3.8+

## 🎯 Uygulanan Teknikler

### Bölüm 1: Veri Analizi
- Veri seti yükleme ve istatistiksel analiz
- Çözünürlük, dosya boyutu, kanal analizi

### Bölüm 2: Görselleştirme
- RGB vs Grayscale karşılaştırma
- Histogram analizi ve yorumlama

### Bölüm 3: Görüntü İyileştirme
- Kontrast germe (Contrast Stretching)
- Histogram eşitleme (Histogram Equalization)
- Gamma düzeltme

### Bölüm 4: Gürültü Azaltma
- Median blur (kenar koruyucu)
- Gaussian blur
- Karşılaştırmalı analiz

### Bölüm 5: Veri Augmentation
- Rastgele döndürme (0-10°)
- Yatay ayna çevirme
- 1000 → 22,000 görüntü potansiyeli

### Bölüm 6: Frekans Alanı
- Fast Fourier Transform (FFT)
- Alçak geçiren filtre
- RGB vs Grayscale FFT analizi

### Bölüm 7: Keskinleştirme
- Unsharp masking
- Bicubic enterpolasyon
- Enterpolasyon yöntemleri karşılaştırması

## 🚀 Kurulum
```bash
# Repository'yi klonlayın
git clone https://github.com/engindalgic86-cpu/isic-skin-lesion-analysis.git
cd isic-skin-lesion-analysis

# Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt

# Veri setini indirin
# Kaggle: https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic
# ISIC klasörüne yerleştirin


## 💻 Kullanım
```python
# Ana programı çalıştırın
python isic_project.py


## 📈 Sonuçlar

### Önemli Bulgular
- **En etkili yöntem:** Histogram eşitleme + Median blur
- **Optimal parametre:** Median blur kernel=5
- **Veri augmentation:** %2,100 artış potansiyeli
- **RGB vs Grayscale:** RGB renk bilgisi tanı için kritik

### Örnek Çıktılar

![Veri Analizi](results/01_veri_analizi.png)
![Histogram Eşitleme](results/05_histogram_esitleme_1.png)
![FFT Analizi](results/13_fft_spectrum_1.png)

## 📚 Teknolojiler

- **Python 3.8+**
- **NumPy** - Sayısal hesaplamalar
- **OpenCV** - Görüntü işleme
- **Matplotlib/Seaborn** - Görselleştirme
- **Pandas** - Veri analizi



## 📄 Lisans

MIT License

## 👤 Yazar

**[Mustafa Engin Dalgıç]**
- GitHub: [@engindalgic86-cpu ](https://github.com/engindalgic86-cpu )


