# Uydu Görüntülerinden Su Kütlesi Tespiti (Water Body Segmentation)

Bu proje, uydu görüntülerinden su alanlarını tespit etmek için U-Net derin öğrenme modelini kullanır.

## 1. Kurulum (Setup)

1. Projeyi bilgisayarınıza indirin.
2. Gerekli kütüphaneleri kurun:
   ```bash
   pip install -r requirements.txt
   ```
   *Not: Eğer NVIDIA ekran kartınız varsa, PyTorch'un GPU versiyonunu kurmanız önerilir. (Bakınız: [pytorch.org](https://pytorch.org/get-started/locally/))*

## 2. Veri Hazırlığı (Data Preparation)
GitHub deposunda dosya boyutunu düşük tutmak için işlenmiş veri seti (`dataset_processed`) **bulunmamaktadır**. Aşağıdaki adımları takip ederek veriyi kendiniz hazırlamalısınız.

1. **Ham Veriyi İndirin:** GID (Gaofen Image Dataset) verilerini projenin ana dizinindeki `data/` klasörüne koyun. (Eğer `data` klasörü yoksa oluşturun).
   
   Klasör yapısı tam olarak şöyle görünmelidir:
   ```
   data/
   ├── GID-img-1/     (Orijinal görüntüler burada olmalı)
   ├── GID-img-2/     (Varsa diğer parça)
   ├── ...
   └── GID-label/     (Renkli etiket/maske dosyaları burada olmalı)
   ```

2. **Veriyi İşleyin:**
   Aşağıdaki komutu çalıştırarak eğitim için gerekli olan siyah-beyaz maskeleri ve klasör düzenini otomatik oluşturun:
   ```bash
   python prepare_dataset.py
   ```
   ⏳ *Bu işlem bilgisayar hızınıza göre 5-15 dakika sürebilir.*
   
   İşlem bittiğinde `dataset_processed/` adında yeni bir klasör oluşacak ve içinde `images` ile `masks` klasörleri yer alacaktır.

## 3. Eğitim (Training)
Modeli eğitmek için:
```bash
python train.py
```
*   Kod otomatik olarak veriyi %80 Eğitim, %20 Test olarak ayırır.
*   Eğitim bittiğinde (veya her iyileşmede) en iyi model `best_model.pth` olarak kaydedilecektir.
*   Eğer eğitim çok uzun sürerse `train.py` içindeki `EPOCHS` sayısını düşürebilirsiniz.

## 4. Tahmin (Prediction)
Modeli test etmek ve yeni görüntülerde su tespiti yapmak için:
1. `best_model.pth` dosyasının olduğundan emin olun.
2. `predict.py` dosyasını açıp `TEST_IMG_DIR` değişkenine test etmek istediğiniz resimlerin klasör yolunu yazın.
3. Kodu çalıştırın:
   ```bash
   python predict.py
   ```
4. Sonuçlar (siyah-beyaz maskeler) `predictions/` klasörüne kaydedilir.
