# Classifying-the-Representation-of-Numbers

## PCA ve Makine Öğrenmesi ile Rakam Tanıma

Proje Açıklaması

Bu proje, sklearn.datasets içerisindeki digits veri setini kullanarak el yazısı rakamların sınıflandırılmasını hedefler. Modelleme sürecinde PCA (Principal Component Analysis) ile boyut indirgeme işlemi uygulanarak model performansı artırılmıştır. Destek Vektör Makineleri (SVM), Rastgele Orman (Random Forest) ve K-En Yakın Komşu (KNN) algoritmaları Grid Search kullanılarak optimize edilmiştir. Son olarak, Voting Classifier ile en iyi modeller birleştirilerek nihai bir sınıflandırıcı oluşturulmuştur.

## ⚙️ Kullanılan Teknolojiler ve Kütüphaneler

- 🐍 **Python**
- 📦 **NumPy** – Matematiksel işlemler
- 📊 **Matplotlib** – Görselleştirme
- 📚 **Scikit-learn (sklearn)** – Veri seti, modelleme, PCA, GridSearch, sınıflandırma algoritmaları

---

## 📌 Proje Adımları

### 1️⃣ Veri Yükleme ve Ön İşleme
- `load_digits()` fonksiyonu ile veri seti yüklendi.
- `train_test_split` ile eğitim ve test verileri ayrıldı.
- `StandardScaler` ile veriler normalize edildi.
- PCA uygulanarak boyut indirgeme sağlandı.
- PCA ile veri **2 boyutlu görselleştirildi**.

### 2️⃣ Model Eğitimi ve Hiperparametre Optimizasyonu
- **SVM**, **Random Forest** ve **KNN** algoritmaları için `GridSearchCV` ile en uygun parametreler belirlendi.
- En iyi parametrelere sahip modeller seçildi ve eğitildi.

### 3️⃣ Voting Classifier ile Nihai Model Oluşturma
- En iyi 3 model (SVM, RF, KNN) birleştirilerek `VotingClassifier` yapısı oluşturuldu.
- Test verisiyle modelin performansı değerlendirildi.

### 4️⃣ Sonuçların Değerlendirilmesi
- Doğruluk oranları ve en iyi parametreler ekrana yazdırıldı.
- `confusion_matrix` ile tahmin performansı görselleştirildi.
- Modellerin karşılaştırmalı sonuçları analiz edildi.

---

## 🧪 Sonuçlar

- 🔍 Her algoritma için en iyi hiperparametreler başarıyla belirlendi.
- 🧠 **Voting Classifier**, test verisinde yüksek doğruluk ile sınıflandırma yaptı.
- 📊 Model performansı **confusion matrix** ile grafiksel olarak sunuldu.

---
## Görseller

### PCA ile 2 Boyutlu Veri Görselleştirme
![image](https://github.com/user-attachments/assets/f0d0dd1a-1b4d-411e-bb6b-048b33b2f435)

### Confusion Matrix
![image](https://github.com/user-attachments/assets/089d283f-b9fc-4091-adb2-07c0d3747433)


## 👨‍💻 Geliştirici

Bu proje **Ömer Faruk Karakoy** tarafından geliştirilmiştir. 
