# Classifying-the-Representation-of-Numbers
# PCA ve Makine Öğrenmesi ile Rakam Tanıma

Proje Açıklaması

Bu proje, sklearn.datasets içerisindeki digits veri setini kullanarak el yazısı rakamların sınıflandırılmasını hedefler. Modelleme sürecinde PCA (Principal Component Analysis) ile boyut indirgeme işlemi uygulanarak model performansı artırılmıştır. Destek Vektör Makineleri (SVM), Rastgele Orman (Random Forest) ve K-En Yakın Komşu (KNN) algoritmaları Grid Search kullanılarak optimize edilmiştir. Son olarak, Voting Classifier ile en iyi modeller birleştirilerek nihai bir sınıflandırıcı oluşturulmuştur.

Kullanılan Kütüphaneler

-numpy

-matplotlib

-sklearn

Adımlar

1-Veri Yükleme ve Ön İşleme

  load_digits() ile veri seti yüklenir.

  Eğitim ve test veri setleri train_test_split ile ayrılır.

  StandardScaler kullanılarak veriler standardize edilir.

  PCA ile boyut indirgeme yapılır.

  PCA ile 2 boyutlu görselleştirme gerçekleştirilir.

2-Model Eğitimi ve Optimizasyonu

  GridSearchCV ile SVM, Random Forest ve KNN algoritmaları için en iyi hiperparametreler bulunur.

  En iyi parametrelere sahip modeller seçilir.

3-Voting Classifier ile Son Model Oluşturma

  Üç model birleştirilerek VotingClassifier oluşturulur.

  Nihai model test verisi ile değerlendirilir.

4-Sonuçların Değerlendirilmesi

  Confusion matrix oluşturulup görselleştirilir.

  En iyi parametreler ve doğruluk skoru ekrana yazdırılır.

Sonuçlar

-En iyi hiperparametreler ekrana yazdırılır.

-Voting Classifier doğruluk skoru hesaplanır ve gösterilir.

-Sonuçlar confusion matrix ile görselleştirilir.

## Görseller

### PCA ile 2 Boyutlu Veri Görselleştirme
![image](https://github.com/user-attachments/assets/f0d0dd1a-1b4d-411e-bb6b-048b33b2f435)

### Confusion Matrix
![image](https://github.com/user-attachments/assets/089d283f-b9fc-4091-adb2-07c0d3747433)

