# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 00:00:00 2025

@author: omerf
"""

## Veri yükleme ve ön işleme
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#veriyi yükle
digits = load_digits()
X, y = digits.data, digits.target

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#veriyi standardize hale getirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

## pca ile boyut indirgeme (dimension reduction)
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

pca_2d = PCA(n_components=2)
X_train_pca_2d = pca_2d.fit_transform(X_train_scaled)
X_test_pca_2d = pca_2d.transform(X_test_scaled)

plt.figure()
for i in np.unique(y_train):
    plt.scatter(X_train_pca_2d[y_train == i,0], X_train_pca_2d[y_train == i,1],
               label = digits.target_names[i])

plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("PCA ile 2 boyutlu veri gorsellestirme")
plt.legend(title = "Siniflar", loc = "best")
plt.show()

## model training and grid search

#svm
svm_params = {"C":[0.1,1,10],"kernel":["linear","rbf"]}
svm = SVC()
svm_grid = GridSearchCV(svm, svm_params, cv=5)
svm_grid.fit(X_train_pca, y_train)

#random forest
rf_params = {"n_estimators":[50,100,200]}
rf = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf, rf_params, cv=5)
rf_grid.fit(X_train_pca, y_train)

#knn
knn_params = {"n_neighbors":[3,5,7]}
knn = KNeighborsClassifier()
knn_grid = GridSearchCV(knn, knn_params, cv=5)
knn_grid.fit(X_train_pca, y_train)

#en iyi parametrelere sahip modelin seçilmesi
best_svm = svm_grid.best_estimator_
best_rf = rf_grid.best_estimator_
best_knn = knn_grid.best_estimator_

## voting classifier
voting_clf = VotingClassifier(estimators=[
    ("svm",best_svm),
    ("rf",best_rf),
    ("knn",best_knn)],voting="hard")

voting_clf.fit(X_train_pca, y_train)

y_pred = voting_clf.predict(X_test_pca)


## Sonuclarin degerlendirilmesi
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=digits.target_names)
disp.plot(cmap = plt.cm.Blues)
plt.show()

print(f"Best scm params: {svm_grid.best_params_}")
print(f"Best rf params: {rf_grid.best_params_}")
print(f"Best knn params: {knn_grid.best_params_}")
print(f"Voting classifier accuracy: {voting_clf.score(X_test_pca, y_test)}")