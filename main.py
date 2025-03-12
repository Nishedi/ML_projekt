from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from knn_classifier import KNNClassifier
import numpy as np
import utils


# Wczytanie danych

data = load_wine() # 178 próbek, 13 cech

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Inicjalizacja i trenowanie modelu
model = KNNClassifier(k=5)
model.fit(X_train, y_train)

# Predykcja


y_pred = model.predict(X_test)
num_classes = len(np.unique(y))
ut = utils.Utils(num_classes)
confusion_matrix = ut.compute_confusion_matrix(y_test, y_pred)

# Wyświetlenie macierzy pomyłek
print("Macierz Pomyłek:")
print(confusion_matrix)

# Ocena modelu
accuracy = ut.compute_accuracy(y_test, y_pred)
print(f"Dokładność modelu: {accuracy * 100:.2f}%")

print(f"Precyzja macro: {ut.precision(y_test, y_pred, average='macro')}")
print(f"Precyzja micro: {ut.precision(y_test, y_pred, average='micro')}")

print(f"Recall macro: {ut.recall(y_test, y_pred, average='macro')}")
print(f"Recall micro: {ut.recall(y_test, y_pred, average='micro')}")

print(f"F1-score macro: {ut.f1_score(y_test, y_pred, average='macro')}")
print(f"F1-score micro: {ut.f1_score(y_test, y_pred, average='micro')}")


