import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from knn_classifier import KNNClassifier

# Wczytanie danych
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Podział na zbiór treningowy (80%), walidacyjny (10%) i testowy (10%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Lista wartości k i p do sprawdzenia
k_values = range(1, 16)
p_values = [1, 1.5, 2, 2.5, 3, 4]
results = {}

# Przeszukiwanie najlepszej wartości k i p na zbiorze walidacyjnym
for p in p_values:
    errors = []  # Przechowywanie błędu dla każdego k
    for k in k_values:
        model = KNNClassifier(k=k, p=p)
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)

        error = 1 - np.mean(y_val_pred == y_val)  # Błąd klasyfikacji (1 - accuracy)
        errors.append(error)

    results[p] = errors

# Rysowanie wykresów dla różnych wartości p
plt.figure(figsize=(10, 6))
for p, errors in results.items():
    plt.plot(k_values, errors, marker='o', label=f"p={p}")

plt.xlabel("Liczba sąsiadów (k)")
plt.ylabel("Błąd klasyfikacji (1 - Accuracy)")
plt.title("Wpływ k i p na skuteczność modelu KNN")
plt.legend()
plt.grid()
plt.show()

# Wybór najlepszego k i p
best_k = None
best_p = None
best_error = float("inf")

for p, errors in results.items():
    for i, error in enumerate(errors):
        if error < best_error:
            best_error = error
            best_k = k_values[i]
            best_p = p

print(f"Najlepsze k: {best_k}, Najlepsze p: {best_p}, Najniższy błąd: {best_error:.4f}")

# Trenowanie finalnego modelu z najlepszym k i p
final_model = KNNClassifier(k=best_k, p=best_p)
final_model.fit(X_train, y_train)

# Predykcja na zbiorze testowym
y_test_pred = final_model.predict(X_test)

# Ocena końcowa
test_accuracy = np.mean(y_test_pred == y_test)
print(f"Testowa dokładność dla k={best_k}, p={best_p}: {test_accuracy:.4f}")
