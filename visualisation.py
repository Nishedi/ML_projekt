import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from knn_classifier import KNNClassifier

# Wczytanie danych
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Redukcja wymiarów danych do 2D za pomocą t-SNE
X_embedded = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X_embedded, y, test_size=0.2, random_state=42)

# Wybór jednej wartości k

k_list = [1,2,3,4,5]
p_list = [1, 1.5, 2, 2.5, 3, 4]

# Trenowanie i predykcja
for k in k_list:
    for p in p_list:
        model = KNNClassifier(k=k)
        model.fit(pd.DataFrame(X_train), y_train)
        y_pred = model.predict(pd.DataFrame(X_test))

        # Wizualizacja wyników dla każdej klasy osobno
        unique_classes = np.unique(y)

        # Wspólny wykres z identycznymi kolorami dla tych samych klas
        plt.figure(figsize=(8, 6))
        colors = ['red', 'blue', 'green']  # Definiowanie kolorów dla klas

        for i, cls in enumerate(unique_classes):
            plt.scatter(X_train[y_train == cls, 0], X_train[y_train == cls, 1], color=colors[i],
                        label=f'Treningowe Klasa {cls}', marker='x')
            plt.scatter(X_test[y_pred == cls, 0], X_test[y_pred == cls, 1], color=colors[i], alpha=0.6,
                        label=f'Przewidywane Klasa {cls}', marker='o')

        plt.title(f"Wspólna wizualizacja KNN z t-SNE dla p={p}, k={k}")
        plt.legend()
        plt.show()