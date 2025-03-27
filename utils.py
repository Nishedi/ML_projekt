import numpy as np
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from knn_classifier import KNNClassifier
import pandas as pd

class Utils:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def compute_accuracy(self, y_true, y_pred):
        """Oblicza dokładność klasyfikacji."""
        return np.mean(y_true == y_pred)


    def precision(self, y_true, y_pred, average='macro'):
        """Oblicza precyzję. Obsługuje micro/macro averaging."""
        confusion_matrix = self.compute_confusion_matrix(y_true, y_pred)

        if average == 'macro':
            precisions = []
            for i in range(self.num_classes):
                TP = confusion_matrix[i, i]
                FP = sum(confusion_matrix[:, i]) - TP
                precision_i = TP / (TP + FP) if (TP + FP) > 0 else 0
                precisions.append(precision_i)
            return sum(precisions) / self.num_classes

        elif average == 'micro':
            TP = np.sum(np.diag(confusion_matrix))
            FP = np.sum(confusion_matrix) - TP
            return TP / (TP + FP) if (TP + FP) > 0 else 0


    def recall(self, y_true, y_pred, average='macro'):
        """Oblicza recall. Obsługuje micro/macro averaging."""
        confusion_matrix = self.compute_confusion_matrix(y_true, y_pred)

        if average == 'macro':
            recalls = []
            for i in range(self.num_classes):
                TP = confusion_matrix[i, i]
                FN = sum(confusion_matrix[i, :]) - TP
                recall_i = TP / (TP + FN) if (TP + FN) > 0 else 0
                recalls.append(recall_i)
            return sum(recalls) / self.num_classes

        elif average == 'micro':
            TP = np.sum(np.diag(confusion_matrix))
            FN = np.sum(confusion_matrix) - TP
            return TP / (TP + FN) if (TP + FN) > 0 else 0


    def f1_score(self, y_true, y_pred, average='macro'):
        """Oblicza F1-score. Obsługuje micro/macro averaging."""
        prec = self.precision(y_true, y_pred, average)
        rec = self.recall(y_true, y_pred, average)
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    def compute_confusion_matrix(self, y_true, y_pred):
        """Wyznacza macierz pomyłek."""
        num_classes = 3
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        for true_label, pred_label in zip(y_true, y_pred):
            confusion_matrix[true_label, pred_label] += 1

        return confusion_matrix

    def plot_decision_boundaries(self, X, y, k, p, filename, seed=512):
        # dimension reduction
        X_embedded = TSNE(n_components=2, random_state=seed).fit_transform(X)
        X_embedded_df = pd.DataFrame(X_embedded, columns=["dim1", "dim2"])

        # Podział na zbiór treningowy i testowy
        X_train_emb, X_test_emb, y_train_emb, y_test_emb = train_test_split(
            X_embedded_df, y, test_size=0.2, random_state=seed
        )

        # Trening na zredukowanych danych
        model_emb = KNNClassifier(k=k, p=p)
        model_emb.fit(X_embedded_df, y)

        # Siatka punktów
        h = 0.5
        x_min, x_max = X_embedded[:, 0].min() - 1, X_embedded[:, 0].max() + 1
        y_min, y_max = X_embedded[:, 1].min() - 1, X_embedded[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_df = pd.DataFrame(grid_points, columns=["dim1", "dim2"])

        # Predykcja etykiet dla punktów siatki
        grid_pred = model_emb.predict(grid_df)
        Z = np.array(grid_pred).reshape(xx.shape)

        # Wykres
        plt.figure(figsize=(10, 6))
        cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
        cmap_bold = ["red", "green", "blue"]

        plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)

        for class_value in np.unique(y):
            idx = y == class_value
            plt.scatter(
                X_embedded_df.loc[idx, "dim1"],
                X_embedded_df.loc[idx, "dim2"],
                label=f"Klasa {class_value}",
                c=cmap_bold[class_value],
                edgecolor="k",
                s=60
            )

        plt.title(f"t-SNE: Granice decyzyjne k-NN (k={k}, p={p})")
        plt.xlabel("Wymiar 1 (t-SNE)")
        plt.ylabel("Wymiar 2 (t-SNE)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()