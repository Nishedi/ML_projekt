from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from knn_classifier import KNNClassifier
import numpy as np


def compute_accuracy(y_true, y_pred):
    """Oblicza dokładność klasyfikacji."""
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred, average='macro'):
    """Oblicza precyzję. Obsługuje micro/macro averaging."""
    confusion_matrix = compute_confusion_matrix(y_true, y_pred)

    if average == 'macro':
        precisions = []
        for i in range(num_classes):
            TP = confusion_matrix[i, i]
            FP = sum(confusion_matrix[:, i]) - TP
            precision_i = TP / (TP + FP) if (TP + FP) > 0 else 0
            precisions.append(precision_i)
        return sum(precisions) / num_classes

    elif average == 'micro':
        TP = np.sum(np.diag(confusion_matrix))
        FP = np.sum(confusion_matrix) - TP
        print(FP)
        return TP / (TP + FP) if (TP + FP) > 0 else 0


def recall(y_true, y_pred, average='macro'):
    """Oblicza recall. Obsługuje micro/macro averaging."""
    confusion_matrix = compute_confusion_matrix(y_true, y_pred)

    if average == 'macro':
        recalls = []
        for i in range(num_classes):
            TP = confusion_matrix[i, i]
            FN = sum(confusion_matrix[i, :]) - TP
            recall_i = TP / (TP + FN) if (TP + FN) > 0 else 0
            recalls.append(recall_i)
        return sum(recalls) / num_classes

    elif average == 'micro':
        TP = np.sum(np.diag(confusion_matrix))
        FN = np.sum(confusion_matrix) - TP
        print(FN)
        return TP / (TP + FN) if (TP + FN) > 0 else 0


def f1_score(y_true, y_pred, average='macro'):
    """Oblicza F1-score. Obsługuje micro/macro averaging."""
    prec = precision(y_true, y_pred, average)
    rec = recall(y_true, y_pred, average)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

def compute_confusion_matrix(y_true, y_pred):
    """Wyznacza macierz pomyłek."""
    num_classes = len(np.unique(y_true))
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        confusion_matrix[true_label, pred_label] += 1

    return confusion_matrix


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
confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

for true_label, pred_label in zip(y_test, y_pred):
    confusion_matrix[true_label, pred_label] += 1



# Wyświetlenie macierzy pomyłek
print("Macierz Pomyłek:")
print(confusion_matrix)

# Ocena modelu
accuracy = compute_accuracy(y_test, y_pred)
print(f"Dokładność modelu: {accuracy * 100:.2f}%")

print(f"Precyzja macro: {precision(y_test, y_pred, average='macro')}")
print(f"Precyzja micro: {precision(y_test, y_pred, average='micro')}")

print(f"Recall macro: {recall(y_test, y_pred, average='macro')}")
print(f"Recall micro: {recall(y_test, y_pred, average='micro')}")

print(f"F1-score macro: {f1_score(y_test, y_pred, average='macro')}")
print(f"F1-score micro: {f1_score(y_test, y_pred, average='micro')}")