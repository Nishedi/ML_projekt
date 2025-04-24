from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_tsne_decision_boundary_perceptron(X, y, perceptron_model, filename="tsne_perceptron.png", seed=512):
    X_embedded = TSNE(n_components=2, random_state=seed).fit_transform(X)
    X_embedded_df = pd.DataFrame(X_embedded, columns=["dim1", "dim2"])

    perceptron_model.fit(X_embedded, y)

    h = 0.5
    x_min, x_max = X_embedded[:, 0].min() - 1, X_embedded[:, 0].max() + 1
    y_min, y_max = X_embedded[:, 1].min() - 1, X_embedded[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    grid_pred = perceptron_model.predict(grid_points)
    Z = np.array(grid_pred).reshape(xx.shape)

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

    plt.title("t-SNE: Granice decyzyjne Perceptronu")
    plt.xlabel("Wymiar 1 (t-SNE)")
    plt.ylabel("Wymiar 2 (t-SNE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def compute_accuracy(y_true, y_pred):
    """Oblicza dokładność klasyfikacji."""
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred, average='macro'):
    """Oblicza precyzję. Obsługuje micro/macro averaging."""
    confusion_matrix = compute_confusion_matrix(y_true, y_pred)
    if average == 'macro':
        precisions = []
        for i in range(3):
            TP = confusion_matrix[i, i]
            FP = sum(confusion_matrix[:, i]) - TP
            precision_i = TP / (TP + FP) if (TP + FP) > 0 else 0
            precisions.append(precision_i)
        return sum(precisions) / 3

    elif average == 'micro':
        TP = np.sum(np.diag(confusion_matrix))
        FP = np.sum(confusion_matrix) - TP
        return TP / (TP + FP) if (TP + FP) > 0 else 0


def recall(y_true, y_pred, average='macro'):
    """Oblicza recall. Obsługuje micro/macro averaging."""
    confusion_matrix = compute_confusion_matrix(y_true, y_pred)
    if average == 'macro':
        recalls = []
        for i in range(3):
            TP = confusion_matrix[i, i]
            FN = sum(confusion_matrix[i, :]) - TP
            recall_i = TP / (TP + FN) if (TP + FN) > 0 else 0
            recalls.append(recall_i)
        return sum(recalls) / 3

    elif average == 'micro':
        TP = np.sum(np.diag(confusion_matrix))
        FN = np.sum(confusion_matrix) - TP
        return TP / (TP + FN) if (TP + FN) > 0 else 0


def f1_score(y_true, y_pred, average='macro'):
    """Oblicza F1-score. Obsługuje micro/macro averaging."""

    prec = precision(y_true, y_pred, average)
    rec = recall(y_true, y_pred, average)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

def compute_confusion_matrix(y_true, y_pred):
    """Wyznacza macierz pomyłek."""
    classes = np.unique(np.concatenate((y_true, y_pred)))
    num_classes = len(classes)
    class_to_index = {label: idx for idx, label in enumerate(classes)}
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        confusion_matrix[class_to_index[true_label], class_to_index[pred_label]] += 1
    return confusion_matrix


def standard_scale(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0
    X_scaled = (X - mean) / std
    return X_scaled

def variance_threshold(X, threshold=0.01):
    variances = np.var(X, axis=0)
    selected_columns = variances >= threshold
    return X[:, selected_columns], selected_columns



def plot_tsne_decision_boundary_perceptron(X, y, base_model_class, model_kwargs=None, filename="tsne_perceptron.png", seed=512):
    print(len(y))
    from sklearn.manifold import TSNE
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    if model_kwargs is None:
        model_kwargs = {}

    X_embedded = TSNE(n_components=2, random_state=seed).fit_transform(X)
    X_embedded_df = pd.DataFrame(X_embedded, columns=["dim1", "dim2"])

    # Trenujemy nowy model na zredukowanych danych do wizualizacji
    model = base_model_class(**model_kwargs)
    model.fit(X_embedded, y)

    h = 0.5
    x_min, x_max = X_embedded[:, 0].min() - 1, X_embedded[:, 0].max() + 1
    y_min, y_max = X_embedded[:, 1].min() - 1, X_embedded[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    grid_pred = model.predict(grid_points)
    Z = np.array(grid_pred).reshape(xx.shape)

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

    plt.title("t-SNE: Granice decyzyjne Perceptronu (osobny model)")
    plt.xlabel("Wymiar 1 (t-SNE)")
    plt.ylabel("Wymiar 2 (t-SNE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
