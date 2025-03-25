from sklearn.datasets import load_wine
from sklearn.model_selection import KFold, train_test_split
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import seaborn as sns
from knn_classifier import KNNClassifier
from utils import Utils

def create_latex_chart(data_dict, fig_label, caption, ylabel, xlabel, data_dict2, filename="k_chart.tex"):
    colors = ["red", "blue", "green", "yellow", "purple"]
    x_values = list(data_dict.keys())
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\\begin{figure}[h!]\n\\centering\n\\begin{tikzpicture}\n\\begin{axis}[\n")
        f.write("width=0.8\\textwidth,\n")
        f.write(f"ylabel={{{ylabel}}},\n")
        f.write(f"xlabel={{{xlabel}}},\n")
        f.write(f"xtick={{ {','.join(map(str, x_values))} }},\n")
        f.write("legend pos=north west\n]\n")
        f.write("\\addplot[no markers, color=black]\ncoordinates {\n")
        for key, data in data_dict.items():
            f.write(f"({key},{round(data, 3)})\n")
        f.write("};\n\\addlegendentry{średnio}\n")
        for i, (p, values) in enumerate(data_dict2.items()):
            color = colors[i % len(colors)]
            f.write(f"\\addplot[no markers, color={color}]\ncoordinates {{\n")
            for x, y in values:
                f.write(f"({x},{round(y, 3)})\n")
            f.write("};\n")
            f.write(f"\\addlegendentry{{p={p}}}\n")
        f.write("\\end{axis}\n\\end{tikzpicture}\n")
        f.write(f"\\caption{{{caption}}}\n\\label{{fig:{fig_label}}}\n"
                +"\\end{figure}\n")


def optimize_k_p_crossval(X, y, k_values, p_values, k_fold=5, seed=42):
    best_score = -1
    best_k = None
    best_p = None
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=seed)
    k_chart = dict()
    k_chart2 = dict()

    for k in k_values:
        errors = []
        for p in p_values:
            scores = []
            errors_p = []
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                model = KNNClassifier(k=k, p=p)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                ut = Utils(len(np.unique(y)))
                score = ut.f1_score(y_val, y_pred, average='macro')
                scores.append(score)
                errors.append(1 - np.mean(y_pred == y_val))
                errors_p.append(1 - np.mean(y_pred == y_val))

            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_k = k
                best_p = p

            k_chart2[(k, p)] = round(np.mean(errors_p), 3)
        k_chart[k] = round(np.mean(errors), 3)

    return best_k, best_p, k_chart, k_chart2


# ==== GŁÓWNY PROGRAM ====
seed = 512
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

k_values = range(1, 16)
p_values = [1, 1.5, 2, 3, 4]

# Optymalizacja hiperparametrów
best_k, best_p, k_chart, k_chart2 = optimize_k_p_crossval(X, y, k_values, p_values, k_fold=5, seed=seed)
print(f"Seed: {seed}, Najlepsze parametry: k={best_k}, p={best_p}")

# Przygotowanie danych do wykresu
nowy_slownik = {}
for (k, p), value in k_chart2.items():
    nowy_slownik.setdefault(p, []).append((k, value))

# Generowanie wykresu w LaTeX-u
create_latex_chart(
    data_dict=k_chart,
    data_dict2=nowy_slownik,
    fig_label="k_comp",
    caption="Porównanie jakości wyników dla różnych wartości $k$ oraz metryk $p$",
    ylabel="Błąd klasyfikacji",
    xlabel="Liczba sąsiadów"
)

# Ewaluacja na zbiorze testowym
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
model = KNNClassifier(k=best_k, p=best_p)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

ut = Utils(len(np.unique(y)))
print("\n== METRYKI KOŃCOWE ==")
print("Macierz Pomyłek:")
print(ut.compute_confusion_matrix(y_test, y_pred))
print(f"Dokładność: {ut.compute_accuracy(y_test, y_pred):.4f}")
print(f"Precyzja macro: {ut.precision(y_test, y_pred, average='macro'):.4f}")
print(f"Precyzja micro: {ut.precision(y_test, y_pred, average='micro'):.4f}")
print(f"Recall macro: {ut.recall(y_test, y_pred, average='macro'):.4f}")
print(f"Recall micro: {ut.recall(y_test, y_pred, average='micro'):.4f}")
print(f"F1-score macro: {ut.f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"F1-score micro: {ut.f1_score(y_test, y_pred, average='micro'):.4f}")

# === Wizualizacja t-SNE z klasyfikacją ===
from matplotlib.colors import ListedColormap

# Redukcja wymiarów
X_embedded = TSNE(n_components=2, random_state=seed).fit_transform(X)
X_embedded_df = pd.DataFrame(X_embedded, columns=["dim1", "dim2"])

# Dopasuj zredukowane dane do etykiet
X_train_emb, X_test_emb, y_train_emb, y_test_emb = train_test_split(
    X_embedded_df, y, test_size=0.2, random_state=seed
)

# Trenuj klasyfikator na zredukowanych danych
model_emb = KNNClassifier(k=best_k, p=best_p)
model_emb.fit(X_train_emb, y_train_emb)

# Stworzenie siatki punktów (grid) do wizualizacji granic decyzyjnych
h = 0.5  # gęstość siatki
x_min, x_max = X_embedded[:, 0].min() - 1, X_embedded[:, 0].max() + 1
y_min, y_max = X_embedded[:, 1].min() - 1, X_embedded[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_df = pd.DataFrame(grid_points, columns=["dim1", "dim2"])

# Predykcja etykiet dla punktów siatki
grid_pred = model_emb.predict(grid_df)
Z = np.array(grid_pred).reshape(xx.shape)

# Rysowanie wykresu
plt.figure(figsize=(10, 6))
cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
cmap_bold = ["red", "green", "blue"]

# Tło – granice decyzyjne
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)

# Punkty danych
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

plt.title("Wizualizacja t-SNE z granicami decyzyjnymi k-NN")
plt.xlabel("Wymiar 1 (t-SNE)")
plt.ylabel("Wymiar 2 (t-SNE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
