from sklearn.datasets import load_wine
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
import numpy as np
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
    worst_score = 1
    best_k = None
    best_p = None
    worst_k = None
    worst_p = None
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
            if mean_score < worst_score:
                worst_score = mean_score
                worst_k = k
                worst_p = p

            k_chart2[(k, p)] = round(np.mean(errors_p), 3)
        k_chart[k] = round(np.mean(errors), 3)

    return best_k, best_p, k_chart, k_chart2, worst_k, worst_p

# main
seed = 512
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

k_values = range(1, 16)
p_values = [1, 1.5, 2, 3, 4]

# hiperparameters optimization
best_k, best_p, k_chart_mean, k_chart, worst_k, worst_p = optimize_k_p_crossval(X, y, k_values, p_values, k_fold=5, seed=seed)
print(f"Seed: {seed}, Najlepsze parametry: k={best_k}, p={best_p}")

# Creating chart
k_chart_p = {}
for (k, p), value in k_chart.items():
    k_chart_p.setdefault(p, []).append((k, value))

create_latex_chart(
    data_dict=k_chart_mean,
    data_dict2=k_chart_p,
    fig_label="k_comp",
    caption="Porównanie jakości wyników dla różnych wartości $k$ oraz metryk $p$",
    ylabel="Błąd klasyfikacji",
    xlabel="Liczba sąsiadów"
)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
model = KNNClassifier(k=best_k, p=best_p)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

ut = Utils(len(np.unique(y)))
print("Macierz Pomyłek:")
print(ut.compute_confusion_matrix(y_test, y_pred))
print(f"Dokładność: {ut.compute_accuracy(y_test, y_pred):.4f}")
print(f"Precyzja macro: {ut.precision(y_test, y_pred, average='macro'):.4f}")
print(f"Precyzja micro: {ut.precision(y_test, y_pred, average='micro'):.4f}")
print(f"Recall macro: {ut.recall(y_test, y_pred, average='macro'):.4f}")
print(f"Recall micro: {ut.recall(y_test, y_pred, average='micro'):.4f}")
print(f"F1-score macro: {ut.f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"F1-score micro: {ut.f1_score(y_test, y_pred, average='micro'):.4f}")

#Ploting decision boundaries for best and worst models
ut.plot_decision_boundaries(X, y, best_k, best_p, filename="figures/tsne_best_knn2.png")
ut.plot_decision_boundaries(X, y, worst_k, worst_p, filename="figures/tsne_worst_knn2.png")



