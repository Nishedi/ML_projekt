from sklearn.datasets import load_wine
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
import numpy as np
from knn_classifier import KNNClassifier
import utils

def create_latex_chart(data_dict, fig_label, caption,  ylabel, xlabel, data_dict2, filename="k_chart.tex"):
    colors = ["red", "blue", "green", "yellow", "purple"]

    x_values = list(data_dict.keys())
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\\begin{figure}[h!]\n")
        f.write("\\centering\n")
        f.write("\\begin{tikzpicture}\n")
        f.write("\\begin{axis}[\n")
        f.write("width=0.8\\textwidth,\n")
        f.write(f"ylabel={{{ylabel}}},\n")
        f.write(f"xlabel={{{xlabel}}},\n")
        f.write(f"xtick={{ {','.join(map(str, x_values))} }},\n")
        f.write("legend pos=north west\n")
        f.write("]\n")

        f.write(f"\\addplot[no markers, color=black]\n")
        f.write("coordinates {\n")
        for key, data in data_dict.items():
            f.write(f"({key},{round(data, 3)})")
        f.write("};\n")
        f.write("\\addlegendentry{średnio}\n")
        iter = 0
        for p, values in data_dict2.items():
            color = colors[iter % len(colors)]
            iter += 1
            f.write(f"\\addplot[no markers, color={color}]\n")
            f.write("coordinates {\n")
            for x, y in values:
                f.write(f"({x},{round(y, 3)})")
            f.write("};\n")
            f.write(f"\\addlegendentry{{p={p}}}\n")
        f.write("\\end{axis}\n")
        f.write("\\end{tikzpicture}\n")
        f.write("\\caption{"+caption+"}\n")
        f.write("\\label{fig:"+fig_label+"}\n")
        f.write("\\end{figure}\n")



def optimize_k_p_crossval(X, y, k_values, p_values, k_fold=5, seed = 42):
    best_score = -1
    best_k = None
    best_p = None

    kf = KFold(n_splits=k_fold, shuffle=True, random_state=seed)
    iter = 0
    k_chart = dict()
    k_chart2 = dict()
    for k in k_values:

        errors = []
        for p in p_values:
            iter+=1
            scores = []
            errors2 = []
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = KNNClassifier(k=k, p=p)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)

                num_classes = len(np.unique(y))
                ut = utils.Utils(num_classes)
                score = ut.f1_score(y_val, y_pred, average='macro')
                scores.append(score)
                errors.append(1 - np.mean(y_pred == y_val))
                errors2.append(1 - np.mean(y_pred == y_val))

            mean_score = np.mean(scores)

            if mean_score > best_score:
                best_score = mean_score
                best_k = k
                best_p = p
            k_chart2[(k,p)]=round(float(np.float64(sum(errors)/len(errors))), 3)
        k_chart[k]=round(float(np.float64(sum(errors)/len(errors))), 3)
    return best_k, best_p, k_chart, k_chart2

data = load_wine() #data loading
X = pd.DataFrame(data.data, columns=data.feature_names) #dataframe creation
y = pd.Series(data.target) #target creation
k_values = range(1, 16) #k values
p_values = [1, 1.5, 2, 3, 4] #p values

seed = 512
best_k, best_p, k_chart, k_chart2 = optimize_k_p_crossval(X, y, k_values, p_values, k_fold=5, seed=seed)
print(f"Seed: {seed}, Najlepsze parametry: k={best_k}, p={best_p}")
nowy_slownik = {}
for (k, p), value in k_chart2.items():
    if p not in nowy_slownik:
        nowy_slownik[p] = []
    nowy_slownik[p].append((k, value))
create_latex_chart(k_chart, "k_comp", "Porówanie jakości wyników dla różnych wartości k oraz różnych próbek", "Błąd klasyfikacji", "Liczba sąsiadów", nowy_slownik,filename="k_chart.tex")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

model = KNNClassifier(k=best_k, p=best_p)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

num_classes = len(np.unique(y))
ut = utils.Utils(num_classes)

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
