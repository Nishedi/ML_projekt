import pandas as pd
import matplotlib as plt
plt.use('tkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import utils

from Perceptron import Perceptron

def optimize_lr_crossval(X,y, lr_values, k_fold=5, seed=42, n_iter = 30):
    best_score = -1
    worst_score = 1
    best_lr = None
    worst_lr = None
    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=seed)
    k_chart = dict()
    sc = []
    for lr in lr_values:
        scores = []
        scores_precision = []
        scores_recall = []
        scores_accuracy = []
        for train_idx, val_idx in skf.split(X,y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            model = Perceptron(learning_rate=lr, n_iter=n_iter)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = utils.f1_score(y_val, y_pred)
            scores.append(score)
            accuracy = utils.compute_accuracy(y_val, y_pred)
            precision = utils.precision(y_val, y_pred)
            recall = utils.recall(y_val, y_pred)
            scores_accuracy.append(accuracy)
            scores_precision.append(precision)
            scores_recall.append(recall)

        mean_score = np.mean(scores)
        mean_accuracy = np.mean(scores_accuracy)
        mean_precision = np.mean(scores_precision)
        mean_recall = np.mean(scores_recall)
        print(f"Learning rate: {lr}, Accuracy: {mean_accuracy:.3f}, Precision: {mean_precision:.3f}, Recall: {mean_recall:.3f}")
        sc.append(mean_accuracy)
        if mean_score > best_score:
            best_score = mean_score
            best_lr = lr
        if mean_score < worst_score:
            worst_score = mean_score
            worst_lr = lr

        k_chart[lr] = round(mean_score, 3)
    return best_lr, k_chart, worst_lr, sc

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
df = pd.read_csv(url, header=None)
df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'class']

X = df.drop("class", axis=1).values
y = df["class"].values


X_var_filtered, _ = utils.variance_threshold(X, threshold=0.01)

X_scaled = utils.standard_scale(X_var_filtered)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

learning_rates = [0] + [10**exp for exp in range(-7, 1)]
scs = []
best_rates = []

for n_iter in [10,50,100]:
    print(f"n_iter={n_iter}")
    best_rate, scores, worst_rate, sc = optimize_lr_crossval(X_train, y_train, learning_rates, n_iter=n_iter)
    scs.append(sc)
    best_rates.append(best_rate)

for idx, sc in enumerate(scs):
    for i in range(len(sc)):
        sc[i] = round(1 - sc[i], 3)  # przekształcenie na "błąd" (1 - F1)
    label = f"n_iter={ [10,50,100][idx] }, best_lr={best_rates[idx]}"
    plt.plot(learning_rates, sc, label=label)

plt.xscale('log')
plt.legend([f"n_iter={i}" for i in [10,50,100]])
plt.xlabel("Learning Rate (log scale)")
plt.ylabel("Błąd klasyfikacji")
plt.title(f"Optymalizacja learning_rate")
plt.grid(True)
plt.show()

#
model = Perceptron(learning_rate=best_rates[2], n_iter=100)
utils.plot_tsne_decision_boundary_perceptron(X, y, model, filename="tsne_perceptron_boundaryx.png")
