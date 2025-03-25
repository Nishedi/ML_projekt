import os

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from knn_classifier import KNNClassifier
import numpy as np
import utils
import os


def test(X, y, filename="results.csv", k=5, n=2, test_size=0.1, num_of_rep=10, is_visualisation=False):
    if not os.path.exists(filename) and not is_visualisation:
        preambula = "k;n;test_size;accuracy;precision_macro;precision_micro;recall_macro;recall_micro;f1_macro;f1_micro\n"
        with open(filename, "w") as file:
            file.write(preambula)

    accuracy = 0
    precision_macro = 0
    precision_micro = 0
    recall_macro = 0
    recall_micro = 0
    f1_macro = 0
    f1_micro = 0

    for i in range(0, num_of_rep):
        # Podział na zbiór treningowy i testowy
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)#, random_state=42)

        # Inicjalizacja i trenowanie modelu
        model = KNNClassifier(k=k)
        model.fit(X_train, y_train)

        # Predykcja
        y_pred = model.predict(X_test)
        num_classes = len(np.unique(y))
        ut = utils.Utils(num_classes)


        # Ocena modelu
        accuracy += ut.compute_accuracy(y_test, y_pred)

        precision_macro += ut.precision(y_test, y_pred, average='macro')
        precision_micro += ut.precision(y_test, y_pred, average='micro')

        recall_macro += ut.recall(y_test, y_pred, average='macro')
        recall_micro += ut.recall(y_test, y_pred, average='micro')

        f1_macro += ut.f1_score(y_test, y_pred, average='macro')
        f1_micro += ut.f1_score(y_test, y_pred, average='micro')
    if is_visualisation:
        print(
            f"{k};{n};{test_size};{accuracy / num_of_rep:.2f};{precision_macro / num_of_rep:.2f};{precision_micro / num_of_rep:.2f}"
            f";{recall_macro / num_of_rep:.2f};{recall_micro / num_of_rep:.2f};{f1_macro / num_of_rep:.2f};{f1_micro / num_of_rep:.2f}")
    else:
        with open(filename, "a") as file:
            file.write(
                f"{k};{n};{test_size};{accuracy / num_of_rep:.2f};{precision_macro / num_of_rep:.2f};{precision_micro / num_of_rep:.2f}"
                f";{recall_macro / num_of_rep:.2f};{recall_micro / num_of_rep:.2f};{f1_macro / num_of_rep:.2f};{f1_micro / num_of_rep:.2f}\n"
            )


# Wczytanie danych

data = load_wine() # 178 próbek, 13 cech

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
test(X,y, num_of_rep=1, is_visualisation=True)

for k in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
    for j in [1,1.5,2,3,4]:
        test(X,y,filename="results2.csv", k=k, n=j, test_size=0.3, num_of_rep=100, is_visualisation=False)

