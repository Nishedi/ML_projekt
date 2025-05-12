import numpy as np
import sklearn.svm as svm
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, StratifiedKFold
from utils import compute_accuracy, precision, recall, f1_score, standard_scale, plot_points, variance_threshold
from OvRClassifier import OvRClassifier
from Perceptron import Perceptron
import pandas as pd

withStandardScaler = True
withVarianceThreshold = True
def test(isSvm=False,withStandardScaler=True, withVarianceThreshold=True, isPloting=False, k_fold=5, seed=42):
    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=seed)
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    if withVarianceThreshold:
        X, _ = variance_threshold(X, threshold=0.1)
    # Przeskalowanie
    if withStandardScaler:
        X = standard_scale(X)

    scores_f1 = []
    scores_precision = []
    scores_recall = []
    scores_accuracy = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = None
        if isSvm:
            model = svm.SVC(kernel='linear', C=1.0, random_state=42)
        else:
            model = OvRClassifier(base_class=Perceptron, learning_rate=0.01, n_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        score_f1 = f1_score(y_val, y_pred)
        scores_f1.append(score_f1)
        accuracy = compute_accuracy(y_val, y_pred)
        precision_score = precision(y_val, y_pred)
        recall_score = recall(y_val, y_pred)
        scores_accuracy.append(accuracy)
        scores_precision.append(precision_score)
        scores_recall.append(recall_score)
    mean_score_f1 = np.mean(scores_f1)
    mean_accuracy = np.mean(scores_accuracy)
    mean_precision = np.mean(scores_precision)
    mean_recall = np.mean(scores_recall)


    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #
    # # Model OvR
    # model = None
    # if isSvm:
    #     model = svm.SVC(kernel='linear', C=1.0, random_state=42)
    # else:
    #     model = OvRClassifier(base_class=Perceptron, learning_rate=0.01, n_iter=1000)
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    #
    # # Metryki
    # accuracy = compute_accuracy(y_test, y_pred)
    # _precision = precision(y_test, y_pred, average='macro')
    # _recall = recall(y_test, y_pred, average='macro')
    # f1 = f1_score(y_test, y_pred, average='macro')


    if isPloting:
        plot_points(X, y)
        print("Accuracy:", mean_accuracy)
        print("Precision:", mean_precision)
        print("Recall:", mean_recall)
        print("F1 Score:", mean_score_f1)

    return mean_accuracy, mean_precision, mean_recall, mean_score_f1


def check_impact_of_variance_and_scaling():
    x, y, w, z = 0,0,0,0
    num_of_rep = 20
    for i in range(20):
        acc, _,_,_= test(False,True, True)
        x += acc
    for i in range(20):
        acc, _,_,_= test(False,True, False)
        y += acc
    for i in range(20):
        acc, _,_,_= test(False,False, True)
        w += acc
    for i in range(20):
        acc, _,_,_= test(False,False, False)
        z += acc

    print("Srednia accuracy dla znormalizowanych danych: ", x/num_of_rep)
    print("Srednia accuracy dla nienormalizowanych danych: ", y/num_of_rep)
    print("Srednia accuracy dla znormalizowanych danych z progowaniem: ", w/num_of_rep)
    print("Srednia accuracy dla nienormalizowanych danych z progowaniem: ", z/num_of_rep)

test(False,True, True, True)
test(True,True, True, True)
