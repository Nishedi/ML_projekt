import numpy as np
import sklearn.svm as svm
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, StratifiedKFold
from utils import compute_accuracy, precision, recall, f1_score, standard_scale, plot_points, variance_threshold, compute_confusion_matrix
from OvRClassifier import OvRClassifier
from Perceptron import Perceptron


withStandardScaler = True
withVarianceThreshold = True

def test_with_crossvalidation(isSvm=False,withStandardScaler=True, withVarianceThreshold=True, isPloting=False, k_fold=5, seed=42):
    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=seed)
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    if withVarianceThreshold:
        X, _ = variance_threshold(X, threshold=0.01)
    # Przeskalowanie
    if withStandardScaler:
        X = standard_scale(X)

    scores_f1 = []
    scores_f1_micro = []
    scores_precision = []
    scores_precision_micro = []
    scores_recall = []
    scores_recall_micro = []
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
        score_f1_micro = f1_score(y_val, y_pred, average='micro')
        scores_f1.append(score_f1)
        accuracy = compute_accuracy(y_val, y_pred)
        precision_score = precision(y_val, y_pred)
        precision_score_micro = precision(y_val, y_pred, average='micro')
        recall_score = recall(y_val, y_pred)
        recall_score_micro = recall(y_val, y_pred, average='micro')
        scores_accuracy.append(accuracy)
        scores_precision.append(precision_score)
        scores_recall.append(recall_score)
        scores_precision_micro.append(precision_score_micro)
        scores_recall_micro.append(recall_score_micro)
        scores_f1_micro.append(score_f1_micro)

    mean_score_f1 = np.mean(scores_f1)
    mean_accuracy = np.mean(scores_accuracy)
    mean_precision = np.mean(scores_precision)
    mean_recall = np.mean(scores_recall)
    mean_score_f1_micro = np.mean(scores_f1_micro)
    mean_precision_micro = np.mean(scores_precision_micro)
    mean_recall_micro = np.mean(scores_recall_micro)

    if isPloting:
        print("Accuracy:", round(mean_accuracy,3))
        print("Precision:", round(mean_precision,3))
        print("Precision Micro:", round(mean_precision_micro,3))
        print("Recall:", round(mean_recall,3))
        print("Recall Micro:", round(mean_recall_micro,3))
        print("F1 Score:", round(mean_score_f1,3))
        print("F1 Score Micro:", round(mean_score_f1_micro,3))


    return round(mean_accuracy,3), round(mean_precision,3), round(mean_recall,3), round(mean_score_f1,3)
def test(isSvm=False,withStandardScaler=True, withVarianceThreshold=True, isPloting=False, random_state=42):
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    if withVarianceThreshold:
        X, _ = variance_threshold(X, threshold=0.1)
    if withStandardScaler:
        X = standard_scale(X)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y)

    model = None
    if isSvm:
        model = svm.SVC(kernel='linear', C=1.0, random_state=random_state)
    else:
        model = OvRClassifier(base_class=Perceptron, learning_rate=0.01, n_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metryki
    accuracy = compute_accuracy(y_test, y_pred)
    _precision = precision(y_test, y_pred, average='macro')
    _recall = recall(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')


    if isPloting:
        filename="x.png"
        if isSvm:
            filename = "svm_tsne_visualisation.png"
        else:
            filename = "tsne_visualisation.png"
        plot_points(X, y,filename=filename, isSvm=isSvm,random_state=random_state)
        print("Accuracy:", round(accuracy,3))
        print("Precision:", round(_precision,3))
        print("Recall:", round(_recall,3))
        print("F1 Score:", round(f1,3))
        print("Confusion Matrix:\n", compute_confusion_matrix(y_test, y_pred))

    return round(accuracy,3), round(_precision,3), round(_recall,3), round(f1,3)


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

print("OvR")
test_with_crossvalidation(False, withStandardScaler, withVarianceThreshold, True)
print("SVM")
test_with_crossvalidation(True, withStandardScaler, withVarianceThreshold, True)
# print("----------------------")
# test(False,True, True, True,random_state=42)
# test(True, True, True, True,random_state=42)

