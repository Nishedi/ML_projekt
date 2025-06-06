import csv
import os
import random
import time

import numpy as np
import sklearn.svm as svm
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from knn_classifier import KNNClassifier
from utils import compute_accuracy, precision, recall, f1_score, standard_scale, plot_points, compute_confusion_matrix
from OvRClassifier import OvRClassifier
from Perceptron import Perceptron


withStandardScaler = True
withVarianceThreshold = True
csv_file = "metrics.csv"
file_exists = os.path.isfile(csv_file)
if file_exists:
    os.remove(csv_file)
if os.path.isfile("metrixTime.csv"):
    os.remove("metrixTime.csv")

def test_with_confusion_matrix(model, model_name, withStandardScaler=True, random_state=42):
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    if withStandardScaler:
        X = standard_scale(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Model: {model_name}")
    print("Confusion Matrix:\n", compute_confusion_matrix(y_test, y_pred))


def test_with_crossvalidation(model,model_name,withStandardScaler=True, withVarianceThreshold=True, isPloting=False, k_fold=5, seed=42):
    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=seed)
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

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
    scores_time_learning = []
    scores_time_prediction = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        start_learning_time = time.time()
        model.fit(X_train, y_train)
        scores_time_learning.append(time.time() - start_learning_time)
        start_prediction_time = time.time()
        y_pred = model.predict(X_val)
        scores_time_prediction.append(time.time() - start_prediction_time)
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
    mean_time_learning = np.mean(scores_time_learning)
    mean_time_prediction = np.mean(scores_time_prediction)

    if isPloting:
        print("Accuracy:", round(mean_accuracy,3))
        print("Precision:", round(mean_precision,3))
        print("Precision Micro:", round(mean_precision_micro,3))
        print("Recall:", round(mean_recall,3))
        print("Recall Micro:", round(mean_recall_micro,3))
        print("F1 Score:", round(mean_score_f1,3))
        print("F1 Score Micro:", round(mean_score_f1_micro,3))
        print("Learning Time:", round(mean_time_learning,3))
        print("Prediction Time:", round(mean_time_prediction,3))



        file_exists = os.path.isfile(csv_file)

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file, delimiter=';')

            if not file_exists:
                writer.writerow([
                    "model", "accuracy", "precision_macro", "precision_micro",
                    "recall_macro", "recall_micro", "f1_macro", "f1_micro"
                ])

            writer.writerow([
                model_name,
                round(mean_accuracy, 3),
                round(mean_precision, 3),
                round(mean_precision_micro, 3),
                round(mean_recall, 3),
                round(mean_recall_micro, 3),
                round(mean_score_f1, 3),
                round(mean_score_f1_micro, 3)
            ])

        with open("metrixTime.csv", mode='a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            if not file_exists:
                writer.writerow(["model", "learning_time", "prediction_time"])
            writer.writerow([
                model_name,
                round(mean_time_learning, 3),
                round(mean_time_prediction, 3)
            ])

    return round(mean_accuracy,3), round(mean_precision,3), round(mean_recall,3), round(mean_score_f1,3), round(mean_time_learning,3), round(mean_time_prediction,3)

def optimize_models():
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    X = standard_scale(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = []

    # === 1. BaggingClassifier ===
    bagging = BaggingClassifier(n_estimators=10, random_state=42)
    param_grid_bagging = {
        'n_estimators': [10, 50, 100, 150, 200],
        'max_samples': [0.1, 0.3, 0.5, 0.7, 1.0]
    }

    grid_bagging = GridSearchCV(bagging, param_grid_bagging, cv=cv, scoring='f1_macro', n_jobs=-1)
    grid_bagging.fit(X, y)
    best_bagging = grid_bagging.best_estimator_
    print(f"\nBest Bagging Params: {grid_bagging.best_params_}")
    acc, prec, rec, f1, t1, t2 = test_with_crossvalidation(best_bagging, "bagging", isPloting=True)
    results.append(["bagging", acc, prec, rec, f1, str(grid_bagging.best_params_)])

    # === 2. GradientBoostingClassifier ===
    gradient_boosting = GradientBoostingClassifier(random_state=42)
    param_grid_gb = {
        'n_estimators': [10, 50, 100, 150, 200],
        'learning_rate':  [0.01,0.01, 0.1, 0.2, 0.5]
    }

    grid_gb = GridSearchCV(gradient_boosting, param_grid_gb, cv=cv, scoring='f1_macro', n_jobs=-1)
    grid_gb.fit(X, y)
    best_gb = grid_gb.best_estimator_
    print(f"\nBest Gradient Boosting Params: {grid_gb.best_params_}")
    acc, prec, rec, f1, t1, t2 = test_with_crossvalidation(best_gb, "gradient boosting", isPloting=True)
    results.append(["gradient boosting", acc, prec, rec, f1, str(grid_gb.best_params_)])

    # === 3. HistGradientBoostingClassifier ===
    hist_gb = HistGradientBoostingClassifier(random_state=42)
    param_grid_hist = {
        'learning_rate': [0.01,0.01, 0.1, 0.2, 0.5],
        'max_depth': [1, 3, 5, 7, 9]
    }

    grid_hist = GridSearchCV(hist_gb, param_grid_hist, cv=cv, scoring='f1_macro', n_jobs=-1)
    grid_hist.fit(X, y)
    best_hist = grid_hist.best_estimator_
    print(f"\nBest HistGradientBoosting Params: {grid_hist.best_params_}")
    acc, prec, rec, f1, t1, t2 = test_with_crossvalidation(best_hist, "hist gradient boosting", isPloting=True)
    results.append(["hist gradient boosting", acc, prec, rec, f1, str(grid_hist.best_params_)])

    # === 4. RandomForest ===
    from sklearn.ensemble import RandomForestClassifier
    random_forest = RandomForestClassifier(random_state=42)
    param_grid_rf = {
        'n_estimators': [10, 50, 100, 150, 200],
        'max_depth': [1, 3, 5, 7, 9]
    }
    grid_rf = GridSearchCV(random_forest, param_grid_rf, cv=cv, scoring='f1_macro', n_jobs=-1)
    grid_rf.fit(X, y)
    best_rf = grid_rf.best_estimator_
    print(f"\nBest Random Forest Params: {grid_rf.best_params_}")
    acc, prec, rec, f1, t1, t2 = test_with_crossvalidation(best_rf, "random forest", isPloting=True)
    results.append(["random forest", acc, prec, rec, f1, str(grid_rf.best_params_)])


    # === Zapis do CSV ===
    csv_file = "optimized_results.csv"
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        if not file_exists:
            writer.writerow(["model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "best_params"])
        for row in results:
            writer.writerow(row)

model_knn, model_knn_name = KNNClassifier(k=1, p=1), "knn"
model_ovr = OvRClassifier(base_class=Perceptron, learning_rate=0.01, n_iter=1000), "ovr"
model_bagging_classifier = BaggingClassifier(n_estimators=150, max_samples=0.1, random_state=42), "bagging"
model_gradient_boosting = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42), "gradient boosting"
model_hist_gradient_boosting_classifier = HistGradientBoostingClassifier(learning_rate=0.2, max_depth=3, random_state=42), "hist gradient boosting"

# print("KNN")
# test_with_crossvalidation(model_knn, model_knn_name, withStandardScaler, withVarianceThreshold, True)
# print("OvR")
# test_with_crossvalidation(model_ovr[0], model_ovr[1], withStandardScaler, withVarianceThreshold, True)
# print("Bagging")
# test_with_crossvalidation(model_bagging_classifier[0],model_bagging_classifier[1], withStandardScaler, withVarianceThreshold, True)
# print("Gradient Boosting")
# test_with_crossvalidation(model_gradient_boosting[0],model_gradient_boosting[1], withStandardScaler, withVarianceThreshold, True)
# print("Hist Gradient Boosting")
# test_with_crossvalidation(model_hist_gradient_boosting_classifier[0],model_hist_gradient_boosting_classifier[1], withStandardScaler, withVarianceThreshold, True)
#
test_with_confusion_matrix(model_knn, model_knn_name, withStandardScaler)
test_with_confusion_matrix(model_ovr[0], model_ovr[1], withStandardScaler)
test_with_confusion_matrix(model_bagging_classifier[0], model_bagging_classifier[1], withStandardScaler)
test_with_confusion_matrix(model_gradient_boosting[0], model_gradient_boosting[1], withStandardScaler)
test_with_confusion_matrix(model_hist_gradient_boosting_classifier[0], model_hist_gradient_boosting_classifier[1], withStandardScaler)

# optimize_models()

# def test(isSvm=False,withStandardScaler=True, withVarianceThreshold=True, isPloting=False, random_state=42):
#     data = load_wine()
#     X = pd.DataFrame(data.data, columns=data.feature_names)
#     y = pd.Series(data.target)
#     if withVarianceThreshold:
#         X, _ = variance_threshold(X, threshold=0.1)
#     if withStandardScaler:
#         X = standard_scale(X)
#
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y)
#
#     model = None
#     if isSvm:
#         model = svm.SVC(kernel='linear', C=1.0, random_state=random_state)
#     else:
#         model = OvRClassifier(base_class=Perceptron, learning_rate=0.01, n_iter=1000)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#
#     # Metryki
#     accuracy = compute_accuracy(y_test, y_pred)
#     _precision = precision(y_test, y_pred, average='macro')
#     _recall = recall(y_test, y_pred, average='macro')
#     f1 = f1_score(y_test, y_pred, average='macro')
#
#
#     if isPloting:
#         filename="x.png"
#         if isSvm:
#             filename = "svm_tsne_visualisation.png"
#         else:
#             filename = "tsne_visualisation.png"
#         plot_points(X, y,filename=filename, isSvm=isSvm,random_state=random_state)
#         print("Accuracy:", round(accuracy,3))
#         print("Precision:", round(_precision,3))
#         print("Recall:", round(_recall,3))
#         print("F1 Score:", round(f1,3))
#         print("Confusion Matrix:\n", compute_confusion_matrix(y_test, y_pred))
#
#     return round(accuracy,3), round(_precision,3), round(_recall,3), round(f1,3)
#
# def check_impact_of_variance_and_scaling():
#     x, y, w, z = 0,0,0,0
#     num_of_rep = 20
#     for i in range(20):
#         acc, _,_,_= test(False,True, True)
#         x += acc
#     for i in range(20):
#         acc, _,_,_= test(False,True, False)
#         y += acc
#     for i in range(20):
#         acc, _,_,_= test(False,False, True)
#         w += acc
#     for i in range(20):
#         acc, _,_,_= test(False,False, False)
#         z += acc
#
#     print("Srednia accuracy dla znormalizowanych danych: ", x/num_of_rep)
#     print("Srednia accuracy dla nienormalizowanych danych: ", y/num_of_rep)
#     print("Srednia accuracy dla znormalizowanych danych z progowaniem: ", w/num_of_rep)
#     print("Srednia accuracy dla nienormalizowanych danych z progowaniem: ", z/num_of_rep)