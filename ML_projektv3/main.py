import sklearn.svm as svm
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from utils import compute_accuracy, precision, recall, f1_score, standard_scale, plot_points, variance_threshold
from OvRClassifier import OvRClassifier
from Perceptron import Perceptron
import pandas as pd

withStandardScaler = True
withVarianceThreshold = True
def test(isSvm=False,withStandardScaler=True, withVarianceThreshold=True, isPloting=False):
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    if withVarianceThreshold:
        X, _ = variance_threshold(X, threshold=0.1)
    # Przeskalowanie
    if withStandardScaler:
        X = standard_scale(X)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model OvR
    model = None
    if isSvm:
        model = svm.SVC(kernel='linear', C=1.0, random_state=42)
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
        plot_points(X, y)
        print("Accuracy:", accuracy)
        print("Precision:", _precision)
        print("Recall:", _recall)
        print("F1 Score:", f1)

    return accuracy, _precision, _recall, f1


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

check_impact_of_variance_and_scaling()
