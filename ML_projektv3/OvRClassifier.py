
import numpy as np
from Perceptron import Perceptron

class OvRClassifier:
    def __init__(self, base_class=Perceptron, **kwargs):
        self.base_class = base_class
        self.classifiers = {}
        self.kwargs = kwargs

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        for cls in self.classes_:
            binary_y = np.where(y == cls, 1, -1)
            clf = self.base_class(**self.kwargs)
            clf.fit(X, binary_y)
            self.classifiers[cls] = clf

    def predict(self, X):
        probas = {cls: clf.predict_proba(X) for cls, clf in self.classifiers.items()}
        probas_matrix = np.vstack([probas[cls] for cls in self.classes_]).T
        return self.classes_[np.argmax(probas_matrix, axis=1)]
