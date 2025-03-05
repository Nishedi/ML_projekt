import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3, metric='euclidean'):
        self.k = k
        self.metric = metric

    def fit(self, X_train, y_train):
        """Trenuje model na danych X_train z etykietami y_train."""
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """Przewiduje klasy dla danych X_test."""
        predictions = [self._predict(x) for x in X_test.values]
        return np.array(predictions)

    def _predict(self, x):
        """Pomocnicza metoda do obliczania klasy dla pojedynczego punktu x."""
        distances = []
        for x_train in self.X_train.values:
            distances.append(self._compute_distance(x, x_train))

        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train.iloc[k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    # def _compute_distance(self, x1, x2):
    #     """Oblicza odległość między dwoma punktami (metryka euklidesowa)."""
    #     return np.sqrt(np.sum((x1 - x2) ** 2))
    def _compute_distance(self, x1, x2):

        """Oblicza odległość między dwoma punktami (metryka euklidesowa)."""
        return np.sqrt(np.sum((x1 - x2) ** 2))



