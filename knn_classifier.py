import random

import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3, metric='euclidean', p=2):
        self.k = k
        self.p = p
        self.metric = metric

    def fit(self, X_train, y_train):
        """Trenuje model na danych X_train z etykietami y_train."""
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """Przewiduje klasy dla danych X_test."""
        predictions = []
        for x in X_test.values:
            distances = []
            for x_train in self.X_train.values:
                distances.append(self._compute_distance(x, x_train))

            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train.iloc[k_indices]
            most_common = Counter(k_nearest_labels).most_common(1)
            counter = Counter(k_nearest_labels)
            max_count = max(counter.values())  # Znajdź największą liczbę wystąpień
            candidates = [k for k, v in counter.items() if v == max_count]
            chosen = random.choice(candidates)
            #predictions.append(chosen)
            predictions.append(most_common[0][0])
        return np.array(predictions)

    def _compute_distance(self, x1, x2):
        """Oblicza odległość między dwoma punktami (metryka Minkowskiego, dla p = 2 to metryka euklidesowa)."""
        return np.sum(np.abs(x1 - x2) ** self.p) ** (1 / self.p)



