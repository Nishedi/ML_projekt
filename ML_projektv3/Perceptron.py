import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iter=1000, patience=200):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.patience = patience
        self.weights = None
        self.bias = None

    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = np.random.randn() * 0.01

        y_train = np.where(y_train <= 0, -1, 1)
        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X_train.values):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self._heaviside(linear_output)

                update = self.learning_rate * (y_train[idx] - y_pred)
                self.weights += update * x_i
                self.bias += update
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self._activation(linear_output)


    def predict_proba(self, X):
        return np.dot(X, self.weights) + self.bias
    def _activation(self, x):
        return np.heaviside(x, 0).astype(int)

    def _heaviside(self, x):
        return np.where(x >= 0, 1, -1)
