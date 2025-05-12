
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from utils import compute_accuracy, precision, recall, f1_score, standard_scale, plot_points, variance_threshold
from OvRClassifier import OvRClassifier
from Perceptron import Perceptron
import pandas as pd
# Dane
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
X = variance_threshold(X, threshold=0.01)
# Przeskalowanie
X = standard_scale(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model OvR
model = OvRClassifier(base_class=Perceptron, learning_rate=0.01, n_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
#
# Metryki

print("Accuracy:", compute_accuracy(y_test, y_pred))
print("Precision:", precision(y_test, y_pred, average='macro'))
print("Recall:", recall(y_test, y_pred, average='macro'))
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))

plot_points(X,y)