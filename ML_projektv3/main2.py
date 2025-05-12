
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from utils import compute_accuracy, precision, recall, f1_score, standard_scale, plot_tsne_decision_boundary_perceptron, plot_points
from OvRClassifier import OvRClassifier
from Perceptron import Perceptron

# Dane
data = load_wine()
X, y = data.data, data.target

# Przeskalowanie
X = standard_scale(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model OvR
model = OvRClassifier(base_class=Perceptron, learning_rate=0.01, n_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metryki
for i in range(len(y_test)):
    if y_test[i] != y_pred[i]:
        print(f"i={i}: False: {y_test[i]}, Predicted: {y_pred[i]}")

print("Accuracy:", compute_accuracy(y_test, y_pred))
print("Precision:", precision(y_test, y_pred, average='macro'))
print("Recall:", recall(y_test, y_pred, average='macro'))
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))

from OvRClassifier import OvRClassifier
# plot_tsne_decision_boundary_perceptron(
#     X, y,
#     base_model_class=OvRClassifier,
#     model_kwargs={"base_class": Perceptron, "learning_rate": 0.01, "n_iter": 1000}
# )
plot_points(X,y)