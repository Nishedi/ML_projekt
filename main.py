from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from knn_classifier import KNNClassifier


# Wczytanie danych
data = load_wine() # 178 próbek, 13 cech

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Inicjalizacja i trenowanie modelu
model = KNNClassifier(k=5)
model.fit(X_train, y_train)

# Predykcja


y_pred = model.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)
print(f"Dokładność modelu: {accuracy * 100:.2f}%")
