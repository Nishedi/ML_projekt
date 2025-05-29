import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Wczytaj dane
df = pd.read_csv("cornus.csv")
df.columns = ['fruit_circ', 'seed_circ', 'fruit_len', 'seed_len', 'class']

# Dane wejściowe i etykiety
X = df.drop("class", axis=1).values
y = LabelEncoder().fit_transform(df["class"])  # przekształć klasy tekstowe do liczb
y = to_categorical(y)  # one-hot encoding

# Skalowanie cech
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Podział danych
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Budowa modelu
model = Sequential([
    Dense(32, activation='relu', input_shape=(X.shape[1],)),
    Dense(16, activation='relu'),
    Dense(y.shape[1], activation='softmax')  # tyle wyjść, ile klas
])

# Kompilacja i trening
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.1)

# Ewaluacja
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Dokładność na zbiorze testowym: {acc:.2%}")
