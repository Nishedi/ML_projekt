from pathlib import Path
# Ścieżka do pliku OvRClassifier
ovr_path = Path("OvRClassifier.py")

# Wczytaj zawartość pliku
with open(ovr_path, "r") as file:
    ovr_code = file.read()

# Zastąpienie przypisania self.classes_ z dodaniem sortowania
updated_ovr_code = ovr_code.replace(
    "self.classes_ = np.unique(y)",
    "self.classes_ = np.sort(np.unique(y))"
)

# Zapisz poprawiony plik
with open(ovr_path, "w") as file:
    file.write(updated_ovr_code)

ovr_path.name
