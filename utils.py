import numpy as np

class Utils:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def compute_accuracy(self, y_true, y_pred):
        """Oblicza dokładność klasyfikacji."""
        return np.mean(y_true == y_pred)


    def precision(self, y_true, y_pred, average='macro'):
        """Oblicza precyzję. Obsługuje micro/macro averaging."""
        confusion_matrix = self.compute_confusion_matrix(y_true, y_pred)

        if average == 'macro':
            precisions = []
            for i in range(self.num_classes):
                TP = confusion_matrix[i, i]
                FP = sum(confusion_matrix[:, i]) - TP
                precision_i = TP / (TP + FP) if (TP + FP) > 0 else 0
                precisions.append(precision_i)
            return sum(precisions) / self.num_classes

        elif average == 'micro':
            TP = np.sum(np.diag(confusion_matrix))
            FP = np.sum(confusion_matrix) - TP
            print(FP)
            return TP / (TP + FP) if (TP + FP) > 0 else 0


    def recall(self, y_true, y_pred, average='macro'):
        """Oblicza recall. Obsługuje micro/macro averaging."""
        confusion_matrix = self.compute_confusion_matrix(y_true, y_pred)

        if average == 'macro':
            recalls = []
            for i in range(self.num_classes):
                TP = confusion_matrix[i, i]
                FN = sum(confusion_matrix[i, :]) - TP
                recall_i = TP / (TP + FN) if (TP + FN) > 0 else 0
                recalls.append(recall_i)
            return sum(recalls) / self.num_classes

        elif average == 'micro':
            TP = np.sum(np.diag(confusion_matrix))
            FN = np.sum(confusion_matrix) - TP
            print(FN)
            return TP / (TP + FN) if (TP + FN) > 0 else 0


    def f1_score(self, y_true, y_pred, average='macro'):
        """Oblicza F1-score. Obsługuje micro/macro averaging."""
        prec = self.precision(y_true, y_pred, average)
        rec = self.recall(y_true, y_pred, average)
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    def compute_confusion_matrix(self, y_true, y_pred):
        """Wyznacza macierz pomyłek."""
        num_classes = len(np.unique(y_true))
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

        for true_label, pred_label in zip(y_true, y_pred):
            confusion_matrix[true_label, pred_label] += 1

        return confusion_matrix
