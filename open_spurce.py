from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

class ClassificationMetrics:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def calculate_metrics(self):
        metrics = {
            'Accuracy': accuracy_score(self.y_true, self.y_pred),
            'Precision': precision_score(self.y_true, self.y_pred),
            'Recall': recall_score(self.y_true, self.y_pred),
            'F1 Score': f1_score(self.y_true, self.y_pred),
            'ROC AUC': roc_auc_score(self.y_true, self.y_pred),
            'Confusion Matrix': confusion_matrix(self.y_true, self.y_pred)
        }
        return metrics
