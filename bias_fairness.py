from sklearn.metrics import confusion_matrix
from aif360.metrics import ClassificationMetric

class BiasFairnessMetrics:
    def __init__(self, y_true, y_pred, sensitive_attributes):
        self.y_true = y_true
        self.y_pred = y_pred
        self.sensitive_attributes = sensitive_attributes

    def calculate_bias_fairness_metrics(self):
        confusion = confusion_matrix(self.y_true, self.y_pred)
        classification_metric = ClassificationMetric(self.y_true, self.y_pred, self.sensitive_attributes)

        bias_fairness_metrics = {
            'Confusion Matrix': confusion,
            'Disparate Impact': classification_metric.disparate_impact(),
            'Equal Opportunity Difference': classification_metric.equal_opportunity_difference(),
            # Add more fairness metrics here
        }

        return bias_fairness_metrics
