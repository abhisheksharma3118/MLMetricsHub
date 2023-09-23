class ThresholdAnalysis:
    def __init__(self, y_true, y_pred_probabilities):
        self.y_true = y_true
        self.y_pred_probabilities = y_pred_probabilities

    def analyze_thresholds(self, threshold_values):
        threshold_metrics = {}

        for threshold in threshold_values:
            y_pred_thresholded = [1 if prob >= threshold else 0 for prob in self.y_pred_probabilities]
            metrics_calculator = ClassificationMetrics(self.y_true, y_pred_thresholded)
            metrics = metrics_calculator.calculate_metrics()
            threshold_metrics[f'Threshold {threshold}'] = metrics

        return threshold_metrics
