# MLMetricsHub

# MyMLMetrics

MyMLMetrics is an open-source Python package for machine learning metrics, model comparison, threshold analysis, and bias/fairness evaluation.

## Features

- Calculate common classification metrics like accuracy, precision, recall, F1 score, and ROC AUC.
- Compare the performance of multiple machine learning models.
- Perform threshold analysis to explore the impact of different classification thresholds.
- Evaluate bias and fairness metrics for machine learning models.

- ## Usage

### Calculate Classification Metrics

```python
from mymlmetrics.metrics import ClassificationMetrics
from sklearn.metrics import accuracy_score

y_true = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0]
y_pred = [0, 1, 0, 0, 1, 1, 0, 1, 0, 1]

metrics_calculator = ClassificationMetrics(y_true, y_pred)
metrics = metrics_calculator.calculate_metrics()

accuracy = metrics['Accuracy']
print(f'Accuracy: {accuracy}')

# ModelComparison

from mymlmetrics.comparison import ModelComparison

y_true = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0]
model1_preds = [0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
model2_preds = [0, 1, 1, 0, 1, 0, 0, 0, 1, 1]

comparison = ModelComparison([model1_preds, model2_preds], y_true)
model_metrics = comparison.compare_models()

for model, metrics in model_metrics.items():
    print(model)
    print(metrics)

# ThresholdAnalysis

from mymlmetrics.threshold_analysis import ThresholdAnalysis

y_true = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0]
y_pred_probabilities = [0.8, 0.6, 0.7, 0.3, 0.9, 0.4, 0.2, 0.7, 0.6, 0.5]
threshold_values = [0.4, 0.5, 0.6]

threshold_analysis = ThresholdAnalysis(y_true, y_pred_probabilities)
threshold_metrics = threshold_analysis.analyze_thresholds(threshold_values)

for threshold, metrics in threshold_metrics.items():
    print(f'Threshold: {threshold}')
    print(metrics)

# BiasFairnessMetrics

from mymlmetrics.bias_fairness import BiasFairnessMetrics

y_true = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0]
y_pred = [0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
sensitive_attributes = [0, 1, 0, 1, 1, 0, 1, 0, 1, 0]

bias_fairness = BiasFairnessMetrics(y_true, y_pred, sensitive_attributes)
bias_fairness_metrics = bias_fairness.calculate_bias_fairness_metrics()

for metric_name, metric_value in bias_fairness_metrics.items():
    print(f'{metric_name}: {metric_value}')


