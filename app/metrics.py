from prometheus_client import Counter


predictions_total = Counter(
    "ml_predictions_total",
    "Total number of successful ML predictions",
    ["flower"]
)