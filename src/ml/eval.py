# imports
from typing import Tuple, Dict
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from mlflow import log_metric

# local absolute imports
from src.utils.utils import category_mapping


def evaluate_pipeline(pipeline: Pipeline, test_data: Tuple) -> Dict:
    """Evaluates a Pipeline on test data.

    Expected to be called in an mlflow experiment context.

    Args:
        pipeline (Pipeline): Pipeline for evaluation.
        test_data (Tuple): Test data features and labels x_test, y_test.

    Returns:
        metrics (Dict): A sklearn classification report.
    """

    report = classification_report(
        test_data[1],
        pipeline.predict(test_data[0]),
        labels=None,
        output_dict=True,
        target_names=[c for c in category_mapping().values()]
    )

    for category, metrics in zip(report.keys(), report.values()):
        if not category == "accuracy":
            for metric, score in zip(metrics.keys(), metrics.values()):
                log_metric(f"{category}_{metric}", score)

    return report
