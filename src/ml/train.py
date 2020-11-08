# imports
import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from mlflow import log_params, log_artifact

# local absolute imports
from src.utils.utils import load_dataset_from_csv, get_train_test_data
from src.ml.eval import evaluate_pipeline


def get_sklearn_pipeline() -> Pipeline:
    """Initializes an sklearn pipeline without parameters.

    Returns:
        pipeline (Pipeline): An sklearn Pipeline object.
    """

    pipeline = Pipeline([
            ("vectorizer", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("naivebayes", MultinomialNB()),
    ])

    return pipeline


def train_and_hyperparameter_search(
    pipeline: Pipeline, config: Dict, x_train: np.array, y_train: np.array
) -> Tuple:
    """Trains a pipeline with hyperparameters using grid search.

    Args:
        pipeline (Pipeline): sklearn Pipeline object.
        config (Dict): Dict with hyperparameters for training.
        x_train (np.array): Array with training features.
        y_train (np.array): Array with training labels.

    Returns:
        best_estimator (Pipeline): The fitted final pipeline.
        best_params (Dict): The best hyperparameters.
    """

    # grid search model fitting
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=config["parameters"],
        cv=config["search"]["k_splits"],
        scoring=config["search"]["metric"],
        n_jobs=config["search"]["n_jobs"]
    ).fit(x_train, y_train)

    # store grid search results as csv
    _tmp_file = "grid_search_results.csv"
    pd.DataFrame(grid_search.cv_results_).to_csv(_tmp_file)
    log_artifact(_tmp_file, "grid_search_results")
    os.remove(_tmp_file)

    return grid_search.best_estimator_, grid_search.best_params_
