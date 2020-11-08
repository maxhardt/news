# imports
import os
from contextlib import suppress
from typing import Dict
import click
import yaml
import numpy as np
from mlflow import log_params, log_artifact
from mlflow.exceptions import MlflowException
import mlflow

# local absolute imports
from src.utils.utils import load_dataset_from_csv, get_train_test_data
from src.ml.train import get_sklearn_pipeline, train_and_hyperparameter_search
from src.ml.eval import evaluate_pipeline


@click.command()
@click.option(
    "--config-file",
    default="pipeline.yaml",
    help="Name of the pipeline file with hyperparameters."
)
def run_training(config_file: str) -> None:
    """[summary]

    Args:
        config_file (str): [description]

    Returns:
        run (...): [description]
        final_params (Dict): [description]
        test_results (Dict): [description]
    """

    with open(config_file, "r") as f:
        config = yaml.safe_load(stream=f)

    _id = mlflow.set_experiment("news") # pylint: disable=assignment-from-no-return
    with suppress(MlflowException): mlflow.delete_experiment("0")

    with mlflow.start_run(experiment_id=_id) as run:

        log_artifact(config_file, "pipeline config")
        np.random.seed(config["random_seed"])

        # initialize pipeline with hyperparameters
        pipeline = get_sklearn_pipeline()

        # get training and testing data
        news = load_dataset_from_csv()
        x_train, y_train, x_test, y_test = get_train_test_data(news, config["test_size"])

        # fit and tune on training data
        final_pipeline, final_params = train_and_hyperparameter_search(
            pipeline, config, x_train, y_train)
        log_params(final_params)

        # evaluate final pipeline
        test_results = evaluate_pipeline(final_pipeline, (x_test, y_test))

        # log final model to the run
        mlflow.sklearn.log_model(
            sk_model=final_pipeline,
            artifact_path="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE
        )

        # show meta information about the run
        return run, final_params, test_results

if __name__ == "__main__":
    run_training() # pylint: disable=no-value-for-parameter
