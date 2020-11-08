# imports
import os
from contextlib import suppress
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
@click.argument("config-file", type=click.Path(exists=True))
@click.option("--verbose", default=True, is_flag=True)
def run_training(config_file: str, verbose: bool) -> None:
    """CLI for training, hyperparametersearch and evaluation.

    Args:
        config_file (str): Path to the .yaml file with hyperparameters for training.
        verbose (bool): If true, prints results and meta-information of the run.

    Returns:
        run (mlflow.run): mlflow.run object with meta-information on the current run.
        final_params (Dict): Final hyperparameters of the fitted pipeline.
        test_results (Dict): Test evaluation results of the final pipeline.
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

        print(f"\nMeta-information on the training run: \n{run.info}\n")
        print(f"\nFinal hyperparameters from gridsearch: \n{final_params}\n")
        print(f"\nAchieved evaluation results on test set: \n{test_results}\n")

        # show meta information about the run
        return run, final_params, test_results

if __name__ == "__main__":
    run_training() # pylint: disable=no-value-for-parameter
