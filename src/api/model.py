# imports
import logging
import numpy as np
import mlflow
from pydantic import BaseModel # pylint: disable=no-name-in-module


# pydantic data model for predictions
class NewsCategory(BaseModel):
    title: str
    label: str
    category: str

# pytantic data model for inputs
class NewsTitle(BaseModel):
    title: str

class NewsClassifier(object):
    """Wrapper for news classifier functionality.
    """

    def __init__(self, run_id: str):
        """Initializes a NewsClassifier instance.

        Args:
            run_id (str): mlflow run_id string.
        """

        self.run_id = run_id
        self.load_model_from_run_id()

    def load_model_from_run_id(self):
        """Loads a model from its mlflow run_id.
        """

        fp = f"runs:/{self.run_id}/model"
        self.model = mlflow.sklearn.load_model(fp)
        logging.info(f"model {self.run_id} loaded successfully")

    def predict(self, title: str) -> np.array:
        """Predicts the category of a single news title.

        Converts a given string ```s``` to a list ```[s]``` for sklearn model.

        Args:
            title (str): A single news title as string.

        Returns:
            prediction: A numpy array with the predicted label.
        """
        return self.model.predict([title])
