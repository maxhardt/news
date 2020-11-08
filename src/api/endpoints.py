# imports
import logging
from typing import Dict
import numpy as np
from fastapi import APIRouter, HTTPException
from starlette.requests import Request

# local absolute imports
from src.utils.utils import category_mapping
from run_ml import run_training
from src.api.model import NewsCategory, NewsTitle, NewsClassifier


def get_prediction(request: Request, payload: NewsTitle) -> NewsCategory:
    """Predicts the category of a given news title.

    Args:
        payload (NewsTitle): The json payload of the request with
            fields defined in the pydantic ```NewsTitle```object.

    Raises:
        HTTPException: An exception if no model is found.

    Returns:
        NewsCategory: The json response with fields defined in
            the pydantic ```NewsCategory```object.
    """

    title = payload.title

    try:
        classifier: NewsClassifier = request.app.state.model
    except:
        raise HTTPException(
            status_code=400,
            detail="No model found."
        )

    prediction: np.array = classifier.predict(title)
    label: str = prediction[0]
    category: str = category_mapping()[label]

    return NewsCategory(
        title=title,
        label=label,
        category=category
    )

def train_and_deploy(request: Request, pipeline_config: str) -> Dict:
    """Optional endpoint for training and deploying a new model.

    Returns:
        response (Dict): jsonified dictionary with information
            about the training run and the new model id.
    """

    run, final_params, test_results = run_training(config_file=pipeline_config)
    new_model = NewsClassifier(run.info.run_id)
    request.app.state.model = new_model

    return {
        "new model id": run.info.run_id,
        "final hyperparameters": final_params,
        "evaluation results": test_results,
        "run information": run.info,
    }
