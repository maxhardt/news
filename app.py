# imports
import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from mlflow.exceptions import MlflowException

# local absolute imports
from src.api.model import NewsClassifier
from src.api.endpoints import get_prediction, train_and_deploy
from run_ml import run_training


app = FastAPI(
    title="News classifier API",
    description="Classifies news titles into 4 distinct categories.",
    version="0.1",
    debug=True
)

app.add_api_route("/predict", get_prediction, methods=["POST"])
app.add_api_route("/train_and_deploy", train_and_deploy, methods=["GET"])
app.add_api_route("/", lambda: app.title, methods=["GET"])
