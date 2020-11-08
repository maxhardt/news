# imports
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

# local absolute imports
from src.api.model import NewsClassifier
from src.api.endpoints import get_prediction


load_dotenv()
MODEL_ID = os.getenv("MODEL_ID")

def start_app_handler(app: FastAPI) -> None:
    """Handles loading the model when starting the app.

    Args:
        app (FastAPI): Application FastAPI object.
    """
    def _startup() -> None:
        model_instance = NewsClassifier(MODEL_ID)
        app.state.model = model_instance
    return _startup

app = FastAPI(
    title="News classifier API",
    description="Classifies news titles into 4 distinct categories.",
    version="0.1",
    debug=True
)

app.add_event_handler("startup", start_app_handler(app))
app.add_api_route("/predict", get_prediction, methods=["POST"])
app.add_api_route("/", lambda: app.title, methods=["GET"])
