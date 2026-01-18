from fastapi import APIRouter, HTTPException
from models import TrainRequest, TrainResponse, PredictRequest, PredictResponse
from modules.model_training import train_and_select_best, save_model
from modules.model_deployment import predict
import pandas as pd
from modules.utils import get_logger
import uuid

logger = get_logger(__name__)

router = APIRouter()

@router.post("/train", response_model=TrainResponse)
def train_model(request: TrainRequest):
    try:
        X = pd.DataFrame(request.X)
        y = pd.Series(request.y)
        model, metrics = train_and_select_best(X, y)
        model_filename = f"{uuid.uuid4()}.pkl"
        save_model(model, model_filename)
        return TrainResponse(model_filename=model_filename, metrics=metrics)
    except Exception as e:
        logger.error(f"Train error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict", response_model=PredictResponse)
def make_prediction(request: PredictRequest):
    try:
        data = pd.DataFrame(request.data)
        predictions = predict(request.model_filename, data)
        return PredictResponse(predictions=predictions)
    except Exception as e:
        logger.error(f"Predict error: {e}")
        raise HTTPException(status_code=500, detail=str(e))