from modules.model_training import load_model
from modules.utils import get_logger
import pandas as pd

logger = get_logger(__name__)

def predict(model_filename, data):
    model = load_model(model_filename)
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    predictions = model.predict(data)
    logger.info(f"Predictions made: {len(predictions)}")
    return predictions.tolist()