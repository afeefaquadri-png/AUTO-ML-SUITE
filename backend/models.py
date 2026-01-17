from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class UploadResponse(BaseModel):
    message: str
    columns: List[str]
    shape: tuple
    preview: List[Dict[str, Any]]
    data: List[Dict[str, Any]]

class DBLoadRequest(BaseModel):
    source: str  # 'mongo' or 'postgres'
    collection_or_query: str
    db_name: Optional[str] = "auto_ml_db"

class PreprocessRequest(BaseModel):
    data: List[Dict[str, Any]]
    features: List[str]
    target: str
    missing_strategy_num: Optional[str] = "mean"
    missing_strategy_cat: Optional[str] = "most_frequent"
    encoding: Optional[str] = "onehot"
    scaling: Optional[str] = "standard"

class PreprocessResponse(BaseModel):
    X: List[Dict[str, Any]]
    y: List[Any]

class TrainRequest(BaseModel):
    X: List[Dict[str, Any]]
    y: List[Any]

class TrainResponse(BaseModel):
    model_filename: str
    metrics: Dict[str, Any]

class PredictRequest(BaseModel):
    model_filename: str
    data: List[Dict[str, Any]]

class PredictResponse(BaseModel):
    predictions: List[Any]