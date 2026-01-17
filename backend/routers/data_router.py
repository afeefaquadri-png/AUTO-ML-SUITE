from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.models import UploadResponse, DBLoadRequest, PreprocessRequest, PreprocessResponse
from modules.data_ingestion import load_csv, load_excel, load_from_mongo, load_from_postgres
from modules.data_preprocessing import select_features_target, handle_missing, encode_categorical, scale_numerical
import pandas as pd
import io
from modules.utils import get_logger

logger = get_logger(__name__)

router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        preview = df.head(5).to_dict('records')
        return UploadResponse(
            message="File uploaded successfully",
            columns=list(df.columns),
            shape=df.shape,
            preview=preview,
            data=df.to_dict('records')
        )
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/load_db", response_model=UploadResponse)
def load_db(request: DBLoadRequest):
    try:
        if request.source == 'mongo':
            df = load_from_mongo(request.collection_or_query, db_name=request.db_name)
        elif request.source == 'postgres':
            df = load_from_postgres(request.collection_or_query)
        else:
            raise HTTPException(status_code=400, detail="Invalid source")
        
        preview = df.head(5).to_dict('records')
        return UploadResponse(
            message="Data loaded from DB",
            columns=list(df.columns),
            shape=df.shape,
            preview=preview,
            data=df.to_dict('records')
        )
    except Exception as e:
        logger.error(f"DB load error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/preprocess", response_model=PreprocessResponse)
def preprocess_data(request: PreprocessRequest):
    try:
        df = pd.DataFrame(request.data)
        X, y = select_features_target(df, request.features, request.target)
        X = handle_missing(X, request.missing_strategy_num, request.missing_strategy_cat)
        X = encode_categorical(X, request.encoding)
        X = scale_numerical(X, request.scaling)
        
        return PreprocessResponse(
            X=X.to_dict('records'),
            y=y.tolist()
        )
    except Exception as e:
        logger.error(f"Preprocess error: {e}")
        raise HTTPException(status_code=500, detail=str(e))