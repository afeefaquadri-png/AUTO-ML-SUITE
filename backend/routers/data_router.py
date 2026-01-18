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
    logger.info(f"Received file upload: {file.filename}, size: {file.size} bytes")
    try:
        # Read file contents
        contents = await file.read()
        logger.info(f"Read {len(contents)} bytes from file")
        
        # Check if file is empty
        if not contents:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        
        # Validate file extension
        if not any(file.filename.endswith(ext) for ext in ['.csv', '.xlsx', '.xls']):
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload CSV or Excel files.")
        
        # Reset file pointer and parse based on extension
        df = None
        try:
            if file.filename.endswith('.csv'):
                # Try to read CSV with multiple encoding options and settings
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                df = None
                last_error = None
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(
                            io.BytesIO(contents), 
                            encoding=encoding,
                            on_bad_lines='skip',
                            engine='python'
                        )
                        if not df.empty and len(df.columns) > 0:
                            break
                    except Exception as e:
                        last_error = e
                        continue
                
                if df is None or df.empty or len(df.columns) == 0:
                    raise HTTPException(status_code=400, detail=f"Unable to parse CSV file with any encoding. Last error: {str(last_error)}")
                    
            elif file.filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(io.BytesIO(contents))
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type. Please upload CSV or Excel files.")
        except pd.errors.EmptyDataError as e:
            logger.error(f"EmptyDataError for file {file.filename}: {e}")
            raise HTTPException(status_code=400, detail="The uploaded file is empty or contains no data. Please ensure the file has headers and data rows.")
        except pd.errors.ParserError as e:
            logger.error(f"ParserError for file {file.filename}: {e}")
            raise HTTPException(status_code=400, detail="Unable to parse file. Please check the file format and try again.")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected parsing error for file {file.filename}: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")
        
        # Validate dataframe
        if df.empty:
            raise HTTPException(status_code=400, detail="The uploaded file contains no data rows.")
        
        if len(df.columns) == 0:
            raise HTTPException(status_code=400, detail="The uploaded file contains no columns.")
        
        preview = df.head(5).to_dict('records')
        return UploadResponse(
            message="File uploaded successfully",
            columns=list(df.columns),
            shape=df.shape,
            preview=preview,
            data=df.to_dict('records')
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error for file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

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