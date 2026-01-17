import pandas as pd
from pymongo import MongoClient
import psycopg2
from config import MONGO_URI, POSTGRES_URI
from modules.utils import get_logger

logger = get_logger(__name__)

def load_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded CSV from {file_path}, shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        raise

def load_excel(file_path):
    try:
        df = pd.read_excel(file_path)
        logger.info(f"Loaded Excel from {file_path}, shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading Excel: {e}")
        raise

def load_from_mongo(collection_name, query={}, db_name="auto_ml_db"):
    try:
        client = MongoClient(MONGO_URI)
        db = client[db_name]
        collection = db[collection_name]
        data = list(collection.find(query))
        df = pd.DataFrame(data)
        if '_id' in df.columns:
            df.drop('_id', axis=1, inplace=True)
        logger.info(f"Loaded data from MongoDB {collection_name}, shape: {df.shape}")
        client.close()
        return df
    except Exception as e:
        logger.error(f"Error loading from MongoDB: {e}")
        raise

def load_from_postgres(query, db_name="auto_ml_db"):
    try:
        conn = psycopg2.connect(POSTGRES_URI)
        df = pd.read_sql_query(query, conn)
        logger.info(f"Loaded data from PostgreSQL, shape: {df.shape}")
        conn.close()
        return df
    except Exception as e:
        logger.error(f"Error loading from PostgreSQL: {e}")
        raise