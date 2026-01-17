import os

# Database configurations
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/auto_ml_db")
POSTGRES_URI = os.getenv("POSTGRES_URI", "postgresql://user:password@localhost/auto_ml_db")

# Logging configuration
LOG_FILE = "logs/auto_ml.log"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Model save path
MODEL_PATH = "models/"

# Other configs
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB