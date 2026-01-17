# Auto ML Suite

An automated machine learning platform that allows users to upload data, connect to databases, preprocess data, train models, and make predictions.

## Features

- **Data Ingestion**: Upload CSV/Excel files or connect to MongoDB/PostgreSQL databases
- **Data Preprocessing**: Handle missing values, encode categorical data, scale numerical data
- **Model Training**: Auto-detect regression/classification, train multiple models, hyperparameter tuning, select best model
- **Model Deployment**: Save models, provide prediction API
- **UI**: Streamlit-based interface for easy interaction
- **API**: FastAPI backend for programmatic access

## Project Structure

```
AUTO-ML-SUITE/
├── backend/
│   ├── main.py
│   ├── models.py
│   └── routers/
│       ├── data_router.py
│       └── model_router.py
├── frontend/
│   └── app.py
├── modules/
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_deployment.py
│   └── utils.py
├── database/
├── logs/
├── models/
├── config.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variables for databases (optional):
   - `MONGO_URI` for MongoDB connection
   - `POSTGRES_URI` for PostgreSQL connection
   - `LOG_LEVEL` (default INFO)
4. Start the backend: `uvicorn backend.main:app --reload`
5. Start the frontend: `streamlit run frontend/app.py`

## Usage

- Upload a CSV or Excel file, or connect to a database.
- Select features (X) and target (Y) columns.
- The system will preprocess the data, train models, and select the best one.
- View metrics and make predictions.

## API Endpoints

- `GET /` : Root
- `POST /api/data/upload` : Upload file
- `POST /api/data/load_db` : Load from DB
- `POST /api/data/preprocess` : Preprocess data
- `POST /api/model/train` : Train model
- `POST /api/model/predict` : Make prediction

## Usage

### API

- **POST /api/data/upload**: Upload CSV/Excel file
- **POST /api/data/load_db**: Load data from MongoDB/PostgreSQL
- **POST /api/data/preprocess**: Preprocess data
- **POST /api/model/train**: Train model
- **POST /api/model/predict**: Make predictions

### UI

Use the Streamlit app for a graphical interface.

## Requirements

- Python 3.8+
- MongoDB (optional)
- PostgreSQL (optional)

## License

MIT
