import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from modules.utils import get_logger
import joblib
from config import MODEL_PATH
import os

logger = get_logger(__name__)

def detect_problem_type(y):
    if pd.api.types.is_numeric_dtype(y):
        return 'regression'
    else:
        return 'classification'

def get_models(problem_type):
    if problem_type == 'classification':
        models = {
            'LogisticRegression': (LogisticRegression(), {'C': [0.1, 1, 10]}),
            'RandomForest': (RandomForestClassifier(), {'n_estimators': [10, 50, 100]}),
            'SVM': (SVC(), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']})
        }
    else:
        models = {
            'LinearRegression': (LinearRegression(), {}),
            'RandomForest': (RandomForestRegressor(), {'n_estimators': [10, 50, 100]}),
            'SVM': (SVR(), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']})
        }
    return models

def evaluate_model(model, X_test, y_test, problem_type):
    y_pred = model.predict(X_test)
    if problem_type == 'classification':
        acc = accuracy_score(y_test, y_pred)
        return {'accuracy': acc}
    else:
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        return {'r2': r2, 'mse': mse}

def train_and_select_best(X, y, test_size=0.2):
    problem_type = detect_problem_type(y)
    logger.info(f"Detected problem type: {problem_type}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    models = get_models(problem_type)
    best_model = None
    best_score = -float('inf')
    best_metrics = {}
    
    for name, (model, params) in models.items():
        if params:
            grid = GridSearchCV(model, params, cv=3, scoring='accuracy' if problem_type == 'classification' else 'r2')
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
        else:
            model.fit(X_train, y_train)
        
        metrics = evaluate_model(model, X_test, y_test, problem_type)
        score = metrics['accuracy'] if problem_type == 'classification' else metrics['r2']
        
        if score > best_score:
            best_score = score
            best_model = model
            best_metrics = metrics
            best_metrics['model_name'] = name
    
    logger.info(f"Best model: {best_metrics['model_name']}, score: {best_score}")
    return best_model, best_metrics

def save_model(model, filename):
    os.makedirs(MODEL_PATH, exist_ok=True)
    path = os.path.join(MODEL_PATH, filename)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")
    return path

def load_model(filename):
    path = os.path.join(MODEL_PATH, filename)
    model = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return model