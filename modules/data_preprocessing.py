import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from modules.utils import get_logger

logger = get_logger(__name__)

def select_features_target(df, features, target):
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe")
    if not all(f in df.columns for f in features):
        missing = [f for f in features if f not in df.columns]
        raise ValueError(f"Feature columns not found in dataframe: {missing}")
    X = df[features]
    y = df[target]
    logger.info(f"Selected features: {features}, target: {target}")
    return X, y

def handle_missing(X, strategy_num='mean', strategy_cat='most_frequent'):
    num_cols = X.select_dtypes(include=['number']).columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    
    transformers = []
    if len(num_cols) > 0:
        transformers.append(('num_imputer', SimpleImputer(strategy=strategy_num), num_cols))
    if len(cat_cols) > 0:
        transformers.append(('cat_imputer', SimpleImputer(strategy=strategy_cat), cat_cols))
    
    if transformers:
        preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
        X = pd.DataFrame(preprocessor.fit_transform(X), columns=X.columns)
    logger.info("Handled missing values")
    return X

def encode_categorical(X, encoding='onehot'):
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) == 0:
        return X
    
    if encoding == 'onehot':
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        encoded = encoder.fit_transform(X[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
        X = X.drop(cat_cols, axis=1)
        X = pd.concat([X, encoded_df], axis=1)
    elif encoding == 'label':
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
    logger.info(f"Encoded categorical columns with {encoding}")
    return X

def scale_numerical(X, scaler='standard'):
    num_cols = X.select_dtypes(include=['number']).columns
    if len(num_cols) == 0:
        return X
    
    if scaler == 'standard':
        sc = StandardScaler()
    elif scaler == 'minmax':
        sc = MinMaxScaler()
    else:
        logger.warning(f"Unknown scaler type '{scaler}', defaulting to standard")
        sc = StandardScaler()
    
    X[num_cols] = sc.fit_transform(X[num_cols])
    logger.info(f"Scaled numerical columns with {scaler} scaler")
    return X