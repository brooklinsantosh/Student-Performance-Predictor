import os
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

#Data related
DATA_PATH = "notebook\data\stud.csv"

TRAIN_DATA_FILE_NAME = "train.csv"
TEST_DATA_FILE_NAME = "test.csv"
RAW_DATA_FILE_NAME = "raw.csv"

#Train Test Split
TEST_SIZE = 0.2
RANDOM_STATE = 42

#Artifacts 
PREPROCESSOR_OBJ_PATH = os.path.join(os.getcwd(), "artifacts", "preprocessor.pkl")
MODEL_FILE_PATH = os.path.join(os.getcwd(), "artifacts", "model.pkl")

#Target
TARGET_COLUMN = "math_score"

#Models
MODELS = {
    "Random Forest": RandomForestRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "Linear Regression": LinearRegression(),
    #"K-Neighbors Regressor": KNeighborsRegressor(),
    "XGB Regressor": XGBRegressor(),
    "CatBoost Regressor": CatBoostRegressor(verbose=False),
    "AdaBoost Regressor": AdaBoostRegressor(),
}

PARAMS = params={
    "Decision Tree": {
        'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        # 'splitter':['best','random'],
        # 'max_features':['sqrt','log2'],
    },
    "Random Forest":{
        # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        
        # 'max_features':['sqrt','log2',None],
        'n_estimators': [8,16,32,64,128,256]
    },
    "Gradient Boosting":{
        # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
        'learning_rate':[.1,.01,.05,.001],
        'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
        # 'criterion':['squared_error', 'friedman_mse'],
        # 'max_features':['auto','sqrt','log2'],
        'n_estimators': [8,16,32,64,128,256]
    },
    "Linear Regression":{},
    "XGB Regressor":{
        'learning_rate':[.1,.01,.05,.001],
        'n_estimators': [8,16,32,64,128,256]
    },
    "CatBoost Regressor":{
        'depth': [6,8,10],
        'learning_rate': [0.01, 0.05, 0.1],
        'iterations': [30, 50, 100]
    },
    "AdaBoost Regressor":{
        'learning_rate':[.1,.01,0.5,.001],
        # 'loss':['linear','square','exponential'],
        'n_estimators': [8,16,32,64,128,256]
    } 
}
