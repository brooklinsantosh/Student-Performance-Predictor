import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

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

from sklearn.metrics import r2_score

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_model
import config

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = config.MODEL_FILE_PATH

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr: np.array, test_arr: np.array, preprocessor_path: str) -> None:
        try:
            logging.info("Splitting training and validation input data.")
            #Need to change code to add validation strategy
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                #"K-Neighbors Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            model_report: pd.DataFrame = evaluate_model(
                X_train= X_train, 
                y_train= y_train, 
                X_test= X_test,
                y_test= y_test,
                models= models)
            
            logging.info("Trained all the models.")
            logging.debug(f"Models Report: \n {model_report}")

            best_model_test_score = model_report['Test_Score'].max()
            logging.debug(f"Best test score: {best_model_test_score}")
            best_model_name = model_report[model_report['Test_Score'] == best_model_test_score]['Model_Name'].values[0]
            
            if best_model_test_score < 0.6:
                raise CustomException("No best model found.")

            best_model= models[best_model_name]
            logging.info(f"Best model is: {best_model_name} with r2_score : {best_model_test_score}")

            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )

            

        except Exception as e:
            raise CustomException(e,sys)