import os
import sys
import numpy as np
import pandas as pd
import dill

from sklearn.metrics import r2_score

from src.logger import logging
from src.exception import CustomException


def save_object(file_path: str, obj: object) -> None:
    """
    This method is used to pickle any given object in the 
    given file path.
    """

    try:
        dir_path= os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Object saved as a pickle file.")
        
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train: np.array, y_train: np.array, X_test: np.array, y_test:np.array, models: dict) -> pd.DataFrame:
    """
    This method is used to evaluate the different models 
    passed and send back the report
    """
    try:
        report = pd.DataFrame(columns=['Model_Name', 'Train_Score', 'Test_Score'])

        for name, model in models.items():
            logging.info(f"Training the model: [{name}]")
            model.fit(X_train,y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            report.loc[len(report.index)] = [name, train_score, test_score]
        
        return report


    except Exception as e:
        raise CustomException(e,sys)