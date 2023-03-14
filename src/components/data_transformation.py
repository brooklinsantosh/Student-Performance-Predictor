import sys
import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import config

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = config.PREPROCESSOR_OBJ_PATH

@dataclass
class DataTransformationResult:
    preprocessor_obj_file_path: str
    train_arr: np.array
    test_arr: np.array

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self, num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
        """
        This function is responsible to create ColumnTransformer object 
        for data transformation based on different type of columns.
        """
        try:
            numerical_columns: list[str] = num_cols
            categorical_columns: list[str] = cat_cols

            logging.debug(f"Numerical columns: {numerical_columns}.")
            logging.debug(f"Categorical columns: {categorical_columns}.")

            numerical_pipeline = Pipeline(
                steps= [
                ("imputer", SimpleImputer(strategy= "median")),
                ("scaler", StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps= [
                ("imputer", SimpleImputer(strategy= "most_frequent")),
                ("encoder", OneHotEncoder(sparse_output=False)),
                ("scaler", StandardScaler())
                ]
            )
            preprocessor = ColumnTransformer(
                [
                ("numerical_pipeline", numerical_pipeline, numerical_columns),
                ("categorical_pipeline", categorical_pipeline, categorical_columns)
                ]
            )
            logging.info("Scaler and Encoder object created.")
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path: str, test_path: str) -> DataTransformationResult:
        """
        This method will perform data transformation on train and test data.
        It imputes missing values, encode categorical columns and scale numerical columns.
        """
        try:
            logging.info("Reading train and test data.")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading train and test data completed.")

            numerical_columns = [col for col in train_df.columns if train_df[col].dtype != 'O' and col != config.TARGET_COLUMN] 
            categorical_columns = [col for col in train_df.columns if train_df[col].dtype == 'O' and col != config.TARGET_COLUMN]

            preprocessor= self.get_data_transformer_obj(numerical_columns,categorical_columns)
            logging.info("Obtained preprocessor object.")

            input_feature_train_df = train_df.drop(config.TARGET_COLUMN,axis=1)
            target_feature_train_df = train_df[config.TARGET_COLUMN]

            input_feature_test_df = test_df.drop(config.TARGET_COLUMN,axis=1)
            target_feature_test_df = test_df[config.TARGET_COLUMN]

            logging.info("Applying preprocessing on train and test dataframes.")

            logging.debug(input_feature_train_df.columns)
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Preprocessing completed.")

            logging.info("Saving preprocessor object.")
            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessor)

            return DataTransformationResult(
                preprocessor_obj_file_path= self.data_transformation_config.preprocessor_obj_file_path,
                train_arr= train_arr, 
                test_arr= test_arr
            )

        except Exception as e:
            raise CustomException(e,sys)
        
    
