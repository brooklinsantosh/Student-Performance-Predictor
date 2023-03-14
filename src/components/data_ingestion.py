import os
import sys

import config
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation, DataTransformationConfig, DataTransformationResult
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts',config.TRAIN_DATA_FILE_NAME)
    test_data_path: str = os.path.join('artifacts',config.TEST_DATA_FILE_NAME)
    raw_data_path: str = os.path.join('artifacts',config.RAW_DATA_FILE_NAME)

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self)-> DataIngestionConfig:
        logging.info("Entered the data ingestion initiate method or component.")
        try:
            df = pd.read_csv(config.DATA_PATH)
            logging.info("Read the dataset as dataframe.")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok= True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split intiated.")
            train_df, test_df = train_test_split(df, test_size= config.TEST_SIZE, random_state= config.RANDOM_STATE)

            train_df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed.")

            return self.ingestion_config

        except Exception as e: 
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    di_result = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    dt_result = data_transformation.initiate_data_transformation(di_result.train_data_path, di_result.test_data_path)
    
    model_trainer = ModelTrainer() 
    model_trainer.initiate_model_trainer(dt_result.train_arr, dt_result.test_arr, dt_result.preprocessor_obj_file_path)

