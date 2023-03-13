import os
import sys
import numpy as np
import pandas as pd
import dill

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

        logging.info("Object saves as a pickle file.")
        
    except Exception as e:
        raise CustomException(e,sys)