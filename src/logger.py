import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(),"logs",datetime.now().strftime('%d_%m_%Y'))
os.makedirs(logs_path, exist_ok= True)

LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename= LOG_FILE_PATH,
    format= "[%(asctime)s] %(lineno)d %(levelname)s %(message)s",
    datefmt= "%Y-%m-%d %H:%M:%S",
    level= logging.INFO
)