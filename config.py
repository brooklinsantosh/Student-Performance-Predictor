import os

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
