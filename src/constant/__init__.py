import os
import sys

TRAIN_PATH=os.path.join('data','coin_gecko_2022-03-16.csv')
TEST_PATH=os.path.join('data','coin_gecko_2022-03-17.csv')

artifacts_folder='artifacts'

MODEL_FILE_NAME='model.pkl'
MODEL_FILE_EXTENSION='.pkl'
TARGET_COLUMN='price'

print(f"TRAIN PATH {TRAIN_PATH}")
print(f"{os.path.exists(TRAIN_PATH)}")