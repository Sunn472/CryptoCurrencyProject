import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys
from src.constant import TRAIN_PATH,TEST_PATH
from src.exception import CustomException
from dataclasses import dataclass
from src.logger import logging

@dataclass
class DataIngestionConfig:
    artifacts_folder: str='artifacts'
    train_file_name: str='coin_gecko_2022-03-16.csv'
    test_file_name: str='coin_gecko_2022-03-17.csv'


class DataIngestion:
    def __init__(self):
        self.config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Starts")
        try:
            #create artifacts folder
            os.makedirs(self.config.artifacts_folder,exist_ok=True)
            logging.info(f"articats folder created at {self.config.artifacts_folder}")
            
            #read train and test data
            df1=pd.read_csv(TRAIN_PATH)
            logging.info(f"Train read succesfully from {TRAIN_PATH} with {df1.shape}")

            df2=pd.read_csv(TEST_PATH)
            logging.info(f"Test data succefully read from {TEST_PATH} with {df2.shape}")

            
            dst_path1=os.path.join(self.config.artifacts_folder,self.config.train_file_name)
            dst_path2=os.path.join(self.config.artifacts_folder,self.config.test_file_name)
            df1.to_csv(dst_path1,index=False)
            df2.to_csv(dst_path2,index=False)
            logging.info(f"Train Data saved at {dst_path1}")
            logging.info(f"Test Data saved at {dst_path2}")
            logging.info("Data ingestion succesfully")
            
        except Exception as e:
            logging.info(f"Error occured during data ingestion str{e}")
            raise CustomException(e,sys)