import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
from src.exception import CustomException
from src.logger import logging
from src.constant import TRAIN_PATH
from src.utils.main_utils import MainUtils

class DatatransformationConfig:
    artifact_dir=os.path.join('artifacts')
    train_file_path=os.path.join(artifact_dir,'coin_gecko_2022-03-16.csv')
    test_file_path: str=os.path.join(artifact_dir,'coin_gecko_2022-03-17.csv')
    transformed_train_file_path: str=os.path.join(artifact_dir,'train.npy')
    transformed_test_file_path: str=os.path.join(artifact_dir,'test.npy')
    transformed_train_csv_path: str=os.path.join(artifact_dir,'train.csv')
    transformed_test_csv_path: str=os.path.join(artifact_dir,'test.csv')
    transformed_object_file_path: str=os.path.join(artifact_dir,'preperocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.config=DatatransformationConfig()
        self.utilis=MainUtils()
    def initiate_data_transformation(self):
        try:
            train_df=pd.read_csv(self.config.train_file_path)
            test_df=pd.read_csv(self.config.test_file_path)

            #feature engineering
            for df in [train_df, test_df]:
                df['Liquidity Ratio'] = df['24h_volume'] / df['mkt_cap']
                df['volatility_index'] = (df['1h'] + df['7d'] + df['24h']) / 3
                df['Volatility'] = df['24h_volume'] / (df['mkt_cap'] * df['volatility_index'])

                # Replace inf/-inf with NaN so SimpleImputer can handle it
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                df.drop(columns=['volatility_index'], axis=1, inplace=True)

            logging.info(f"Liquidity ratio and volattility columns created successfuly")
            
            #Convert date column
            train_df['date']=pd.to_datetime(train_df['date']) 
            test_df['date']=pd.to_datetime(test_df['date'])
            logging.info(f"Date column converted to datetime")

            #split into features and target
            X_train = train_df.drop(columns=['price'])
            y_train = train_df['price']
            X_test=test_df.drop(columns=['price'])
            y_test=test_df['price']
            logging.info(f"Train shape {X_train.shape} Test Shape {X_test.shape}")

            #due to high - cardinality categorical columns
            # drop ['24h_volume','mkt_cap']because from this we created a new feature which is more important 
            drop_cols=['coin','symbol','24h_volume','mkt_cap']
            X_train=X_train.drop(columns=drop_cols)
            X_test=X_test.drop(columns=drop_cols)
            logging.info(f"Dropped high-cardinality columns:{drop_cols}")

            #Identiify columns
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()


            #numeric pipe line
            numeric_pipeline=Pipeline([
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',MinMaxScaler())
            ])

            #Fit-transform on train and transform on test
            X_train_processed=numeric_pipeline.fit_transform(X_train[numeric_cols])
            X_test_processed=numeric_pipeline.transform(X_test[numeric_cols])

            #save processed data
            np.savez(self.config.transformed_train_file_path,X=X_train_processed,Y=y_train.values)
            np.savez(self.config.transformed_test_file_path,X=X_test_processed,Y=y_test.values)
            logging.info(f"Transformed data saved at {self.config.transformed_train_file_path}")
            
            #save as csv
            all_features_name=numeric_cols
            train_df_out=pd.DataFrame(X_train_processed,columns=all_features_name)
            test_df_out=pd.DataFrame(X_test_processed,columns=all_features_name)
            train_df_out['price']=y_train.values
            test_df_out['price']=y_test.values
            train_df_out.to_csv(self.config.transformed_train_csv_path,index=False)
            test_df_out.to_csv(self.config.transformed_test_csv_path,index=False)

            #save the preprocessor
            preprocessor={
                'numeric_pipeline':numeric_pipeline,
                'numeric_cols':numeric_cols
            }
            self.utilis.save_object(self.config.transformed_object_file_path,preprocessor)

            return (
                self.config.transformed_train_file_path,
                self.config.transformed_test_file_path,
                self.config.transformed_object_file_path
            )


        except Exception as e:
            logging.info(f"Error occured during data transformation str{e}")
            raise CustomException(e,sys)
