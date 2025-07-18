import numpy as np
import pandas as pd
import sys
import os

from src.utils.main_utils import MainUtils
from src.exception import CustomException
from src.constant import TEST_PATH
from src.logger import logging

def run_prediction_pipeline():
    try:
        logging.info("Prediction pipeline started.")
        
        utils = MainUtils()
        model = utils.load_object("artifacts/model.pkl")
        preprocessor = utils.load_object("artifacts/preperocessor.pkl")

        # Load raw test data
        df = pd.read_csv(TEST_PATH)
        df['Liquidity Ratio'] = df['24h_volume'] / df['mkt_cap']
        df['volatility_index'] = (df['1h'] + df['7d'] + df['24h']) / 3
        df['Volatility'] = df['24h_volume'] / (df['mkt_cap'] * df['volatility_index'])
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.drop(columns=['volatility_index'], axis=1, inplace=True)

        df['date'] = pd.to_datetime(df['date'])
        X = df.drop(columns=['price', 'coin', 'symbol', '24h_volume', 'mkt_cap'])
        y = df['price']

        X_processed = preprocessor['numeric_pipeline'].transform(X[preprocessor['numeric_cols']])
        predictions = model.predict(X_processed)

        logging.info("Prediction pipeline completed.")
        return pd.DataFrame({'Actual': y.values, 'Predicted': predictions})

    except Exception as e:
        logging.error(f"Error during prediction pipeline: {e}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    results_df = run_prediction_pipeline()
    print(results_df.head())
