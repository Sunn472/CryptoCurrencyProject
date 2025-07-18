import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException
import numpy as np

def run_training_pipeline():
    try:
        logging.info("Training pipeline started.")
        
        # Step 1: Data Ingestion
        ingestion = DataIngestion()
        ingestion.initiate_data_ingestion()

        # Step 2: Data Transformation
        transformer = DataTransformation()
        train_path, test_path, preprocessor_path = transformer.initiate_data_transformation()

        # Step 3: Load transformed data
        train_data = np.load(train_path)
        test_data = np.load(test_path)
        X_train, y_train = train_data['X'], train_data['Y']
        X_test, y_test = test_data['X'], test_data['Y']

        # Step 4: Model Training
        trainer = ModelTrainer()
        model_path = trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)

        logging.info("Training pipeline completed successfully.")
        return model_path

    except Exception as e:
        logging.error(f"Error in training pipeline: {e}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    run_training_pipeline()
