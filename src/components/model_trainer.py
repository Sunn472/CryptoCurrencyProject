import os 
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import r2_score,mean_absolute_error,root_mean_squared_error
from sklearn.pipeline import Pipeline
from src.utils.main_utils import MainUtils 

@dataclass
class ModelTrainerConfig:
    artifact_folder=os.path.join("artifacts")
    trained_model_path=os.path.join(artifact_folder,'model.pkl')
    model_config_path: str=os.path.join("config","model.yaml")


class ModelTrainer:
    def __init__(self):
        self.config=ModelTrainerConfig()
        self.utils=MainUtils()
        self.model={
            'LinearRegression':LinearRegression(),
            'XGBRegressor':XGBRegressor(),
            'RandomForestRegressor':RandomForestRegressor(),
            'GradientBoostingRegressor':GradientBoostingRegressor()
        }
        self.model_param_grid=self.utils.read_yaml_file(self.config.model_config_path)['model_selection']['model']

    def evaluate_model(self,X_train,X_test,y_train,y_test):
        logging.info('model evaluation starting --------')
        report={}
        for name,model in self.model.items():
            logging.info("Evaluate Base model--------")
            model.fit(X_train,y_train)
            y_pred=model.predict(X_test)
            Rsq_score=r2_score(y_test,y_pred)
            rmse_score=root_mean_squared_error(y_test,y_pred)
            mae_score=mean_absolute_error(y_test,y_pred)
            report[name]=Rsq_score
            logging.info(f"{name} r2_score: {Rsq_score:.4f}, MAE_score:{mae_score}, RMSE_score:{rmse_score}")
            print(f"{name} r2_score: {Rsq_score:.4f}, MAE_score:{mae_score}, RMSE_score:{rmse_score}")

        logging.info(f"Model Evaluation Report {report}")
        print(f"Model Evaluation Report {report}")
        return report
        
    def fine_tune_best_model(self,model_name,model,X_train,y_train):
        logging.info(f"GridSearchCV for {model_name}............")
        param_grid= self.model_param_grid[model_name]["search_param_grid"]
        grid_cv=GridSearchCV(model,param_grid=param_grid,cv=5,n_jobs=-1,verbose=2)
        grid_cv.fit(X_train,y_train)
        best_params=grid_cv.best_params_
        print(f"Best parameter for model {best_params}")
        logging.info(f"Best params for {model_name}: {best_params}")
        model.set_params(**best_params)
        return model
    
    def initiate_model_trainer(self, X_train,X_test,y_train, y_test):
        logging.info(f"Initiating model training.....")
        try:
            report = self.evaluate_model(X_train, X_test,y_train, y_test)
            best_model_name = max(report, key=report.get)
            logging.info(f"Best model selected: {best_model_name} with accuracy {report}")

            best_model = self.fine_tune_best_model(best_model_name, self.model[best_model_name], X_train, y_train)

            # Save the model
            self.utils.save_object(self.config.trained_model_path, best_model)
            logging.info(f"Trained Model Saved at: {self.config.trained_model_path}")
            return self.config.trained_model_path 
        
        
        except Exception as e:
            logging.info(f"error occured during model traing str{e}")
            raise CustomException(e,sys)

