import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble  import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class model_trainer_config:
    trained_model_path = os.path.join('artifacts','model.pkl')
    
class model_trainer:
    def __init__(self):
        self.trainer_config = model_trainer_config()
        
    def initiate_model_trainer(self,train_arr,test_arr,preprocessor_path):
        try:
            logging.info("Splitting features and y values...")
            x_train,y_train,x_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1],
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Linear Regression": LinearRegression(),
                "K Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "XGB Regressor": XGBRegressor(),
                "Ada Boost Regressor": AdaBoostRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "Lasso Regression": Lasso(),
                "Ridge Regression": Ridge(),
                "Cat Boost Regressor": CatBoostRegressor(),
            }
            model_report:dict = evaluate_models(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models
            )
            
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            logging.info('Best model found...')
            
            print(best_model_name," : ",best_model_score)
            
            save_object(
                file_path = self.trainer_config.trained_model_path,
                obj = best_model
            )
            
        except Exception as e:
            raise CustomException(e,sys)
