import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder,StandardScaler,OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

@dataclass
class data_transform_config:
    preprocessor_ob_file_path: str = os.path.join('artifacts','preprocessor.pkl')

class data_transformation:
    def __init__(self):
        self.transform_config = data_transform_config()
        
    def get_transformation_object(self):
        try:
            logging.info("Preparing for preprcoessor...")
            
            numerical_column = ['Rating','Size','same_location','company_age','total_competitors','avg_revenue']
            categorical_column = ['Company Name','Industry','Sector','city','state','simplified_job_title','seniority','ownership_type']
            
            num_pipeline = Pipeline([
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
            ])
            
            logging.info("Num pipeline is ready...")
            
            cate_pipeline = Pipeline([
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder())
            ])
            logging.info("Cate pipeline is ready...")
            
            logging.info("num_pipeline and cate_pipeline are ready to use...")
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_column),
                    ("cate_pipeline",cate_pipeline,categorical_column)
                ]
            )
            
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transoformer(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Train and Test data fetched successfully in data transformer...")
            
            preprocessing_obj = self.get_transformation_object()
            
            logging.info("Preprocessing objects are set...")
            
            target_column = "Salary"
            
            x_train = train_df.drop([target_column],axis=1)
            y_train = train_df[target_column]
            x_test = test_df.drop([target_column],axis=1)
            y_test = test_df[target_column]
            
            logging.info(f"Applying preprocessing object on training dataset and testing dataset...")
            
            # for col in x_train.columns:
            #     print(col, type(x_train[col][0]))
            
            x_arr = preprocessing_obj.fit_transform(x_train)
            y_arr = np.array(y_train)
            x_test_arr = preprocessing_obj.fit_transform(x_test)
            y_test_arr = np.array(y_test)
            
            y_arr = y_arr.reshape((593,1))
            y_test_arr = y_test_arr.reshape((66,1))
            train_arr = np.concatenate((x_arr.toarray(),y_arr),axis=1)
            test_arr = np.concatenate((x_test_arr.toarray(),y_test_arr),axis=1)
           
            logging.info(f"Saved preprocessing object...")
            
            save_object(
                file_path=self.transform_config.preprocessor_ob_file_path,
                obj = preprocessing_obj
            )
            
            return(
                train_arr,
                test_arr,
                self.transform_config.preprocessor_ob_file_path
            )
            
        except Exception as e:
            CustomException(e,sys)
