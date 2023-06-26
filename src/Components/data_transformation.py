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
from sklearn.model_selection import train_test_split

@dataclass
class data_transform_config:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
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
        
    def initiate_data_transoformer(self,clean_data_path):
        try:
            try:
                # print(clean_data_path)
                df = pd.read_csv("artifacts\clean_data.csv")
            except Exception as e:
                raise CustomException(e,sys)
    
            logging.info("Clean data fetched successfully in data transformer...")
            
            preprocessing_obj = self.get_transformation_object()
            
            logging.info("Preprocessing objects are set...")
            
            target_column = "Salary"
            
            x_df = df.drop([target_column],axis=1)
            y_df = df[target_column]
            # x_test = test_df.drop([target_column],axis=1)
            # y_test = test_df[target_column]
            
            x_arr = preprocessing_obj.fit_transform(x_df)
            y_arr = np.array(y_df)
            # x_test_arr = preprocessing_obj.fit_transform(x_test)
            # y_test_arr = np.array(y_test)
            
            y_arr = y_arr.reshape((659,1))
            # y_test_arr = y_test_arr.reshape((66,1))
            df_arr = np.concatenate((x_arr.toarray(),y_arr),axis=1)
            # test_arr = np.concatenate((x_test_arr.toarray(),y_test_arr),axis=1)
            
            train_df,test_df = train_test_split(df_arr,test_size=0.1,random_state=42)
            logging.info('Train-Test split completed...')
            
            train_df = pd.DataFrame(train_df)
            test_df = pd.DataFrame(test_df)
            
            train_df.to_csv(self.transform_config.train_data_path, header=True, index=False)
            test_df.to_csv(self.transform_config.test_data_path, header=True, index=False)
            logging.info('Train and Test files successfully created...')
            
            logging.info(f"Applying preprocessing object on training dataset and testing dataset...")
            
            # for col in x_train.columns:
            #     print(col, type(x_train[col][0]))
           
            logging.info(f"Saved preprocessing object...")
            
            save_object(
                file_path=self.transform_config.preprocessor_ob_file_path,
                obj = preprocessing_obj
            )
            
            train_arr = train_df.values
            test_arr = test_df.values
            
            return(
                train_arr,
                test_arr,
                self.transform_config.preprocessor_ob_file_path
            )
            
        except Exception as e:
            CustomException(e,sys)
