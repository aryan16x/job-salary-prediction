import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class data_ingesion_config:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    clean_data_path: str=os.path.join('artifacts','clean_data.csv')
    
class data_ingesion:
    def __init__(self):
        self.ingesion_config = data_ingesion_config()
        
    def initiate_data_ingesion(self):
        logging.info('Entered into data ingesion process...')
        
        try:
            df = pd.read_csv(self.ingesion_config.clean_data_path)
            logging.info('Data fetched successfully...')
            
            train_set,test_set = train_test_split(df,test_size=0.1,random_state=42)
            logging.info('Train-Test split completed...')
            
            train_set.to_csv(self.ingesion_config.train_data_path, header=True, index=False)
            test_set.to_csv(self.ingesion_config.test_data_path, header=True, index=False)
            logging.info('Ingesion of the data is completed...')
            
            return(
                self.ingesion_config.train_data_path,
                self.ingesion_config.test_data_path,
            )
            
        except Exception as e:
            raise CustomException(e,sys)
        
        