import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dataclasses import dataclass
from data_transformation import data_transformation
from model_trainer import model_trainer

@dataclass
class data_ingesion_config:
    clean_data_path: str=os.path.join('artifacts','clean_data.csv')
    
class data_ingesion:
    def __init__(self):
        self.ingesion_config = data_ingesion_config()
        
    def initiate_data_ingesion(self):
        logging.info('Entered into data ingesion process...')
        
        try:
            df = pd.read_csv(self.ingesion_config.clean_data_path)
            logging.info('Data fetched successfully...')
            
            # train_set,test_set = train_test_split(df,test_size=0.1,random_state=42)
            # logging.info('Train-Test split completed...')
            
            # train_set.to_csv(self.ingesion_config.train_data_path, header=True, index=False)
            # test_set.to_csv(self.ingesion_config.test_data_path, header=True, index=False)
            logging.info('Ingesion of the data is completed...')
            
            return(
                self.ingesion_config.clean_data_path,
            )
            
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == '__main__':
    obj = data_ingesion()
    clean_data = obj.initiate_data_ingesion()
    data_transformationx = data_transformation()
    train_arr,test_arr,preprocessor_path = data_transformationx.initiate_data_transoformer(clean_data)
    model_trainer = model_trainer()
    model_trainer.initiate_model_trainer(train_arr,test_arr,preprocessor_path)
    
    