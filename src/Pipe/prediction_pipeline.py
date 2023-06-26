import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from dataclasses import dataclass

@dataclass
class prediction_pipeline_config:
    pass

class prediction_pipeline:
    def __init__(self):
        self.prediction_pipe = prediction_pipeline_config()
        
    def initiate_prediction_pipeline(self,features,preprocessor_path,model_path):
        try:
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            print(df)
            data_scaled = preprocessor.fit_transform(features)
            print(data_scaled)
            prediction = model.predict(data_scaled)
            
            return prediction
        
        except Exception as e:
            raise CustomException(e,sys)
        
class custom_data:
    def __init__(self,
                Rating: float,
                Company_Name: str,
                Size: float,
                Industry: str,
                Sector: str,
                same_location: float,
                city: str,
                state: str,
                company_age: float,
                total_competitors: float,
                simplified_job_title: str,
                seniority: str,
                ownership_type: str,
                avg_revenue: float
                ):
        self.Rating = Rating
        self.Company_Name = Company_Name
        self.Size = Size
        self.Industry = Industry
        self.Sector = Sector
        self.same_location = same_location
        self.city = city
        self.state = state
        self.company_age = company_age
        self.total_competitors = total_competitors
        self.simplified_job_title = simplified_job_title
        self.seniority = seniority
        self.ownership_type = ownership_type
        self.avg_revenue = avg_revenue
        
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "Rating": [self.Rating],
                "Company Name": [self.Company_Name],
                "Size": [self.Size],
                "Industry": [self.Industry],
                "Sector": [self.Sector],
                "same_location": [self.same_location],
                "city": [self.city],
                "state": [self.state],
                "company_age": [self.company_age],
                "total_competitors": [self.total_competitors],
                "simplified_job_title": [self.simplified_job_title],
                "seniority": [self.seniority],
                "ownership_type": [self.ownership_type],
                "avg_revenue": [self.avg_revenue]
            }
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == '__main__':
    obj = prediction_pipeline()
    clx = custom_data(
        Rating=4.2,
        Company_Name="Unicom Technologies INC",
        Size=350.5,
        Industry='Lending',
        Sector='Finance',
        same_location=1,
        city='San Mateo',
        state=' CA',
        company_age=11,
        total_competitors=0,
        simplified_job_title='data scientist',
        seniority='na',
        ownership_type='government',
        avg_revenue=-1
    )
    df = clx.get_data_as_dataframe()
    
    pred = obj.initiate_prediction_pipeline(df,"artifacts/preprocessor.pkl","artifacts/model.pkl")
    
    print(pred)