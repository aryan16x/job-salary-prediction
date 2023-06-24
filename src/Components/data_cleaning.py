'''
Data Cleaning Process
    1. Remove Index
    2. About job title
    3. Parse job salary/Convert into median salary
    4. Remove Job Description
    5. Check if job location is headquarter or not
    6. Split location into city and state
    7. Avg company size
    8. Company Age
    9. Clean the type of ownership column
    10. Avg Revenue of the company
    11. Count competitors if any
    12. also check null values in any step
'''

import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

@dataclass
class data_cleaning_config:
    raw_data_path: str=os.path.join('artifacts','raw_data.csv')
    clean_data_path: str=os.path.join('artifacts','clean_data.csv')
    
class data_cleaning:
    def __init__(self):
        self.cleaning_config = data_cleaning_config()
        
    def job_title(self,title):
        title = title.lower()
        if ('data scientist' in title) or ('data science' in title):
            return 'data scientist'
        elif 'data engineer' in title:
            return 'data engineer'
        elif ('data analyst' in title) or ('data analysis' in title) or ('analyst' in title) or ('analytics' in title):
            return 'data analyst'
        elif 'machine learning' in title:
            return 'machine learning engineer'
        elif 'scientist' in title:
            return 'scientist'
        elif 'engineer' in title:
            return 'engineer'
        else :
            return 'other'

    def seniority(self,title):
        title = title.lower()
        if ('senior' in title) or ('sr' in title):
            return 'senior'
        elif ('junior' in title) or ('jr' in title):
            return 'junior'
        else:
            return 'na'
        
    def ownership_simplify(self,type):
        type = type.lower()
        if ('unknown' in type) or ('-1' in type) or ('other' in type):
            return 'na'
        elif ('private' in type) or ('self-employed' in type) or ('contract' in type):
            return 'private'
        elif ('nonprofit' in type) or ('college' in type) or ('hospital' in type):
            return 'nonprofit'
        else:
            return type
        
    def billion_to_million(self,rev):
        if 'billion' in rev:
            return 1
        return 0
        
    def initiate_data_clening(self):
        logging.info('Data cleaning process is started...')
        
        try:
            df = pd.read_csv(self.cleaning_config.raw_data_path)
            logging.info('Data reading is successful...')
            
            dfx = df
            
            # 1st Work
            df = df.drop('index', axis=1)
            
            #2nd Work
            
            #3rd Work
            df = df[df['Salary Estimate']!='-1'] 
            
            salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
            salary = salary.apply(lambda x: x.lower().replace('$','').replace('k',''))
            df['min_salary'] = salary.apply(lambda x: int(x.split('-')[0]))
            df['max_salary'] = salary.apply(lambda x: int(x.split('-')[1]))
            avg_salary = df[['min_salary', 'max_salary']].mean(axis=1)
            df['Salary Estimate'] = avg_salary        
            
            #4th Work
            df = df.drop('Job Description', axis=1)
            
            #5th Work
            df['same_location'] = df.apply(lambda x: 1 if x['Location']==x['Headquarters'] else 0, axis=1)
            
            #6th Work
            df['city'] = df['Location'].apply(lambda x: x.split(',')[0])
            df['state'] = df['Location'].apply(lambda x: x.split(',')[1] if len(x.split(','))>1 else '')
            
            #7th Work
            company_size = df['Size'].apply(lambda x: x.replace('employees','').replace('+','').replace('unknown','-1'))
            df['min_size'] = company_size.apply(lambda x: -1 if (x==-1 or x.lower()=='unknown') else int(x.split('to')[0]))
            df['max_size'] = company_size.apply(lambda x: x if x==-1 else (int(x.split('to')[1]) if len(x.split('to'))>1 else (10000 if x==10000 else -1)))
            avg_size = df[['min_size','max_size']].mean(axis=1)
            df['Size'] = avg_size
            
            #8th Work
            df['company_age'] = df['Founded'].apply(lambda x: -1 if x==-1 else 2023-x)
            
            #9th Work
            df['Type of ownership'] = df['Type of ownership'].apply(lambda x: x.replace('Company - ',''))
            
            #10th Work
            
            #11th Work
            df['total_competitors'] = df['Competitors'].apply(lambda x: 0 if (x=='-1') else len(x.split(',')))  
            
            df['simplified_job_title'] = df['Job Title'].apply(lambda x: self.job_title(x))
            df['seniority'] = df['Job Title'].apply(self.seniority)
            
            df['Company Name'] = df.apply(lambda x: x['Company Name'] if x['Rating']<0 else x['Company Name'][:-3], axis=1)
            df['Company Name'] = df['Company Name'].apply(lambda x: x.replace('\n',''))
            
            df['ownership_type'] = df['Type of ownership'].apply(self.ownership_simplify)
            
            df['avg_revenue'] = df['Revenue'].apply(lambda x: '-1' if ('unknown' in x.lower()) else (x.replace('(USD)','').replace('$','')))
            df['min_revenue'] = df['avg_revenue'].apply(lambda x: x.split('to')[0] if x!=-1 else '-1')
            df['min_revenue'] = df['min_revenue'].apply(lambda x: x.replace('+ billion','').replace('million','').replace('Less than',''))
            df['max_revenue'] = df['avg_revenue'].apply(lambda x: (x.split('to')[1]).replace('million','').replace('billion','') if (x!=-1 and len(x.split('to'))>1) else '-1')
            
            df['tf'] = df['avg_revenue'].apply(self.billion_to_million)

            df['min_revenue'] = df.apply(lambda x: int(x.min_revenue)*1000 if (x.tf) else int(x.min_revenue), axis=1)
            df['max_revenue'] = df.apply(lambda x: int(x.max_revenue)*1000 if (x.tf) else int(x.max_revenue), axis=1)
            
            df['avg_revenue'] = df[['min_revenue','max_revenue']].mean(axis=1)
            
            df = df.drop_duplicates()
            
            df['Salary'] = df['Salary Estimate']
            df = df.drop(['Job Title','Type of ownership', 'Salary Estimate','min_revenue','max_revenue','tf','Revenue','min_salary','max_salary','min_size','max_size','Founded','Competitors','Location','Headquarters'], axis=1)          
            
            df.to_csv(self.cleaning_config.clean_data_path, index=False, header=True)
            logging.info('Clean_data file is saved successfully...')
            
            return (
                self.cleaning_config.clean_data_path
            )
            
        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == '__main__':
    obj = data_cleaning()
    clean_data = obj.initiate_data_clening()