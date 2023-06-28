# Salary Prediction Model

---

## Introduction

* In this Project, We will explore the development and Implementation of a salary prediction model for job positions in the fields specifically related to Data Science and Machine Learning. The primary goal of this project is to build a reliable model that accurately predicts salaries based on various factors related to job roles, skills, experience, and other relavent features.

## Overview

* Nowadays, the data science related jobs have witnessed significant growth in recent years, leading to an incresed demand for professionals with expertise in these domains. As organizarions strive to stay competitive in the data-driven era, attracting and retaining top telent in these fields has become crucial. Consequently, both job seekers and employers often face challenges in negotiating salaries and understanding the factors that contribute to compensation.

* The purpose of this salary prediction model is to provide insights into the expected salary ranges for specific job positions, thereby assisting job seekers in making informed decisions and helping employers set appropriate compensation packages. By leveraging historical data and machine learning techniques, we aim to build a model that can accurately estimate salaries based on relevant features.

## Project workflow

1. Data Analysis of related Jobs
2. Development of Predictive Model
3. Training and Testing of Model

## Business Requirements

1. Informed Salary Negotiations
2. Competitive Compensation Packages
3. Cost Optimization
4. Talent Acquisition annd Retention
5. Market Insights
6. Streamlined HR Processes

## Proposed Solution

* My solution focuses on, How diiferent machine learning techniques can be used to meet the defined business requirements.

## Approach

Approach to this project follows a structured, step-by-step methodology grounded in data science best practices. Each stage is thoughtfully designed to build upon the previous, ensuring a cohesive and comprehensive solution.

1. Understaing the Data
2. Data Preprocessing/cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Selection
5. Model Development
6. Model training and testing
7. Model Evaluation and Optimization

## Data Preprocessing

1. Data Importing/Collection
2. Data Understanding (Attributes)
3. Data Inspection
4. Data Cleaning & Feature Engineering

### Data Importing/Collection

[Kaggle Dataset Link](https://www.kaggle.com/datasets/rashikrahmanpritom/data-science-job-posting-on-glassdoor)

* Dataset was loaded using pandas framework.
* The dataset contains total 15 different features and 672 total entries.

> **Sample Dataset**

|index|Job Title                                                                                       |Salary Estimate             |Job Description |Rating|Company Name                                           |Location                  |Headquarters             |Size                   |Founded|Type of ownership             |Industry                                |Sector                            |Revenue                         |Competitors                                                                                 |
|-----|------------------------------------------------------------------------------------------------|----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|-------------------------------------------------------|--------------------------|-------------------------|-----------------------|-------|------------------------------|----------------------------------------|----------------------------------|--------------------------------|--------------------------------------------------------------------------------------------|
|0    |Sr Data Scientist                                                                               |$137K-$171K (Glassdoor est.)|Description  The Senior Data Scientist is responsible for defining, building, and improving statistical models.  |3.1   |Healthfirst 3.1                                        |New York, NY              |New York, NY             |1001 to 5000 employees |1993   |Nonprofit Organization        |Insurance Carriers                      |Insurance                         |Unknown / Non-Applicable        |EmblemHealth, UnitedHealth Group, Aetna                                                     |
|1    |Data Scientist                                                                                  |$137K-$171K (Glassdoor est.)|Secure our Nation, Ignite your Future  Join the top Information Technology and Analytic professionals in the industry.  |4.2   |ManTech 4.2                                            |Chantilly, VA             |Herndon, VA              |5001 to 10000 employees|1968   |Company - Public              |Research & Development                  |Business Services                 |$1 to $2 billion (USD)          |-1                                                                                          |
|2    |Data Scientist                                                                                  |$137K-$171K (Glassdoor est.)|Overview   Analysis Group is one of the largest international economics consulting firms, with more than 1,000 professionals across 14 offices. |3.8   |Analysis Group 3.8                                     |Boston, MA                |Boston, MA               |1001 to 5000 employees |1981   |Private Practice / Firm       |Consulting                              |Business Services                 |$100 to $500 million (USD)      |-1                                                                                          |

### Data Understanding (Attributes)

As there are total 15 different Features. Here is the detail about the features.

1. index : Number of the row
2. Job Title : Represent the field/area of the job
3. Salary Estimate : Avg. salary per hour in $
4. Job Description : Describe the job role and day-to-day responsibility
5. Rating : rating of the company from their employees
6. Company Name : Name of the company
7. Location : Location of the job
8. Headquarters : Location of company's headquarters
9. Size : Number of employees working in the company
10. Founded : The year in which company was started/founded
11. Type of ownership : Represents the who is the main stackholder in the company
12. Industry : Industry of the Job
13. Sector : Sector of the Job
14. Revenue : Company's yearly total revenue
15. Competitors : Company who has similar kind of jobs or products in the market

### Data Inspection

* From the 15 features, 2 features have int64 data type, 1 feature has float64, and rest all other 12 features are in object datatype.
* There are total 672 entries and not a single entry contains null value.
* May be there is chance of duplicate entries.

### Data Cleaning & Feature Engineering

This process has been done in total 16 following steps.

1. Drop the index column
2. Remove the entry if Salary Estimation data is not available
3. Finding avg salary per hour in $
4. Droping Job Description column
5. Creating new column to check whether the job is at headquarter or not
6. Spliting location column into city and state
7. Calculating the average size of the comapany
8. Finding the company age
9. Simplifying types of ownership column
10. Calculating total competitors of the company
11. Simplifying Job Title and assigning seniority
12. Removing extra elements from company name
13. Simplifying ownership_type
14. Cleaning the revenue column and calculating average revenue of the company
15. Droping the duplicate entries
16. Removing unnecessary features

* After data cleaning and feature engineering process, we have total 15 features and 659 distinct entries.

> **Sample of Clean Dataset**

|Rating|Company Name                                       |Size  |Industry                                |Sector                            |same_location|city               |state        |company_age|total_competitors|simplified_job_title     |seniority|ownership_type                |avg_revenue|Salary|
|------|---------------------------------------------------|------|----------------------------------------|----------------------------------|-------------|-------------------|-------------|-----------|-----------------|-------------------------|---------|------------------------------|-----------|------|
|3.1   |Healthfirst                                        |3000.5|Insurance Carriers                      |Insurance                         |1            |New York           | NY          |30         |3                |data scientist           |senior   |nonprofit                     |-1.0       |154.0 |
|4.2   |ManTech                                            |7500.5|Research & Development                  |Business Services                 |0            |Chantilly          | VA          |55         |0                |data scientist           |na       |public                        |1500.0     |154.0 |
|3.8   |Analysis Group                                     |3000.5|Consulting                              |Business Services                 |1            |Boston             | MA          |42         |0                |data scientist           |na       |private                       |300.0      |154.0 |

## Data Transformation

1. Pipeline setup for numerical and categorical data
2. Train and Test split