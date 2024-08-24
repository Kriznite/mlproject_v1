import sys
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self) -> None:
        self.data_tranformation_config=DataTransformationConfig()

        def get_data_transformer_object(self):
            try:
                # Create Column Transformer with 3 types of transformers
                num_features = ['writing score', 'reading score']
                cat_features = [
                    "gender",
                    "race/ethnicity",
                    "parental level of education",
                    "lunch",
                    "test preparation course",
                ]
                num_pipeline = Pipeline(
                    steps=[
                        ('imputer',SimpleImputer(strategy='median')), # fill missing values
                        ('scaler', StandardScaler()) # encode data
                    ]
                )
                cat_pipeline = Pipeline(
                    steps=[
                        ('imputer',SimpleImputer(strategy="most_frequent")) # fill missing values
                        ('one_hot_ecoder',OneHotEncoder()),
                        ('scaler',StandardScaler())
                    ]
                )

                logging.info('Numerical columns standard scaling completed')
                logging.info('Categorical columns encoding completed')

                preprocessor=ColumnTransformer(
                    [
                        ('num_pipeline',num_pipeline,num_features)
                        ('cat_pipeline',cat_pipeline,cat_features)
                    ]
                )
            except:
                pass


