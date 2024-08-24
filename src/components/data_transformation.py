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
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self) -> None:
        self.data_tranformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        """
            This function tranforms data based on the different type of data
        """
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
                    ('imputer',SimpleImputer(strategy="most_frequent")), # fill missing values
                    ('one_hot_ecoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info('Numerical columns standard scaling completed.')
            logging.info('Categorical columns encoding completed.')

            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,num_features),
                    ('cat_pipeline',cat_pipeline,cat_features)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed.")

            logging.info("Obtaining preprocessing object.")

            preprocessing_obj = self.get_data_transformer_object()
            target_feature_name="math score"
            num_features = ['writing score', 'reading score']

            input_feature_train_df = train_df.drop(columns=target_feature_name,axis=1)
            target_feature_train_df = train_df[target_feature_name]

            input_feature_test_df = test_df.drop(columns=target_feature_name,axis=1)
            target_feature_test_df = test_df[target_feature_name]

            logging.info('Applying preprocessing object to training dataframe and testing dataframe.')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"saved preprocessing object.")
            
          # save tranformation to a pickle file
            save_object(
                file_path=self.data_tranformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_tranformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)



