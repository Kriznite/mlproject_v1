import sys
import os
import pandas as pd

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self) -> None:
        pass
    
    def predict(self,features):
        """ 
        Apply preprocessor and model pipeline on the user data
        """
        try:
            # load model and preprocessor object from pickle files
            model_path=os.path.join('artifacts/model.pkl')
            preprocessor_path=os.path.join('artifacts/preprocessor.pkl')
            print("loading ....... ")
            model = load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("loading done finished")
            scaled_data = preprocessor.transform(features)
            predicted = model.predict(scaled_data)
            return predicted

        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    """ 
    Collect user data for prediction
    """
    def __init__( self,
        gender:str,
        race_ethnicity:str,
        lunch:str,
        parental_level_of_education:str,
        test_preparation_course:str,
        reading_score:int,
        writing_score:int
    ):
        self.gender = gender
        self.race_ethnicity=race_ethnicity
        self.lunch=lunch
        self.parental_level_of_education = parental_level_of_education
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        """ Get user input as a data frame"""
        try:
            custom_data_input_dict = {
            "gender": [self.gender],
            "race/ethnicity": [self.race_ethnicity],
            "parental level of education": [self.parental_level_of_education],
            "lunch": [self.lunch],
            "test preparation course": [self.test_preparation_course],
            "reading score": [self.reading_score],
            "writing score": [self.writing_score],
        }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)