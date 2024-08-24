import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join(
        'artifacts',"model.pkl"
    )
class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):

        try:
            logging.info("split training and testing data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models ={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # hyper  parameter tuning  
            # default parameters
            # params ={
            #     "Random Forest": {
            #         "n_estimators": [100, 200, 300],
            #         "max_depth": [10, 20, 30],
            #         "min_samples_split": [2, 5, 10],
            #         "min_samples_leaf": [1, 2, 4],
            #         "bootstrap": [True, False]
            #     },
            #     "Decision Tree": {
            #         "criterion": ["squared_error", "friedman_mse"],
            #         "splitter": ["best", "random"],
            #         "max_depth": [None, 10, 20, 30],
            #         "min_samples_split": [2, 5, 10],
            #         "min_samples_leaf": [1, 2, 4]
            #     },
            #     "Gradient Boosting": {
            #         "loss": ["squared_error","huber","absolute_error", "quantile"],
            #         "n_estimators": [100, 200, 300],
            #         "learning_rate": [0.01, 0.1, 0.05],
            #         "max_depth": [3, 5, 7],
            #         "max features":['auto','sqrt','log2'],
            #         "subsample": [0.8, 0.9, 1.0],
            #         "min_samples_split": [2, 5, 10]
            #     },
            #     "Linear Regression": {
            #         # Linear Regression usually has fewer hyperparameters; here we could just try normalizing the data.
            #         "normalize": [True, False]
            #     },
            #     "XGBRegressor": {
            #         "n_estimators": [100, 200, 300],
            #         "learning_rate": [0.01, 0.1, 0.05],
            #         "max_depth": [3, 5, 7],
            #         "subsample": [0.8, 0.9, 1.0],
            #         "colsample_bytree": [0.8, 0.9, 1.0]
            #     },
            #     "CatBoosting Regressor": {
            #         "iterations": [500, 1000],
            #         "learning_rate": [0.01, 0.1, 0.05],
            #         "depth": [4, 6, 8],
            #         "l2_leaf_reg": [1, 3, 5]
            #     },
            #     "AdaBoost Regressor": {
            #         "n_estimators": [50, 100, 200],
            #         "learning_rate": [0.01, 0.1, 0.05],
            #         "loss": ["linear", "square", "exponential"]
            #     }
            # }

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict=evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
                )
            
            ## To get best model score and its name from dict of models scores
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            # set threshold for best model
            if best_model_score < 0.7:
                raise CustomException("No best model found")
            logging.info(f"Best modelfound for both training and testing dataset")

            # Save best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted)

            return r2_square
        
        except Exception as e:
            raise CustomException(e,sys)
