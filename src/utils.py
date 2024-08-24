import os
import sys
import dill
# import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.logger import logging
from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file:
            dill.dump(obj, file)
        
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models,param) -> dict:
    """
    This function evaluates each  model and return a dictionary of its
    r2 score
    """
    try:
        report={}
        # evaluate each model,parameters from list of models,parameter
        for i in range(len(list(models))):
            model = list(models.values())[i]

            # Train model with hyper-parameter tuning
            param_list=param[list(models.keys())[i]]
            g_search = GridSearchCV(
                estimator=model,
                param_grid=param_list,
                cv=3,
                n_jobs=3,
                # verbose=verbose,
               # refit=refit
            )
            g_search.fit(X_train,y_train)

            
            # Train model
            model.set_params(**g_search.best_params_)
            model.fit(X_train, y_train) 
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Evaluate Train and Test dataset
            train_model_score =r2_score(y_train, y_train_pred)
            test_model_score=r2_score(y_test,y_test_pred)

            # Assign score to each model name
            report[list(models.keys())[i]]=test_model_score

        return report

    except Exception as e:
        raise CustomException(e,sys)
