from src.model_trainer import (
    ModelTrainer,
    LogisticRegressionTraining,
    DecisionTreeTraining,
    RandomForestTraining,
    xgbClassifierTraining
)

import mlflow
import os 
import pandas as pd
from zenml import step
import pickle
from zenml.client import Client
from azureml.core import Workspace, Model


experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def model_trainer_step(
    X_train: pd.DataFrame, y_train: pd.Series, model_details: dict    
):  
    model_name = model_details['best_model']
    params = model_details['best_params']
    
    if model_name == 'LR':
        mlflow.sklearn.autolog()
        model = ModelTrainer(LogisticRegressionTraining())
    elif model_name == 'DT':
        mlflow.sklearn.autolog()
        model = ModelTrainer(DecisionTreeTraining())
    elif model_name == 'RF':
        mlflow.sklearn.autolog()
        model = ModelTrainer(RandomForestTraining())
    else:
        mlflow.xgboost.autolog()
        model = ModelTrainer(xgbClassifierTraining())
                
    trained_model = model.train_model(X_train, y_train, params)
    
    # Save the trained model
    model_filename = "trained_model.pkl"
    model_path = f"saved_models/{model_filename}"
    # Create the 'saved_models' folder if it doesn't exist
    os.makedirs('saved_models', exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(trained_model, f)  # 'model' is the variable containing your trained model
    print(f"Model saved as {model_filename}")
    
    return trained_model