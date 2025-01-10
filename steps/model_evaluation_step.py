from src.model_evaluator import ModelEvaluation
import xgboost as xgb
import pandas as pd
from zenml import step
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def model_evaluation_step(model: xgb.XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series):
    """Step to evaluate the model."""
    
    # Initialize the model evaluator
    model_evaluator = ModelEvaluation()
    
    # Evaluate the model
    eval_result = model_evaluator.evaluate(model, X_test, y_test)
    mlflow.log_metric("accuracy", eval_result["accuracy"])
    mlflow.log_metric("f1_score", eval_result["f1_score"])
    return eval_result