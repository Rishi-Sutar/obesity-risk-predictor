from src.model_selection import ModelSelection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from zenml import step
import pandas as pd

models = {
    "LR": LogisticRegression(),
    "DT": DecisionTreeClassifier(),
    "RF": RandomForestClassifier(),
    "XGB": xgb.XGBClassifier()
}

param_grid = {
    "LR": {'C': [0.1, 1, 10]},
        "DT": {'max_depth': [3, 5, 7]},
        "RF": {'n_estimators': [50, 100, 200]},
        "XGB": {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
}

@step
def model_selection_step(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, metric: str = "accuracy"
):
    """Perform model selection using the specified strategy."""
    selector = ModelSelection()
    selector.hyperparameter_tuning(models, param_grid, X_train.values, y_train.values, X_test.values, y_test.values, metric)
    best_model, best_acc, best_params = selector.model_selection()
    return {"best_model": best_model, "best_acc": best_acc, "best_params": best_params}