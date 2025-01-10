import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.base import ClassifierMixin

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract class for Model Selection Strategy
# -------------------------------------------
# This class defines a common interface for different model selection strategies.
class ModelSelectionStrategy(ABC):
    """
    Abstract base class for model selection strategies.
    """
    
    @abstractmethod
    def model_selection(self, models, X_train, y_train, X_test, y_test, metric):
        """
        Abstract method for model selection.
        
        Parameters:
        models (dict): Dictionary of models to be evaluated.
        X_train (array-like): Training features.
        y_train (array-like): Training target.
        X_valid (array-like): Validation features.
        y_valid (array-like): Validation target.
        metric (str): Evaluation metric.
        """
        pass
    
class ModelSelection(ModelSelectionStrategy):
    """
    Model selection strategy using cross-validation and hyperparameter tuning.
    """
    
    def __init__(self):
        self.model_performance = {}

    def hyperparameter_tuning(self, models, param_grid, X_train, y_train, X_test, y_test, metric):
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Parameters:
        models (dict): Dictionary of models to be evaluated.
        param_grid (dict): Dictionary of hyperparameter grids.
        X_train (array-like): Training features.
        y_train (array-like): Training target.
        X_test (array-like): Test features.
        y_test (array-like): Test target.
        metric (str): Evaluation metric.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        for name, model in models.items():
            tuner = GridSearchCV(model, param_grid[name], cv=5, scoring=metric)
            tuner.fit(X_train, y_train)

            # Get the best-performing model and its accuracy
            best_model = tuner.best_estimator_
            best_accuracy = tuner.best_score_
            best_params = tuner.best_params_
            
            y_pred = best_model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            print(f'{name}: Test accuracy = {test_accuracy:.2f}')
            
            # Store the best-performing model and its accuracy
            self.model_performance[name] = (test_accuracy, best_params)
            
    def model_selection(self):
        """
        Select the best-performing model based on the evaluation metric.
        
        Returns:
        best_model: The best-performing model.
        """
        best_model_name = max(self.model_performance, key=lambda x: self.model_performance[x][0])
        best_accuracy, best_params = self.model_performance[best_model_name]
        logging.info(f'\nBest-performing model: {best_model_name} with accuracy = {best_accuracy:.2f}')
        logging.info(f'Best hyperparameters: {best_params}')
        
        return best_model_name, best_accuracy, best_params
        
if __name__ == "__main__":
    pass