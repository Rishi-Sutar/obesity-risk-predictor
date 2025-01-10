import logging
import pandas as pd
from abc import ABC, abstractmethod

from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract class for Model training Strategy
# -------------------------------------------
# This class defines a common interface for different model training strategies.
# Subclasses must implement the build_and_train_model method.
class ModelTrainingStrategy(ABC):
    def build_and_train_model(self, X_train, y_train):
        """
        Abstract method for building and training a model.
        
        Parameters:
        X_train (array-like): Training features.
        y_train (array-like): Training target.
        
        Returns:
        ClassifierMixin: The trained model.
        """
        pass
    
# Concrete Strategy for Logistic Regression
class LogisticRegressionTraining(ModelTrainingStrategy):
    def build_and_train_model(
        self, X_train: pd.DataFrame, y_train: pd.Series, params: dict
        ) -> ClassifierMixin:
        """
        Builds and trains a logistic regression model.
        
        Parameters:
        X_train (array-like): Training features.
        y_train (array-like): Training target.
        
        Returns:
        ClassifierMixin: The trained logistic regression model.
        """
        logging.info("Building and training a logistic regression model.")
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        logging.info("Logistic regression model training completed.")
        return model
    
class DecisionTreeTraining(ModelTrainingStrategy):
    def build_and_train_model(
        self, X_train: pd.DataFrame, y_train: pd.Series, params: dict
        ) -> ClassifierMixin:
        """
        Builds and trains a decision tree model.
        
        Parameters:
        X_train (array-like): Training features.
        y_train (array-like): Training target.
        
        Returns:
        ClassifierMixin: The trained decision tree model.
        """
        logging.info("Building and training a decision tree model.")
        model = DecisionTreeClassifier(**params)
        model.fit(X_train, y_train)
        logging.info("Decision tree model training completed.")
        return model
    
class RandomForestTraining(ModelTrainingStrategy):
    def build_and_train_model(
        self, X_train: pd.DataFrame, y_train: pd.Series, params: dict
        ) -> ClassifierMixin:
        """
        Builds and trains a random forest model.
        
        Parameters:
        X_train (array-like): Training features.
        y_train (array-like): Training target.
        
        Returns:
        ClassifierMixin: The trained random forest model.
        """
        logging.info("Building and training a random forest model.")
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        logging.info("Random forest model training completed.")
        return model
    
class xgbClassifierTraining(ModelTrainingStrategy):
    def build_and_train_model(
        self, X_train: pd.DataFrame, y_train: pd.Series, params: dict
        ) -> xgb.XGBClassifier:
        """
        Builds and trains a XGBoost model.
        
        Parameters:
        X_train (array-like): Training features.
        y_train (array-like): Training target.
        
        Returns:
        ClassifierMixin: The trained random forest model.
        """
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        return model

class ModelTrainer:
    def __init__(self, strategy: ModelTrainingStrategy):
        self.strategy = strategy
        
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, params: dict):
        """
        Trains a model using the specified strategy.
        
        Parameters:
        X_train (array-like): Training features.
        y_train (array-like): Training target.
        
        Returns:
        ClassifierMixin: The trained model.
        """
        return self.strategy.build_and_train_model(X_train, y_train, params)

if __name__ == "__main__":
    pass
