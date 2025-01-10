import logging 
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Feature Engineering Strategy
# ----------------------------------------------------
# This class defines a common interface for different cleaning data strategies.
# Subclasses must implement the handle_data method.
class CleanDataStrategy(ABC):
    @abstractmethod
    def handle_data(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        """
        Abstract method to handle data cleaning.
        
        Parameters:
        df (pd.DataFrame): The input data to be cleaned.
        features (list): The list of features to be cleaned.
        
        Returns:
        pd.DataFrame: The cleaned data.
        """
        pass
    
# Concrete Strategy: Drop unwanted columns
# ----------------------------------------
# This strategy drops unwanted columns from the df.
class DropColumns(CleanDataStrategy):
    def handle_data(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        """
        Drop unwanted columns from the data.
        
        Parameters:
        df (pd.DataFrame): The input data to be cleaned.
        features (list): The list of features to be dropped.
        
        Returns:
        pd.DataFrame: The cleaned data.
        """
        data = df.copy()
        logging.info(f"Dropping columns: {features}")
        return data.drop(columns=features)
    
class DataCleaning:
    """
    Data cleaning class which preprocesses the data and divides it into train and test data.
    """

    def cleaning_data() -> CleanDataStrategy:
        """Handle data based on the provided strategy"""
        return DropColumns()
    
if __name__ == "__main__":
    pass