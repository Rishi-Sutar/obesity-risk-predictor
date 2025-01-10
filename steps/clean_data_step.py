import pandas as pd
from src.clean_data import (
    DropColumns
)
from zenml import step

@step
def clean_data_step(
    df: pd.DataFrame, features:list
) -> pd.DataFrame:
    """" Perform data cleaning on the dataframe """
    
    # Ensure features is a list, even if not provided
    if features is None:
        features = []  # or raise an error if features are required
        
    clean = DropColumns()
    data_clean = clean.handle_data(df, features)
    return data_clean