import pandas as pd
from src.feature_encoding import (
    FeatureEncoding,
    LabelEncoding,
    OneHotEncoding
)
from zenml import step

@step
def feature_encoding_step(
    df: pd.DataFrame, strategy: str = "one_hot", features:list = None
) -> pd.DataFrame:
    """ Perform feature encoding on the dataframe """
        # Ensure features is a list, even if not provided
    if features is None:
        features = []  # or raise an error if features are required


    if strategy == "onehot_encoding":
        engineer = FeatureEncoding(OneHotEncoding(features))
    elif strategy == "label_encoding":
        engineer = FeatureEncoding(LabelEncoding(features))
    else:
        raise ValueError(f"Unsupported feature Encoding strategy: {strategy}")

    transformed_df = engineer.apply_feature_engineering(df)
    return transformed_df
