import pandas as pd
from src.feature_scaling import (
    FeatureScaler,
    LogTransformation,
    MinMaxScaling,
    StandardScaling,
)
from zenml import step

@step
def feature_scaling_step(
    df: pd.DataFrame, strategy: str = "log", features: list = None
) -> pd.DataFrame:
    """Performs feature engineering using FeatureEngineer and selected strategy."""

    # Ensure features is a list, even if not provided
    if features is None:
        features = []  # or raise an error if features are required

    if strategy == "log":
        engineer = FeatureScaler(LogTransformation(features))
    elif strategy == "standard_scaling":
        engineer = FeatureScaler(StandardScaling(features))
    elif strategy == "minmax_scaling":
        engineer = FeatureScaler(MinMaxScaling(features))
    else:
        raise ValueError(f"Unsupported feature transforamtion strategy: {strategy}")

    scaled_df = engineer.apply_feature_scaling(df)
    return scaled_df