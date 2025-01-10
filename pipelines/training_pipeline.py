from steps.data_ingestion_step import data_ingestion_step
from steps.clean_data_step import clean_data_step
from steps.feature_encoding_step import feature_encoding_step
from steps.feature_scaling_step import feature_scaling_step
from steps.data_splitter_step import data_splitter_step
from steps.model_selection_step import model_selection_step
from steps.model_trainer_step import model_trainer_step
from steps.model_evaluation_step import model_evaluation_step
from zenml import Model, pipeline, step
from zenml.client import Client
@pipeline(
    model=Model(
        # The name uniquely identifies this model
        name="obesity_risk_predictor"
    ),
    enable_cache=False
)
def ml_pipeline():    
    """Define an end-to-end machine learning pipeline."""
    
    # Data ingestion step
    raw_data = data_ingestion_step(
        file_path="data/obesity_risk.zip"
    )
    
    # No missing data present in the data provided
    
    # Data cleaning step
    clean_data = clean_data_step(
        raw_data, features=["id", "Unnamed: 0"]
    )
    
    # # Feature onehot encoding step
    # encoded_data = feature_encoding_step(
    #     clean_data, strategy="onehot_encoding", features=["CAEC", "CALC", "MTRANS"]
    # )
    
    # Feature label encoding step
    encoded_data = feature_encoding_step(
        clean_data, strategy="label_encoding", features=['Gender', 'CAEC', 'CALC', 'MTRANS', 'Obesity_Level']
    )
    
    # Log transformation step
    # log_transformed_data = feature_scaling_step(
    #     encoded_data, strategy="log", features=["Age", "family_history_with_overweight", 
    #                                             "FAVC", "NCP", "SMOKE", "SCC"]
    # )
    
    # Standard scaling step
    scaled_data = feature_scaling_step(
        encoded_data, strategy="standard_scaling", features=[
            'Age', 'Height', 'Weight', 'family_history_with_overweight',
            'FAVC', 'FCVC', 'NCP', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',]
    )
    
    # Data splitting step
    X_train, X_test, y_train, y_test = data_splitter_step(
        scaled_data, target_column="Obesity_Level"
    )
    
    # Model selection step
    model_selection_result = model_selection_step(X_train, y_train, X_test, y_test)
    
    # Train selected model
    trained_model = model_trainer_step(X_train, y_train, model_selection_result)
    
    # Model evaluation step
    model_evaluation_step(trained_model, X_test, y_test)
    
    
    # You can add further steps like model evaluation, saving the model, etc. 

if __name__ == "__main__":
    # Running the pipeline
    ml_pipeline()