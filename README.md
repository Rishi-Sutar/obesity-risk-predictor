
# Obesity Risk Predictor

##  Project Overview

The Obesity Risk Predictor is a machine learning pipeline project built using ZenML to predict obesity risk based on various features. The project follows a structured ML workflow, including data preprocessing, model training, and deployment. The model is deployed as a real-time endpoint on Azure Machine Learning Studio and accessed through a Streamlit app.
## Dataset

Dataset Link: https://www.kaggle.com/datasets/jpkochar/obesity-risk-dataset/data

This dataset contains information about individuals' health and lifestyle factors that may contribute to obesity. It includes the following features:

- Demographics: Gender, Age, Height, Weight
- Health & Lifestyle: Family history of overweight, Smoking habits, Alcohol consumption, Water intake
- Dietary Habits: Frequency of high-caloric food, Vegetables, Meals, Snacks
- Physical Activity: Frequency of exercise, Technology use, Mode of transportation
- Obesity Level: Target variable indicating obesity status
This dataset is useful for studying the impact of lifestyle choices on obesity.
<!-- ## Folder Structure

The project is organized into the following directories:

- analysis/: Contains Jupyter Notebook for EDA.
    - EDA.ipynb: Notebook with exploratory data analysis.
- data/: Stores the raw dataset.
    - obesity_risk.zip: Compressed dataset file.
- extracted_data/: Contains extracted CSV file after ingestion.
    - obesity_level.csv: Processed dataset.
- mlruns/: Stores MLflow tracking information.
- pipelines/: Contains ZenML pipeline scripts.
    - training_pipeline.py: Defines the model training pipeline.
    - deployment_pipeline.py: Defines the deployment pipeline.
- report/: Stores visualization reports. 
    - report.pbix: Power BI report.
- saved_models/: Stores trained models. 
    - trained_model.pkl: Saved model file.
- src/: Contains core ML processing scripts. 
    - ingest_data.py: Handles data ingestion.
    - clean_data.py: Performs data cleaning. 
    - feature_encoding.py: Encodes categorical features.
    - feature_scaling.py: Scales numerical features.
    - data_splitter.py: Splits data for training/testing.
    - model_selection.py: Performs model selection.
    - model_trainer.py: Trains the selected model.
    - model_evaluator.py: Evaluates trained model performance.
- steps/: Contains individual step implementations for ZenML. 
    - data_ingestion_step.py: Step for data ingestion.
    - clean_data_step.py: Step for data cleaning.
    - feature_encoding_step.py: Step for feature encoding.
    - feature_scaling_step.py: Step for feature scaling.
    - data_splitter_step.py: Step for splitting dataset.
    - model_selection_step.py: Step for model selection.
    - model_trainer_step.py: Step for training the model.
    - model_evaluation_step.py: Step for evaluating model performance.
- test/: Contains testing scripts. 
    - test_notebook.ipynb: Jupyter Notebook for testing model.
- config.json: Configuration file.
- environment.yml: Conda environment dependencies.
- request_model.py: Script to send requests to the deployed model. 
- run_deployment.py: Script to run deployment pipeline.
- run_pipeline.py: Script to run training pipeline.
- setup.py: Setup script.
- streamlit_app.py: Streamlit web app for user interaction. -->
## Pipeline Workflow

![image](https://drive.google.com/uc?export=view&id=11ivj7ZI_ht9lqsRTdPf8ZRP1YhNw43v3)

1. **Exploratory Data Analysis (EDA)**
EDA is performed separately from the pipeline to gain insights into the dataset.
- Store insights in Jupyter Notebook
- Create a Power BI Report for visualization.

2. **Machine Learning Pipeline**
The pipeline consists of multiple steps:

**Data Processing Steps**

    1. Data Ingestion: Extracts data from a zip file and stores it in the extracted_data folder.
    2. Data Cleaning: Drops unwanted columns from the dataset.
    3. Feature Encoding: Converts categorical features using Label Encoder.
        - One-hot encoding option is available but not applied.
    4. Feature Scaling: Features are scaled using:
        - Log Transformation
        - Standard Scaler
    5. Data Splitting: Splits the dataset into training and testing sets.
**Model Training and Evaluation Steps**

    6. Model Selection:
        - Performs GridSearchCV and hyperparameter tuning.
        - Selects the best model among Tree-Based Models (since Label Encoding is used).
    7. Model Trainer: Trains the selected model using MLflow.
    8. Model Evaluation: Computes accuracy metrics of the trained model using MLflow.
3. Pipelines

- Trained Pipeline:
    - Trains the model using the above steps.
    - Saves the model to local disk and Azure Machine Learning Workspace.

- Deployment Pipeline:
    - Deploys the trained model as a real-time endpoint on Azure Machine Learning Studio.

4. Model Deployment & Prediction

- The deployed model can be accessed via an endpoint.
- A Streamlit app is used to interact with the model for predictions.
- MLflow is used to track the model performance and experiment runs.


## Run the Project

1. Clone the Repository:


```bash
git clone <repository_url>
cd obesity-risk-predictor
```
2. Create a Conda Environment:
```bash
conda create --name obesity_env python=3.8
conda activate obesity_env
```
3. Install Dependencies:
```bash
pip install -r requirements.txt
```
4. Run the Training Pipeline:
```bash
python run_pipeline.py
```
This trains the model and saves it in the saved_models/ folder.

5. Deploy the Model:
```bash
python run_deployment.py
```
This deploys the model on Azure Machine Learning Studio as a real-time endpoint.

Update Streamlit App with Endpoint URL:

Replace the endpoint URL in streamlit_app.py with the newly deployed model’s endpoint.

Run the Streamlit App:
```bash
streamlit run streamlit_app.py
```
## Conclusion

This project follows a structured ML workflow with ZenML, Azure ML, and MLflow to automate model training, tracking, and deployment. The deployed model is accessible in real-time via an API and is integrated with a Streamlit UI for easy interaction.

For further improvements, additional encoding techniques and model architectures can be explored.