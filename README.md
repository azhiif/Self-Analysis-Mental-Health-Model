# Self-Analysis-Mental-Health-Model
Mental Health Prediction Project: A machine learning model to predict the need for mental health support based on survey data. Includes data preprocessing, model training, and an interactive Gradio UI for predictions.


## Overview
This project predicts whether an individual may need mental health support based on survey data. It includes data preprocessing, model training, and a user interface for predictions.

## Dataset Preprocessing Steps
1. **Load the dataset**: Load the `survey.csv` file.
2. **Remove unnecessary columns**: Drop `Country`, `state`, `Timestamp`, and `comments`.
3. **Handle missing values**: Fill missing values in `self_employed` and `work_interfere` columns.
4. **Filter invalid ages**: Remove rows where `Age` is outside the range of 18 to 60.
5. **Standardize categorical data**: Standardize values in the `Gender` column.

## Model Selection Rationale
- **Logistic Regression**: Baseline model for binary classification.
- **Random Forest**: Handles non-linear relationships and feature importance.
- **Decision Tree**: Simple and interpretable model.

## How to Run the Inference Script
1. Install dependencies:
   pip install -r requirements.txt
   
3. Run the interference script
   python inference/predict_mental_health.py

#UI/CLI Instructions
1. Install Gradio
   pip install gradio

2. Run Gradio UI
   python ui/mental_health_ui.py

3. Open the provided URL in your browser and input the required data   

    
