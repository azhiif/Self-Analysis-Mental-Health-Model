import pickle
import pandas as pd
import joblib

# Load the trained model
with open('models/mental_health_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the feature encoder
ct = joblib.load('models/feature_values.pkl')

# Function to preprocess input data
def preprocess_inputs(input_data):
    input_df = pd.DataFrame([input_data])
    transformed_data = ct.transform(input_df)
    return transformed_data

# Function to make predictions
def predict_mental_health(input_data):
    processed_data = preprocess_inputs(input_data)
    prediction = model.predict(processed_data)[0]
    return "You may need mental health support." if prediction == 1 else "You may not need mental health support."

# Example usage
if __name__ == "__main__":
    input_data = {
        'Age': 30,
        'Gender': 'Male',
        'self_employed': 'No',
        'family_history': 'Yes',
        'work_interfere': 'Sometimes',
        'no_employees': '6-25',
        'remote_work': 'No',
        'tech_company': 'Yes',
        'benefits': 'Yes',
        'care_options': 'No',
        'wellness_program': 'No',
        'seek_help': 'Yes',
        'anonymity': 'Yes',
        'leave': 'Somewhat easy',
        'mental_health_consequence': 'No',
        'phys_health_consequence': 'No',
        'coworkers': 'Yes',
        'supervisor': 'Yes',
        'mental_health_interview': 'Yes',
        'phys_health_interview': 'Yes',
        'mental_vs_physical': 'Yes',
        'obs_consequence': 'No'
    }
    result = predict_mental_health(input_data)
    print(result)
