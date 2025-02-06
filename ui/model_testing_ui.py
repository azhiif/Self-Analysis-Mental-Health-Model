import gradio as gr
import pickle
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import OrdinalEncoder


# Load the saved model
with open('models/mental_health_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the feature encoder (with a fix for unknown values)
ct = joblib.load('models/feature_values.pkl')

# Ensure that OrdinalEncoder can handle unknown values
for transformer in ct.transformers_:
    if isinstance(transformer[1], OrdinalEncoder):
        transformer[1].set_params(handle_unknown="use_encoded_value", unknown_value=-1)

# Define input fields
categorical_columns = [
    'Gender', 'self_employed', 'family_history', 'work_interfere', 'no_employees', 
    'remote_work', 'tech_company', 'benefits', 'care_options', 'wellness_program', 'seek_help',
    'anonymity', 'leave', 'mental_health_consequence', 'phys_health_consequence', 
    'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview', 
    'mental_vs_physical', 'obs_consequence'
]

all_columns = ['Age'] + categorical_columns  # Include Age

# **Fix Gender Mapping**
def clean_gender(gender):
    gender_map = {
        'Male': 'Male', 'Male ': 'Male', 'Mail': 'Male', 'Malr': 'Male',
        'Female': 'Female', 'cis-female/femme': 'Female',
        'Non-Binary': 'Non-Binary', 
        'male leaning androgynous': 'Non-Binary',
        'ostensibly male, unsure what that really means': 'Non-Binary'
    }
    return gender_map.get(gender, 'Non-Binary')  # Default to 'Non-Binary'

# **Preprocess user inputs**
def preprocess_inputs(age, *args):
    input_dict = {'Age': age}
    input_dict['Gender'] = clean_gender(args[0])  # Fix Gender Mapping
    input_dict.update({col: args[i] for i, col in enumerate(categorical_columns[1:])})
    return pd.DataFrame([input_dict])

# **Chatbot function**
def chatbot_response(age, *args):
    try:
        input_df = preprocess_inputs(age, *args)

        # **Transform input data using the pre-fitted encoder**
        transformed_data = ct.transform(input_df)

        # **Make Prediction**
        prediction = model.predict(transformed_data)[0]

        # **Return Output**
        result = "You may need mental health support." if prediction == 1 else "You may not need mental health support."
        return result
    except Exception as e:
        return f"Error: {str(e)}"

# **Define Gradio interface**
iface = gr.Interface(
    fn=chatbot_response,
    inputs=[
    gr.Number(label='Age (18-60)', minimum=18, maximum=60),
    gr.Radio(['Male', 'Female', 'Non-Binary'], label='Gender'),
    gr.Radio(['No', 'Yes'], label='Are you self-employed?'),
    gr.Radio(['No', 'Yes'], label='Do you have a family history of mental health issues?'),
    gr.Radio(['Never', 'No', 'Often', 'Rarely', 'Sometimes'], label='Does work interfere with your mental health?'),
    gr.Radio(['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000'], label='Number of Employees in your company'),
    gr.Radio(['No', 'Yes'], label='Do you work remotely?'),
    gr.Radio(['No', 'Yes'], label='Do you work for a tech company?'),
    gr.Radio(["Don't know", 'No', 'Yes'], label='Do you have access to benefits at work?'),
    gr.Radio(['No', 'Not sure', 'Yes'], label='Do you have care options provided by your company?'),
    gr.Radio(["Don't know", 'No', 'Yes'], label='Does your company offer a wellness program?'),
    gr.Radio(["Don't know", 'No', 'Yes'], label='Do you seek help for mental health issues?'),
    gr.Radio(["Don't know", 'No', 'Yes'], label='Do you value anonymity when seeking help?'),
    gr.Radio(["Don't know", 'Somewhat difficult', 'Somewhat easy', 'Very difficult', 'Very easy'], label='Is it easy for you to take leave when needed?'),
    gr.Radio(['Maybe', 'No', 'Yes'], label='Do you think mental health issues can have a consequence at work?'),
    gr.Radio(['Maybe', 'No', 'Yes'], label='Do you think physical health issues can have a consequence at work?'),
    gr.Radio(['No', 'Some of them', 'Yes'], label='Do your coworkers understand mental health issues?'),
    gr.Radio(['No', 'Some of them', 'Yes'], label='Does your supervisor understand mental health issues?'),
    gr.Radio(['Maybe', 'No', 'Yes'], label='Would you attend a mental health interview at work?'),
    gr.Radio(['Maybe', 'No', 'Yes'], label='Would you attend a physical health interview at work?'),
    gr.Radio(["Don't know", 'No', 'Yes'], label='Do you think mental and physical health are related?'),
    gr.Radio(['No', 'Yes'], label='Do you think there are consequences for observing mental health issues in others?'),

    ],
    outputs="text",
    title="Mental Health Self-Assessment Chatbot",
    description="Answer the following questions to assess your mental health status based on the trained model."
)

# **Launch Gradio App**
iface.launch()
