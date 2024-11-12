import sklearn
import streamlit as st
import pandas as pd
import pickle

# Load the trained model using pickle
with open('financial_inclusion_classifier.pkl', 'rb') as file:
    model = pickle.load(file)

# Title and intro
st.title("Financial Inclusion Prediction App")
st.write("Predict whether an individual has or uses a bank account based on demographic information.")

# Input fields for user features
country = st.selectbox("Country", ["Country0", "Country1", "Country2", "Country3"])  
year = st.number_input("Year", min_value=2000, max_value=2024)
uniqueid = st.text_input("Unique Identifier")
location_type = st.selectbox("Location Type", ["Rural", "Urban"])
cellphone_access = st.selectbox("Cellphone Access", ["Yes", "No"])
household_size = st.number_input("Household Size", min_value=1)
age_of_respondent = st.number_input("Age of Respondent", min_value=18, max_value=100)
gender_of_respondent = st.selectbox("Gender", ["Male", "Female"])
relationship_with_head = st.selectbox("Relationship with Head of Household", [
    "Head of Household", "Spouse", "Child", "Parent", "Other relative", "Other non-relatives", "Don’t know"])
marital_status = st.selectbox("Marital Status", [
    "Married/Living together", "Divorced/Separated", "Widowed", "Single/Never Married", "Don’t know"])
education_level = st.selectbox("Education Level", [
    "No formal education", "Primary education", "Secondary education", 
    "Vocational/Specialised training", "Tertiary education", "Other/Dont know/RTA"])
job_type = st.selectbox("Job Type", [
    "Farming and Fishing", "Self employed", "Formally employed Government", 
    "Formally employed Private", "Informally employed", "Remittance Dependent", 
    "Government Dependent", "Other Income", "No Income", "Don’t Know/Refuse to answer"])

# Collect input data
data = {
    "country": country,
    "year": year,
    "uniqueid": uniqueid,
    "location_type": location_type,
    "cellphone_access": cellphone_access,
    "household_size": household_size,
    "age_of_respondent": age_of_respondent,
    "gender_of_respondent": gender_of_respondent,
    "relationship_with_head": relationship_with_head,
    "marital_status": marital_status,
    "education_level": education_level,
    "job_type": job_type,
}

input_df = pd.DataFrame([data])

# Prediction output
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.write(f"Prediction: {'Bank Account' if prediction[0] == 1 else 'No Bank Account'}")
