import pandas as pd
import numpy as np
import streamlit as st
import pickle
from datetime import datetime

def welcome():
    return "Welcome All"

pickle_in = open("rfc.pkl","rb")
rfc=pickle.load(pickle_in)

def predict_crime_probabilities(input_df):
    return rfc.predict_proba(input_df.values.reshape(1, -1))[0]

def determine_higher_probability(input_df):
    crime_probabilities = predict_crime_probabilities(input_df)
    violent_prob = crime_probabilities[:, 0]  # Probability of violent crime
    property_prob = crime_probabilities[:, 1]  # Probability of property crime
    if violent_prob > property_prob:
        return "Violent crime is more likely to occur."
    elif property_prob > violent_prob:
        return "Property crime is more likely to occur."
    else:
        return "Equal probabilities for both types of crimes."


def main():
    st.title("Baltimore Crime Predictor")
    
    
    latitude=st.text_input("Latitude")
    longitude = st.text_input("Longitude")
    date_str = st.text_input("Date (MM/DD/YYYY) ")
    time_str = st.text_input("Time", help="Enter time in the format HH:MM")
    age = st.text_input("Age")
    
    if date_str and time_str:
        try:
            date = datetime.strptime(date_str, '%m/%d/%Y')
            time = datetime.strptime(time_str, '%H:%M')
            
            input_data = {
            'Latitude': [latitude],
            'Longitude': [longitude],
            'Year': [date.year],
            'Month': [date.month],
            'day_of_month': [date.day],
            'day_of_year': [date.timetuple().tm_yday],
            'week_of_year': [date.isocalendar()[1]],
            'day_of_week': [date.weekday()],
            'Hour': [time.hour],
            'Age': [age]
            }
            
            input_df=pd.DataFrame(input_data)
        except ValueError:
            print("Enter date and time in valid format")
    else:
        date_str=None
        time_str=None
    
    
    result=""
    if st.button("Predict"):
        result=determine_higher_probability(input_df)
    
    st.success(result)
           
        
        
        
        
if __name__=='__main__':
    main()