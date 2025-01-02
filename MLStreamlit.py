# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 11:22:53 2025

@author: KARTHIK
"""

import numpy as np
import pickle
import streamlit as st

diabetes_model = pickle.load(open('diabetes_model.sav','rb'))

def diabetes_prediction(input_data):
    input_data = (5,166,72,19,175,25.8,0.587,51)

    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = diabetes_model.predict(input_data_reshaped)

    if (prediction[0] == 0):
        print('The person is not diabetic')
    else:
        print('The person is diabetic')


def main():
    st.title('Diabetes Prediction System')
    
    Pregnancies = st.text_input('Number of pregnancies')
    Glucose = st.text_input('Glucose level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin thickness value')
    Insulin = st.text_input('insulin level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the person')
    
    diagnosis=''
    
    if st.button('Diabetes Test Result'):
        
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)

if __name__ == '__main__':
    main()
    
