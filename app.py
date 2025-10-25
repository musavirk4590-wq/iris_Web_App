import streamlit as st
import joblib
import numpy as np


try:
    model = joblib.load('iris_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please train and save the model first.")
    st.stop()


st.title("Iris Flower Species Classifier")
st.write("This app uses a machine learning model to predict the species of an Iris flower based on its measurements.")


st.header("Input Flower Measurements (cm)")
sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.4)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.4)
petal_length = st.slider("Petal Length", 1.0, 7.0, 1.3)
petal_width = st.slider("Petal Width", 0.1, 2.5, 0.2)


if st.button("Predict Species"):
 
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
   
    prediction_code = model.predict(input_data)[0]
    

    species_names = ['Setosa', 'Versicolor', 'Virginica']
    predicted_species = species_names[prediction_code]
    

    st.success(f"The predicted species is: {predicted_species}")