import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import json
import custom_functions as fn
with open('Config/filepaths.json') as f:
    FPATHS = json.load(f)


#Title
st.title("Predicting Movie Review Ratings")

# Get text to predict from the text input box
X_to_pred = st.text_input("### Enter text to predict here:", 
                          value="I loved the movies! Great story.")

@st.cache_data
def load_Xy_data(fpath):
    return joblib.load(fpath)

#test 
X_test, y_test = load_Xy_data(FPATHS['data']['ml']['test'])

#train
X_train, y_train = load_Xy_data(FPATHS['data']['ml']['train'])

# Loading the ML model
@st.cache_resource
def load_ml_model(fpath):
    loaded_model = joblib.load(fpath)
    return loaded_model
# Load model from FPATHS
model_fpath = FPATHS['models']['nbayes']
count_pipe = load_ml_model(model_fpath)

labels = ['High', 'Low']
# load target lookup dict
@st.cache_data
def load_lookup(fpath=FPATHS['data']['ml']['target_lookup']):
    return joblib.load(fpath)

@st.cache_resource
def load_encoder(fpath=FPATHS['data']['ml']['label_encoder'] ):
    return joblib.load(fpath)

# Load the target lookup dictionary
target_lookup = load_lookup()
# Load the encoder
encoder = load_encoder()
# Update the function to decode the prediction
def make_prediction(X_to_pred, count_pipe=count_pipe, lookup_dict= target_lookup):
    # Get Prediction
    pred_class = count_pipe.predict([X_to_pred])[0]
    
    return pred_class
# Trigger prediction with a button
if st.button("Get prediction."):
    pred_class = make_prediction(X_to_pred)
    st.markdown(f"##### Predicted category:  {pred_class}") 
else: 
    st.empty()

if st.button('Evaluate Model'):
    train_report, test_report, eval_fig = fn.evaluate_classification(count_pipe, X_train, y_train, X_test, y_test)
    st.text('Training Report')
    st.text(train_report)
    st.text('Test Report')
    st.text(test_report)
    st.pyplot(eval_fig)