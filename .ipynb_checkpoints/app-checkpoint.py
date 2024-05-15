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
with open('Config/filepaths.json') as f:
    FPATHS = json.load(f)


#Title
st.title("Predicting Movie Review Ratings")

# Get text to predict from the text input box
X_to_pred = st.text_input("### Enter text to predict here:", 
                          value="I loved the movies! Great story.")


# Loading the ML model
@st.cache_resource
def load_ml_model(fpath):
    loaded_model = joblib.load(fpath)
    return loaded_model
# Load model from FPATHS
model_fpath = FPATHS['models']['nbayes']
count_pipe = load_ml_model(model_fpath)

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
    # Decode label
    pred_class = lookup_dict[pred_class]
    return pred_class
# Trigger prediction with a button
if st.button("Get prediction."):
    pred_class = make_prediction(X_to_pred)
    st.markdown(f"##### Predicted category:  {pred_class}") 
else: 
    st.empty()