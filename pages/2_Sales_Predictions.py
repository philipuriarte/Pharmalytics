import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set page title and icon
st.set_page_config(
    page_title="Sales Predictions",
    page_icon="💰",
)

# Main content
st.title("Sales Predictions 💰")
st.markdown(
    """
    Pharmalitics uses the SARIMA model to generate sales predictions based on your uploaded dataset. 
    This advanced technique captures seasonal and trend patterns in the data to forecast future trends.
"""
)

# Load the preprocessed dataset and stop if doesn't exist
preprocessed_dataset = None
if os.path.exists("preprocessed_dataset.csv"):
    preprocessed_dataset = pd.read_csv("preprocessed_dataset.csv", parse_dates=["Date Sold"], index_col="Date Sold")
else:
    st.warning("Dataset not uploaded. Please upload a dataset first in the Home page.")
    st.stop()

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(preprocessed_dataset, test_size=0.2, shuffle=False)

# Show Train Set
st.write("**Train Set**: ", train_data.shape[0])
st.dataframe(train_data)

# Show Test Set
st.write("**Test Set**", test_data.shape[0])
st.dataframe(test_data)
