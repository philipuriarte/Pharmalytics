import streamlit as st
import pandas as pd
import numpy as np
import os

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

# Load the previously uploaded dataset and stop if dataset doesn't exist
uploaded_dataset = None
if os.path.exists("uploaded_dataset.csv"):
    uploaded_dataset = pd.read_csv("uploaded_dataset.csv", index_col=None)
else:
    st.warning("Dataset not uploaded. Please upload a dataset first in the Home page.")
    st.stop()

st.write("Click the button below to generate sales predictions.")

if st.button("Generate", help="Click to generate sales predictions"):
    pass
