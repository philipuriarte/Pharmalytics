import streamlit as st
import pandas as pd
import numpy as np

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

st.write("Click the button below to generate sales predictions.")

if st.button("Generate", help="Click to generate sales predictions"):
    pass
