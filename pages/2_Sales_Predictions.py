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
st.write("*option to open this screen will only show one a dataset file has been uploaded*")
st.write("*will show sales predictions from trained SARIMA model*")