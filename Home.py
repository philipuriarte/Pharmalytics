import streamlit as st
import pandas as pd
import os


# Set page title and icon
st.set_page_config(
    page_title="Home",
    page_icon="🏠",
)

# Main content
st.title("Welcome to Pharmalytics!")
st.markdown(
    """
    Pharmalytics is a sales prediction system developed for FirstMed Pharmacy using the **SARIMA** model.
    ### Instructions
    1. Upload the pharmacy sales dataset below.
    2.  👈 Select Analytics or Predictions from the sidebar
        - To gain insights into the dataset, click on the "Analytics" option. 
        This will provide you with a comprehensive overview and analysis of the sales data, 
        including key statistics, trends, and visualizations.
        - To generate sales predictions, click on "Predictions" option.
        Pharmalitics will employ the SARIMA model to generate sales predictions based on the uploaded dataset.
"""
)

# Prompt to upload dataset
dataset = None
file = st.file_uploader("Upload Sales Dataset")
if file:
    dataset = pd.read_csv(file, index_col=None)
    st.success("Dataset uploaded successfully!")
    st.write("**Dataset Preview:**")
    st.dataframe(dataset, width=700)

    dataset.to_csv("uploaded_dataset.csv", index=None) # Save dataset to local machine

# Load the previously uploaded dataset (if exists)
uploaded_dataset = None
if os.path.exists("uploaded_dataset.csv"):
    uploaded_dataset = pd.read_csv("uploaded_dataset.csv", index_col=None)

# Show the previously uploaded dataset and drop unwanted columns
if uploaded_dataset is not None and dataset is None:
    uploaded_dataset.drop(['Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7'], axis=1, inplace=True)
    st.write("**Uploaded Dataset:**")
    st.dataframe(uploaded_dataset, width=700)
