import streamlit as st
import pandas as pd
import numpy as np
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
        - Requirements: CSV file format, Column Headers [Product Name (string), Quantity (int), Sell Price (int), 
        Date Sold (datetime: dd-mm-yyyy), Product Category (string)]
    2.  👈 Select Analytics or Predictions from the sidebar
        - To gain insights into the dataset, click on the "Analytics" option. 
        This will provide you with a comprehensive overview and analysis of the sales data, 
        including key statistics, trends, and visualizations.
        - To generate sales predictions, click on "Predictions" option.
        Pharmalitics will employ the SARIMA (Seasonal Autoregressive Integrated Moving Average) model 
        to generate sales predictions based on the uploaded dataset.
"""
)

# Prompt to upload dataset
dataset = None
file = st.file_uploader("Upload Sales Dataset", type="csv")
if file:
    dataset = pd.read_csv(file, index_col=False)
    st.success("Dataset uploaded successfully!")
    st.write("**Dataset Preview:**")
    st.dataframe(dataset, width=700)

    dataset.to_csv("uploaded_dataset.csv", index=None) # Save dataset to local machine
elif not os.path.exists("uploaded_dataset.csv") and not file:
    st.stop()

# Load the previously uploaded dataset (if exists)
if os.path.exists("uploaded_dataset.csv") and not file:
    dataset = pd.read_csv("uploaded_dataset.csv", index_col=None)
    st.write("**Uploaded Dataset:**")
    st.dataframe(dataset, width=700)

st.divider()

# Create containers to group codes together
pre_con = st.container()

with pre_con:
    # Drop the unnamed columns
    unnamed_columns = [col for col in dataset.columns if 'Unnamed' in col]
    dataset.drop(unnamed_columns, axis=1, inplace=True)

    # Replace all occurrences of "#REF!" with NaN (because of auto-fill category in Google Sheet)
    dataset.replace("#REF!", np.nan, inplace=True)

    # Drop all rows that contain NaN values (All rows that have a single NaN value will be dropped)
    dataset.dropna(inplace=True)

    cleaned_dataset = dataset.reset_index(drop=True)

    # Convert the "Date Sold" column to datetime format
    cleaned_dataset["Date Sold"] = pd.to_datetime(cleaned_dataset["Date Sold"], format="%m/%d/%Y")

    # Create a new DataFrame with the dates as the index
    indexed_dataset = cleaned_dataset.set_index("Date Sold")

    # Show Preprocessed Dataset
    st.subheader("Data Pre-processing")
    st.markdown(
        """
        1. **Data Cleaning**: Rows and columns with empty cells are removed from the dataset.
        2. **Set DateTime Index**: Replace the index with a datetime index, enabling analysis and tracking of trends over time.
    """
    )
    st.write("**Preprocessed Dataset**")
    st.dataframe(indexed_dataset, width=700)

    indexed_dataset.to_csv("preprocessed_dataset.csv", date_format="%m/%d/%Y") # Save preprocessed dataset to local machine
