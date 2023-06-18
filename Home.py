import streamlit as st
import pandas as pd
import os
from utils import preprocess_dataset


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
        Pharmalytics will employ the SARIMA (Seasonal Autoregressive Integrated Moving Average) model 
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

# Create containers to group codes together
pre_con = st.expander("Show Preprocessing Procedure")

with pre_con:
    preprocessed_dataset = preprocess_dataset(dataset)

    # Show Preprocessed Dataset
    st.subheader("Data Pre-processing")
    st.markdown(
        """
        1. **Data Cleaning**: Rows and columns with empty cells are removed from the dataset.
        2. **Set DateTime Index**: Replace the index with a datetime index, enabling analysis and tracking of trends over time.
    """
    )
    st.write("**Preprocessed Dataset**")
    st.dataframe(preprocessed_dataset, width=700)

    preprocessed_dataset.to_csv("preprocessed_dataset.csv", date_format="%m/%d/%Y") # Save preprocessed dataset to local machine
