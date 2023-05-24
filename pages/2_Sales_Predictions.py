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

# Subheader for Product Sales Predictions
st.subheader("Product Sales Predictions")

# Get unique product names from the dataset
product_names = sorted(preprocessed_dataset["Product Name"].unique())

# Multiselect box to choose products
selected_products = st.multiselect("Select Products", product_names, max_selections=5)

# Filter the dataset for the selected products
product_sales_dataset = preprocessed_dataset[preprocessed_dataset["Product Name"].isin(selected_products)]

# Get the minimum and maximum dates from the filtered dataset and set to beginning and end of the months respectively
min_date = pd.Timestamp(preprocessed_dataset.index.min().date().replace(day=1))
max_date = preprocessed_dataset.index.max().date() + pd.offsets.MonthEnd(0)

# Create a date range from min_date to max_date
date_range = pd.date_range(start=min_date, end=max_date, freq="W-MON")

# Resample and preprocess the data for each selected product
resampled_datasets = {}
for product in selected_products:
    product_data = product_sales_dataset[product_sales_dataset["Product Name"] == product]

    # Resample the data on a daily basis and fill missing dates with zero quantities
    resampled_data = product_data.drop(["Product Name", "Sell Price", "Product Category"], axis=1).resample("W-MON").sum().reindex(date_range, fill_value=0).reset_index()
    resampled_data = resampled_data.set_index("index")  # Set "index" as the index (was previously "Date Sold")

    resampled_datasets[product] = resampled_data

    # Create two columns for the dataframe and plots
    col1, col2, col3 = st.columns(3)

    # Display the dataframe in the first column
    with col1:
        st.write("Dataset for", product)
        st.dataframe(resampled_datasets[product])

        # Calculate and display the sum of quantity
        quantity_sum = resampled_datasets[product]['Quantity'].sum()
        st.write("Sum of Quantity:", quantity_sum)

    # Display the time series plot and ACF plot in the second column
    with col2:
        st.write("Time Series Plot and ACF for", product)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
        
        # Plot the time series
        ax1.plot(resampled_datasets[product].index, resampled_datasets[product]['Quantity'])
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Quantity")

        # Add some spacing between the plots
        ax1.margins(x=0, y=0.1)
        ax2.margins(x=0, y=0.1)
        ax3.margins(x=0, y=0.1)
        plt.subplots_adjust(hspace=0.4)
        
        # Plot the ACF
        plot_acf(resampled_datasets[product]['Quantity'], ax=ax2)
        ax2.set_xlabel("Lag")
        ax2.set_ylabel("Autocorrelation")

        # Plot the PACF
        plot_pacf(resampled_datasets[product]['Quantity'], ax=ax3, lags=10)
        ax3.set_xlabel("Lag")
        ax3.set_ylabel("Partial Autocorrelation")

        st.pyplot(fig)
    
    # Display the ADF test results in the third column
    with col3:
        adf_result = adfuller(resampled_datasets[product]['Quantity'])
        st.write("Augmented Dickey-Fuller Test Results:")
        st.write("ADF Statistic:", adf_result[0])
        st.write("p-value:", adf_result[1])
        st.write("Critical Values:")
        for key, value in adf_result[4].items():
            st.write(f"{key}: {value}")

st.divider()

# Create train test split for each selected product
# for product, data in resampled_datasets.items():
#     # Split the dataset into training and testing sets
#     train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

#     # Print the name of the key
#     st.write("Train Test Set for", product)

#     # Create two columns for train and test sets
#     col1, col2 = st.columns(2)

#     # Display train set in the first column
#     with col1:
#         st.write("Train Set: ", train_data.shape[0])
#         st.dataframe(train_data)

#     # Display test set in the second column
#     with col2:
#         st.write("Test Set: ", test_data.shape[0])
#         st.dataframe(test_data)


