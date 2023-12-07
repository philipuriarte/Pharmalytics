import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet

# Set page title and icon
st.set_page_config(
    page_title="Individual Forecasting",
    page_icon="💰",
)

# Main content
st.title("Individual Forecasting 💰")
st.markdown(
    """
    Pharmalytics uses the Prophet model to generate sales predictions based on your uploaded dataset. 
    This advanced technique captures seasonal and trend patterns in the data to forecast future trends.

    To ensure efficiency and practicality, our app focuses on predicting sales for the top 10% most sold products 
    in FirstMed Pharmacy and leverages the power of the Prophet model for predicting sales.

    👈 Select a product to predict from the sidebar.
"""
)
descrip_exp = st.expander("See Extra Information")
descrip_exp.markdown(
    """
    The model aggregates sales data on a weekly basis and predictions are made for the 
    next 12 weeks beyond the latest date in the dataset.

    The top 10% most sold products are prioritized for predictions because they have a larger dataset, allowing for 
    more accurate forecasts, and their sales have a greater impact on overall revenue compared to products outside 
    the top 10% where the aggregated datasets are smaller, leading to less accurate predictions due to the limited 
    historical sales information.
"""
)

st.sidebar.header("Sales Predictions")

# Load the preprocessed dataset and stop if it doesn't exist
preprocessed_dataset_path = "preprocessed_dataset.csv"
if not os.path.exists(preprocessed_dataset_path):
    st.warning("Dataset not uploaded. Please upload a dataset first in the Home page.")
    st.stop()

preprocessed_dataset = pd.read_csv(preprocessed_dataset_path, parse_dates=["Date Sold"], index_col="Date Sold")

# Create containers to group codes together
top_products_pred_con = st.container()

# Calculate the top 10% most sold products
top_percentage = 0.10
total_products = len(preprocessed_dataset["Product Name"].unique())
top_products_count = round(total_products * top_percentage)

# Get the top 10% most sold products
top_products = preprocessed_dataset.groupby("Product Name")["Quantity"].sum().nlargest(top_products_count).index

# Get the minimum and maximum dates from the filtered dataset and set to beginning and end of the months respectively
min_date = pd.Timestamp(preprocessed_dataset.index.min().date())
max_date = preprocessed_dataset.index.max().date()

# Time interval (daily, weekly, monthly)
time_interval = "W-MON"

# Create a date range from min_date to max_date
date_range = pd.date_range(start=min_date, end=max_date, freq=time_interval)

# Get unique product names from top_30_products
unique_product_names = top_products.unique().tolist()

with st.sidebar:
    # Input Widgets
    product_to_predict = st.selectbox("Select a product to predict", unique_product_names, index=0)
    generate_button = st.button("Generate", help="Click to generate sales predictions")

with top_products_pred_con:
    if generate_button is not True:
        st.stop()
    
    with st.spinner('Processing...'):

        # Get the data for the selected product
        pred_product_data = preprocessed_dataset[preprocessed_dataset["Product Name"] == product_to_predict]
        pred_resampled_data = pred_product_data.drop(["Product Name", "Sell Price", "Product Category"], axis=1).resample(time_interval).sum().reindex(date_range, fill_value=0).reset_index()
        pred_resampled_data = pred_resampled_data.set_index("index")

        # Remove outliers using the IQR method
        Q1 = pred_resampled_data['Quantity'].quantile(0.05)
        Q3 = pred_resampled_data['Quantity'].quantile(0.80)
        IQR = Q3 - Q1

        pred_processed_data = pred_resampled_data[~((pred_resampled_data['Quantity'] < (Q1 - 1.5 * IQR)) | (pred_resampled_data['Quantity'] > (Q3 + 1.5 * IQR)))]


        # Prophet expects a specific format for the input DataFrame
        pred_resampled_data_prophet = pred_processed_data.reset_index().rename(columns={"index": "ds", "Quantity": "y"})        
        pred_resampled_data_prophet2 = pred_resampled_data.reset_index().rename(columns={"index": "ds", "Quantity": "y"})

        # Instantiate the Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,  # Adjust as needed based on your data
            seasonality_prior_scale=10.0,  # Experiment with different values
            changepoint_prior_scale=0.05,  # Experiment with different values
            holidays_prior_scale=10.0,  # Experiment with different values
        )

        model2 = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,  # Adjust as needed based on your data
            seasonality_prior_scale=10.0,  # Experiment with different values
            changepoint_prior_scale=0.05,  # Experiment with different values
            holidays_prior_scale=10.0,  # Experiment with different values
        )
        
        # Fit the model to the entire resampled data
        model = model.fit(pred_resampled_data_prophet)
        model2 = model2.fit(pred_resampled_data_prophet2)

        # Create a DataFrame with the future dates for prediction
        future = model.make_future_dataframe(periods=12, freq=time_interval)
        future2 = model2.make_future_dataframe(periods=12, freq=time_interval)

        # Generate predictions on the future dates
        forecast = model.predict(future)
        forecast2 = model2.predict(future2)

        # Extract predictions
        predictions = forecast[['ds', 'yhat']]
        predictions['ds'] = pd.to_datetime(predictions['ds'])
        predictions = predictions.set_index('ds')
        
        predictions2 = forecast2[['ds', 'yhat']]
        predictions2['ds'] = pd.to_datetime(predictions2['ds'])
        predictions2 = predictions2.set_index('ds')

        st.subheader(product_to_predict + " Sales Predictions")
        no_outlier, with_outlier = st.tabs(["Without Outliers", "With Outliers"])

        col1, col2 = st.columns(2)

        with no_outlier:
            data = pd.concat([pred_resampled_data['Quantity'].rename('Actual'), predictions['yhat'].rename('Predicted')], axis=1).reset_index()

            chart = alt.Chart(data).mark_line().encode(
                x='index:T',
                y=alt.Y('value:Q', axis=alt.Axis(title='Quantity')),
                color=alt.Color('data:N', scale=alt.Scale(domain=['Actual', 'Predicted'], range=['steelblue', 'orange'])),
                tooltip=['index:T', 'value:Q', 'data:N']
            ).transform_fold(
                fold=['Actual', 'Predicted'],
                as_=['data', 'value']
            ).properties(
                title="Actual vs Predicted",
                width=600,
                height=400
            )

            st.altair_chart(chart)
        
        with with_outlier:
            data2 = pd.concat([pred_resampled_data['Quantity'].rename('Actual'), predictions2['yhat'].rename('Predicted')], axis=1).reset_index()

            chart2 = alt.Chart(data2).mark_line().encode(
                x='index:T',
                y=alt.Y('value:Q', axis=alt.Axis(title='Quantity')),
                color=alt.Color('data:N', scale=alt.Scale(domain=['Actual', 'Predicted'], range=['steelblue', 'orange'])),
                tooltip=['index:T', 'value:Q', 'data:N']
            ).transform_fold(
                fold=['Actual', 'Predicted'],
                as_=['data', 'value']
            ).properties(
                title="Actual vs Predicted",
                width=600,
                height=400
            )

            st.altair_chart(chart2)
              
        # Take the intersection of dates
        common_dates = pred_resampled_data.index.intersection(predictions.index)

        # Calculate Mean Absolute Error (MAE) using common dates
        mae = mean_absolute_error(pred_resampled_data.loc[common_dates, 'Quantity'], predictions.loc[common_dates, 'yhat'])

        # Calculate R-squared (R²) using common dates
        y_true = pred_resampled_data.loc[common_dates, 'Quantity']
        y_pred = predictions.loc[common_dates, 'yhat']
        r_squared = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

        # Calculate Root Mean Squared Error (RMSE) using common dates
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        st.write("**Accuracy Results**")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"R-squared (R²): {r_squared:.4f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        
        extra_info_expander = st.expander("See Extra Information")
        with extra_info_expander:
            col1, col2, col3 = st.columns(3)            
            
            with col1:
                st.write(f"**{product_to_predict} Sales**")
                st.dataframe(pred_resampled_data)
            with col2:
                st.write("**Predictions**")
                st.dataframe(predictions)

            st.write("**Forecast Dataframe**")
            st.dataframe(forecast)

