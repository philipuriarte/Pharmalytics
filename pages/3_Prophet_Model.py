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
    page_title="Prophet Sales Predictions",
    page_icon="💰",
)

# Main content
st.title("Prophet Sales Predictions 💰")
st.markdown(
    """
    Pharmalytics uses the Prophet model to generate sales predictions based on your uploaded dataset. 
    This advanced technique captures seasonal and trend patterns in the data to forecast future trends.

    👈 Select a product to predict and how far into the future to predict from the sidebar.
"""
)
descrip_exp = st.expander("See Extra Information")
descrip_exp.markdown(
    """
    To ensure efficiency and practicality, our app focuses on predicting sales for the top 10% most sold products 
    in FirstMed Pharmacy and leverages the power of the Prophet model for predicting sales. By automating the model 
    fitting process, we eliminate the need for manual selection and comparison of different models.
    
    This streamlined approach enables us to provide forecasts for the key items driving the pharmacy's revenue and 
    saves time and resources, allowing us to focus on delivering reliable sales predictions while leaving 
    room for future expansion and inclusion of additional products.

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
top_percentage = 0.10  # You can adjust this value as needed
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


        # Prophet expects a specific format for the input DataFrame
        pred_resampled_data_prophet = pred_resampled_data.reset_index().rename(columns={"index": "ds", "Quantity": "y"})

        # Instantiate the Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,  # Adjust as needed based on your data
            seasonality_prior_scale=10.0,  # Experiment with different values
            changepoint_prior_scale=0.05,  # Experiment with different values
            holidays_prior_scale=10.0,  # Experiment with different values
        )
        
        # Fit the model to the entire resampled data
        model_fit = model.fit(pred_resampled_data_prophet)

        # Create a DataFrame with the future dates for prediction
        future = model.make_future_dataframe(periods=16, freq="W-MON")

        # Generate predictions on the future dates
        forecast = model.predict(future)

        # Extract predictions
        predictions = forecast[['ds', 'yhat']]
        predictions['ds'] = pd.to_datetime(predictions['ds'])
        predictions = predictions.set_index('ds')

        st.subheader(product_to_predict + " Sales Predictions")
        act_pred_tab, final_app_tab = st.tabs(["📒 Actual vs Predicted", "📊 Final App"])

        col1, col2 = st.columns(2)

        with act_pred_tab:
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
        
        # Take the intersection of dates
        common_dates = pred_resampled_data.index.intersection(predictions.index)

        # Calculate Mean Absolute Error (MAE) using common dates
        mae = mean_absolute_error(pred_resampled_data.loc[common_dates, 'Quantity'], predictions.loc[common_dates, 'yhat'])

        # Calculate Mean Absolute Percentage Error (MAPE) using common dates
        mape = (mae / pred_resampled_data.loc[common_dates, 'Quantity'].mean()) * 100

        # Calculate R-squared (R²) using common dates
        y_true = pred_resampled_data.loc[common_dates, 'Quantity']
        y_pred = predictions.loc[common_dates, 'yhat']
        r_squared = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

        # Calculate Root Mean Squared Error (RMSE) using common dates
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        st.write("**Accuracy Results**")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
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



