import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import os
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima

# Set page title and icon
st.set_page_config(
    page_title="Sales Predictions",
    page_icon="💰",
)

# Main content
st.title("Sales Predictions 💰")
st.markdown(
    """
    Pharmalytics uses the SARIMA model to generate sales predictions based on your uploaded dataset. 
    This advanced technique captures seasonal and trend patterns in the data to forecast future trends.

    To ensure efficiency and practicality, our app focuses on predicting sales for the top 30 most sold products 
    in FirstMed Pharmacy and leverages the power of Auto ARIMA for predicting sales. By automating the model 
    fitting process, we eliminate the need for manual selection and comparison of different models. 
    
    This streamlined approach enables us to provide forecasts for the key items driving the pharmacy's revenue and 
    saves time and resources, allowing us to focus on delivering reliable sales predictions while leaving 
    room for future expansion and inclusion of additional products.
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
top_30_products_pred_con = st.container()

# Get the top 30 most sold products
top_30_products = preprocessed_dataset.groupby("Product Name")["Quantity"].sum().nlargest(30).index

# Get the minimum and maximum dates from the filtered dataset and set to beginning and end of the months respectively
min_date = pd.Timestamp(preprocessed_dataset.index.min().date())
max_date = preprocessed_dataset.index.max().date()

# Time interval (daily, weekly, monthly)
time_interval = "W-MON"

# Create a date range from min_date to max_date
date_range = pd.date_range(start=min_date, end=max_date, freq=time_interval)

# Get unique product names from top_30_products
unique_product_names = top_30_products.unique().tolist()

predict_time_intervals = ["1 Week", "2 Weeks", "3 weeks", "1 Month"]

with st.sidebar:
    # Input Widgets
    product_to_predict = st.selectbox("Select a product to predict", unique_product_names, index=0)
    predict_interval = st.select_slider("Select how far into the future to predict", predict_time_intervals)
    generate_button = st.button("Generate", help="Click to generate sales predictions")

with top_30_products_pred_con:
    if generate_button is not True:
        st.stop()
    
    with st.spinner('Processing...'):

        # Get the data for the selected product
        pred_product_data = preprocessed_dataset[preprocessed_dataset["Product Name"] == product_to_predict]
        pred_resampled_data = pred_product_data.drop(["Product Name", "Sell Price", "Product Category"], axis=1).resample(time_interval).sum().reindex(date_range, fill_value=0).reset_index()
        pred_resampled_data = pred_resampled_data.set_index("index")

        # Split data into train and test sets
        train_data, test_data = train_test_split(pred_resampled_data, test_size=0.2, shuffle=False)

        # Use auto-SARIMA to determine the order and seasonal order
        model = auto_arima(train_data['Quantity'], seasonal=True, m=4)
        order = model.order
        seasonal_order = model.seasonal_order

        # Train the SARIMA model
        sarima_model = sm.tsa.SARIMAX(train_data['Quantity'], order=order, seasonal_order=seasonal_order)
        sarima_model_fit = sarima_model.fit()

        # Generate predictions on the test set
        predictions = sarima_model_fit.predict(start=test_data.index[0], end=test_data.index[-1])

        # Calculate accuracy statistics
        mae = mean_absolute_error(test_data['Quantity'], predictions)
        mse = mean_squared_error(test_data['Quantity'], predictions)
        rmse = np.sqrt(mse)
        
        st.subheader(product_to_predict + " Sales Predictions")
        act_pred_tab, final_app_tab = st.tabs(["📒 Actual vs Predicted", "📊 Final App"])

        col1, col2 = st.columns(2)

        with act_pred_tab:
            data = pd.concat([pred_resampled_data['Quantity'].rename('Actual'), predictions.rename('Predicted')], axis=1).reset_index()

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
        
        with final_app_tab:
            # Use auto-SARIMA to determine the order and seasonal order
            model_final = auto_arima(pred_resampled_data['Quantity'], seasonal=True, m=4)
            order_final = model_final.order
            seasonal_order_final = model_final.seasonal_order

            # Train the SARIMA model
            sarima_model_final = sm.tsa.SARIMAX(pred_resampled_data['Quantity'], order=order_final, seasonal_order=seasonal_order_final)
            sarima_model_fit_final = sarima_model_final.fit()

            # 
            match predict_interval:
                case "1 Week":
                    offset = pd.offsets.DateOffset(weeks=1)
                case "2 Weeks":
                    offset = pd.offsets.DateOffset(weeks=2)
                case "3 weeks":
                    offset = pd.offsets.DateOffset(weeks=3)
                case "1 Month":
                    offset = pd.offsets.DateOffset(months=1)
            
            # Create predictions start and end date variables
            pred_start = pred_resampled_data.index[-1]
            pred_end = pred_start + offset

            # Generate predictions            
            predictions_final = sarima_model_fit_final.predict(start=pred_start, end=pred_end)
                        
            data = pd.concat([pred_resampled_data['Quantity'].rename('Actual'), predictions_final.rename('Predicted')], axis=1).reset_index()

            chart = alt.Chart(data).mark_line().encode(
                x='index:T',
                y=alt.Y('value:Q', axis=alt.Axis(title='Quantity')),
                color=alt.Color('data:N', scale=alt.Scale(domain=['Actual', 'Predicted'], range=['steelblue', 'orange'])),
                tooltip=['index:T', 'value:Q', 'data:N']
            ).transform_fold(
                fold=['Actual', 'Predicted'],
                as_=['data', 'value']
            ).properties(
                title=f"Sales Predictions for {product_to_predict} in {predict_interval}",
                width=600,
                height=400
            )

            st.altair_chart(chart)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Auto-ARIMA parameters**")
            st.write("Order: ", order)
            st.write("Seasonal Order: ", seasonal_order)
        with col2:        
            st.write("**Accuracy Results**")
            st.write("MAE: ", mae)
            st.write("RMSE: ", rmse)

        extra_info_expander = st.expander("See Extra Information")
        with extra_info_expander:
            st.write("**Product Sales Aggregated Dataset**")
            st.write(product_to_predict)
            st.dataframe(pred_resampled_data)
            
            # Perform ACF and PACF analysis
            fig, ax = plt.subplots(2, 1, figsize=(10, 8))
            plot_acf(pred_resampled_data['Quantity'], lags=20, ax=ax[0])
            plot_pacf(pred_resampled_data['Quantity'], lags=10, ax=ax[1])
            ax[0].set_title(f"ACF Plot - {product_to_predict}")
            ax[1].set_title(f"PACF Plot - {product_to_predict}")
            plt.tight_layout()
            st.write("**ACF and PCF Plot**")
            st.pyplot(fig)

            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Train Set")
                st.dataframe(train_data)
            with col2:
                st.write("Test Set")
                st.dataframe(test_data)

