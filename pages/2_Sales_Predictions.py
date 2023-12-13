import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet


# Function to create line plot visualization
def altair_chart(actual, pred, product_name):
    data = pd.concat([actual['Quantity'].rename('Actual'), pred['yhat'].rename('Predicted')], axis=1).reset_index()

    chart = alt.Chart(data).mark_line().encode(
        x='index:T',
        y=alt.Y('value:Q', axis=alt.Axis(title='Quantity')),
        color=alt.Color('data:N', scale=alt.Scale(domain=['Actual', 'Predicted'], range=['steelblue', 'orange'])),
        tooltip=['index:T', 'value:Q', 'data:N']
    ).transform_fold(
        fold=['Actual', 'Predicted'],
        as_=['data', 'value']
    ).properties(
        title=f"{product_name} Actual vs Predicted",
        width=600,
        height=400
    )

    return chart


# Set page title and icon
st.set_page_config(
    page_title="Sales Predictions",
    page_icon="💰",
    layout ="wide"
)

# Main content
st.title("Sales Predictions 💰")
st.markdown(
    """
    Pharmalytics uses the Prophet model to generate sales predictions based on your uploaded dataset. 
    This advanced technique captures seasonal and trend patterns in the data to forecast future trends.

    To ensure efficiency and practicality, our app focuses on predicting sales for the top 10% most sold products 
    in FirstMed Pharmacy and leverages the power of the Prophet model for predicting sales.
    
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

# Get unique product names from top_products
unique_product_names = top_products.unique().tolist()

# Dictionary to store predictions for each product
product_data = {}
product_predictions_matrix = []


generate_button = st.button("Generate", help="Click to generate sales predictions")

if generate_button is not True:
    st.stop()
    
with st.spinner('Processing Predictions...'):
    for product in unique_product_names:
        # Get the data for product
        pred_product_data = preprocessed_dataset[preprocessed_dataset["Product Name"] == product]
        pred_resampled_data = pred_product_data.drop(["Product Name", "Sell Price", "Product Category"], axis=1).resample(time_interval).sum().reindex(date_range, fill_value=0).reset_index()
        pred_resampled_data = pred_resampled_data.set_index("index")

        # Remove outliers using the IQR method
        Q1 = pred_resampled_data['Quantity'].quantile(0.05)
        Q3 = pred_resampled_data['Quantity'].quantile(0.80)
        IQR = Q3 - Q1

        pred_processed_data = pred_resampled_data[~((pred_resampled_data['Quantity'] < (Q1 - 1.5 * IQR)) | (pred_resampled_data['Quantity'] > (Q3 + 1.5 * IQR)))]
        
        # Prophet expects a specific format for the input DataFrame            
        pred_resampled_data_prophet = pred_processed_data.reset_index().rename(columns={"index": "ds", "Quantity": "y"})        

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
        model = model.fit(pred_resampled_data_prophet)

        # Create a DataFrame with the future dates for prediction
        future = model.make_future_dataframe(periods=12, freq=time_interval)

        # Generate predictions on the future dates
        forecast = model.predict(future)

        # Extract predictions
        predictions = forecast[['ds', 'yhat']]
        predictions['ds'] = pd.to_datetime(predictions['ds'])
        predictions = predictions.set_index('ds')

        # Set values less than zero in 'yhat' column to zero
        predictions['yhat'] = predictions['yhat'].clip(lower=0)

        # Extract predictions for only future periods
        future_predictions = predictions[predictions.index > pred_resampled_data.index.max()]['yhat']

        product_data[product] = {'actual':pred_resampled_data, 'predicted':predictions}

        # Take the intersection of dates
        common_dates = pred_resampled_data.index.intersection(predictions.index)

        # Calculate Mean Absolute Error (MAE) using common dates
        mape = mean_absolute_error(pred_resampled_data.loc[common_dates, 'Quantity'], predictions.loc[common_dates, 'yhat'])

        # Create list for product predictions
        product_predictions = []
        product_predictions.append(product)
        product_predictions.append(mape)

        for pred in future_predictions:
            product_predictions.append(pred)
        
        # Append the list to the matrix
        product_predictions_matrix.append(product_predictions)
        
    # Convert matrix to dataframe
    predictions_df = pd.DataFrame(product_predictions_matrix)

    # Extract future dates
    future_dates = forecast[forecast['ds'] > pred_resampled_data.index.max()]['ds'].dt.date.tolist()

    # Rename columns
    predictions_df.rename(columns={0: "Products", 1: "MAE"}, inplace=True)
    predictions_df.rename(columns={i: future_dates[i - 2] for i in range(2, len(future_dates) + 2)}, inplace=True)
    predictions_df.index += 1

    # Add a new column containing the sum of each row
    predictions_df['Total'] = predictions_df.iloc[:, 2:].sum(axis=1)

    st.subheader("Sales Predictions for Top 10% Products")
    descrip_exp = st.expander("See Extra Information")
    descrip_exp.markdown(
        """
        Sales predictions are computed on a weekly basis, forecasting the next 12 weeks 
        beyond the latest date in the dataset. Model accuracy is assessed using Mean 
        Absolute Error (MAE), and the products are ranked based on sales performance. 
        The 'Total' column reflects the cumulative predicted sales for the upcoming 12 weeks.
    """
    )
    st.dataframe(predictions_df)


    # Create all visualizations
    st.subheader("Sales Prediction Visualization")
    
    i = 1
    for product in unique_product_names:
        actual = product_data[product]['actual']
        pred = product_data[product]['predicted']

        product_name_count = f"{i}. {product}"
        i += 1

        chart = altair_chart(actual, pred, product_name_count)

        col1, col2 = st.columns(2)

        with col1:
            st.altair_chart(chart)
        with col2:            
            # Display the row from predictions_df for the current product
            product_row = predictions_df[predictions_df['Products'] == product].iloc[0]
            product_data_frame = pd.DataFrame(product_row[2:].values, index=product_row[2:].index, columns=['Predictions'])
            st.dataframe(product_data_frame)