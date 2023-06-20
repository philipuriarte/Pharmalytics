import pandas as pd
import numpy as np
import altair as alt


# HOME

def preprocess_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    # Drop unnamed columns
    unnamed_columns = [col for col in dataset.columns if 'Unnamed' in col]
    dataset.drop(unnamed_columns, axis=1, inplace=True)

    # Replace all occurrences of "#REF!" with NaN (because of auto-fill category in Google Sheet)
    dataset.replace("#REF!", np.nan, inplace=True)

    # Drop all rows that contain NaN values (All rows that have a single NaN value will be dropped)
    dataset.dropna(inplace=True)

    preprocessed_dataset = dataset.reset_index(drop=True)

    # Convert the "Date Sold" column to datetime format and set as index
    preprocessed_dataset["Date Sold"] = pd.to_datetime(preprocessed_dataset["Date Sold"], format="%m/%d/%Y")
    preprocessed_dataset = preprocessed_dataset.set_index("Date Sold")

    return preprocessed_dataset


# DATASET ANALYTICS

def total_analytics(dataset: pd.DataFrame, column: str) -> pd.DataFrame:
    data_per_product = dataset.groupby(["Product Name"])[column].sum()
    quantity_df = pd.DataFrame(data_per_product).reset_index()

    return quantity_df


def top_analytics(dataset: pd.DataFrame, column: str or None, max_range: int) -> pd.DataFrame:
    if column is None:
        top_data = dataset.sort_values(ascending=False).head(max_range).reset_index() # Get top *max_range* products
    else:
        top_data = dataset.sort_values(column, ascending=False).head(max_range).reset_index() # Get top *max_range* products
        top_data = top_data.drop("index", axis=1) # Remove Index column
    
    top_data.index += 1 # Start with index 1 instead 0

    return top_data


# Generate altair bar graph
def altair_chart(dataset: pd.DataFrame, x_label: str, y_label: str) -> alt.Chart:
    chart = alt.Chart(dataset).mark_bar().encode(
        x= alt.X(x_label, sort=None),
        y= y_label
    )
    
    return chart
