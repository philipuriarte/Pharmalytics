import pandas as pd
import numpy as np

def preprocess_dataset(dataset):
    # Drop the unnamed columns
    unnamed_columns = [col for col in dataset.columns if 'Unnamed' in col]
    dataset.drop(unnamed_columns, axis=1, inplace=True)

    # Replace all occurrences of "#REF!" with NaN (because of auto-fill category in Google Sheet)
    dataset.replace("#REF!", np.nan, inplace=True)

    # Drop all rows that contain NaN values (All rows that have a single NaN value will be dropped)
    dataset.dropna(inplace=True)

    preprocessed_dataset = dataset.reset_index(drop=True)

    # Convert the "Date Sold" column to datetime format
    preprocessed_dataset["Date Sold"] = pd.to_datetime(preprocessed_dataset["Date Sold"], format="%m/%d/%Y")

    # Create a new DataFrame with the dates as the index
    preprocessed_dataset = preprocessed_dataset.set_index("Date Sold")

    return preprocessed_dataset
