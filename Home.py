import streamlit as st
import pandas as pd
import os


def preprocess_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the dataset by dropping unnamed columns, replacing "#REF!" with None,
    dropping rows with None values, and converting the "Date Sold" column to datetime format.
    """
    # Drop unnamed columns
    unnamed_columns = [col for col in dataset.columns if 'Unnamed' in col]
    dataset.drop(unnamed_columns, axis=1, inplace=True)

    # Replace all occurrences of "#REF!" with None due to auto-fill category in Google Sheet dataset
    dataset.replace("#REF!", None, inplace=True)

    # Drop all rows that contain None values
    dataset.dropna(inplace=True)

    preprocessed_dataset = dataset.reset_index(drop=True)

    # Convert the "Date Sold" column to datetime format and set as index
    preprocessed_dataset["Date Sold"] = pd.to_datetime(preprocessed_dataset["Date Sold"], format="%m/%d/%Y")
    preprocessed_dataset = preprocessed_dataset.set_index("Date Sold")

    # Convert "Sell Price" column to numeric and then round to zero decimal places
    preprocessed_dataset["Sell Price"] = pd.to_numeric(preprocessed_dataset["Sell Price"], errors="coerce")
    preprocessed_dataset["Sell Price"] = preprocessed_dataset["Sell Price"].round(0)

    return preprocessed_dataset


def main():
    st.set_page_config(
        page_title="Home",
        page_icon="🏠",
    )

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
    
    file = st.file_uploader("Upload Sales Dataset", type="csv")
    dataset_exists = os.path.exists("uploaded_dataset.csv")

    # Load previously uploaded dataset (if exists and no new dataset is uploaded)
    if dataset_exists and file is None:
        dataset = pd.read_csv("uploaded_dataset.csv", index_col=None)
    elif file is None:
        st.warning("Please upload a CSV file.")
        st.stop()    
    else:
        dataset = pd.read_csv(file, index_col=False)
        dataset.to_csv("uploaded_dataset.csv", index=None)  # Save dataset

    st.write("**Dataset Preview:**")
    st.dataframe(dataset, width=700)

    pre_con = st.expander("Show Preprocessing Procedure")

    with pre_con:
        preprocessed_dataset = preprocess_dataset(dataset)
        preprocessed_dataset.to_csv("preprocessed_dataset.csv", date_format="%m/%d/%Y")  # Save preprocessed dataset

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


if __name__ == "__main__":
    main()
