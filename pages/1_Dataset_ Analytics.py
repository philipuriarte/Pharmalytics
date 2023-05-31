import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os

# Set page title and icon
st.set_page_config(
    page_title="Dataset Analytics",
    page_icon="📈",
)

st.sidebar.header("Dataset Analytics")

# Main content
st.title("Dataset Analytics 📈")

# Load the preprocessed dataset and stop if it doesn't exist
preprocessed_dataset_path = "preprocessed_dataset.csv"
if not os.path.exists(preprocessed_dataset_path):
    st.warning("Dataset not uploaded. Please upload a dataset first in the Home page.")
    st.stop()

preprocessed_dataset = pd.read_csv(preprocessed_dataset_path, parse_dates=["Date Sold"], index_col="Date Sold")

# Create containers to group codes together
sales_trend_con = st.container()
total_sales_con = st.container()
total_rev_con = st.container()
top_sales_con = st.container()
top_rev_con = st.container()
top_cat_con = st.container()
cat_rank_con = st.container()

with sales_trend_con:
    st.subheader("Product Sales Trend Over Time")

    # Get unique product names from the dataset
    product_names = sorted(preprocessed_dataset["Product Name"].unique())

    # Multiselect box to choose products
    selected_products = st.multiselect("Select products", product_names, max_selections=5)

    # Radio buttons to choose the time interval
    time_interval = st.radio("Select Time Interval", ["Daily", "Weekly", "Monthly"])

    # Filter the dataset for the selected products
    product_sales_dataset = preprocessed_dataset[preprocessed_dataset["Product Name"].isin(selected_products)]

    # Get the minimum and maximum dates from the filtered dataset and set to beginning and end of the months respectively
    min_date = pd.Timestamp(preprocessed_dataset.index.min().date().replace(day=1))
    max_date = preprocessed_dataset.index.max().date() + pd.offsets.MonthEnd(0)

    # Resample the DataFrame based on the selected time interval
    resampled_datasets = []
    for product in selected_products:
        if time_interval == "Daily":
            resampled_data = product_sales_dataset[product_sales_dataset["Product Name"] == product].resample("D").sum().fillna(0)
            date_range = pd.date_range(start=min_date, end=max_date, freq="D")
        elif time_interval == "Weekly":
            resampled_data = product_sales_dataset[product_sales_dataset["Product Name"] == product].resample("W-MON").sum().fillna(0)
            date_range = pd.date_range(start=min_date, end=max_date, freq="W-MON")
        elif time_interval == "Monthly":
            resampled_data = product_sales_dataset[product_sales_dataset["Product Name"] == product].resample("M").sum().fillna(0)
            date_range = pd.date_range(start=min_date, end=max_date, freq="M")
        resampled_data = resampled_data.drop("Product Category", axis=1)  # Remove the "Product Categories" column
        resampled_data["Product Name"] = product  # Add a column with the product name
        resampled_datasets.append(resampled_data)

    if len(resampled_datasets) > 0:
        # Concatenate the resampled datasets
        combined_dataset = pd.concat(resampled_datasets)

        # Sort the combined dataset by date in ascending order
        combined_dataset = combined_dataset.sort_index().reset_index()

        # Expand the combined dataset to include all dates in the range
        expanded_datasets = []
        for product in selected_products:
            product_data = combined_dataset[combined_dataset["Product Name"] == product]
            expanded_data = pd.DataFrame(data=date_range, columns=["Date Sold"])
            expanded_data = expanded_data.merge(product_data, on="Date Sold", how="left").fillna(0)
            expanded_data["Product Name"] = product
            expanded_datasets.append(expanded_data)

        expanded_dataset = pd.concat(expanded_datasets)

        # Line chart of sales for the selected products
        chart = alt.Chart(expanded_dataset).mark_line().encode(
            x=alt.X("Date Sold:T", axis=alt.Axis(format="%b %d, %Y")),  # Format x-axis to display month, day, year
            y="Quantity:Q",
            color="Product Name:N",
            tooltip=["Date Sold:T", "Product Name:N", "Quantity:Q"]  # Include date, product name, and quantity in tooltip
        ).properties(
            title={
                "text": "Product Sales Trend Over Time",
                "align": "center"
            }
        )

        # Render the chart using st.altair_chart
        st.altair_chart(chart, use_container_width=True)

        st.write("Render Dataframe for extra information in testing")
        st.dataframe(expanded_dataset)
    else:
        st.warning("No data available for the selected products.")

with total_sales_con:
    # TABS for total quantity of sales per product
    st.subheader("Total Sales Per Product")
    total_sales_data_tab, total_sales_chart_tab = st.tabs(["📒 Data", "📊 Bar Chart"])

    with total_sales_data_tab:
        # Group the dataframe by product name and get the sum of the quantity
        quantity_per_product = preprocessed_dataset.groupby(["Product Name"])["Quantity"].sum()
        # Convert the resulting series to a dataframe
        quantity_df = pd.DataFrame(quantity_per_product).reset_index()
        st.dataframe(quantity_df)

    with total_sales_chart_tab:
        # Sort the dataframe by the Quantity column in descending order
        sorted_quantity_df = quantity_df.sort_values("Quantity", ascending=False)
        # Create the Altair bar chart
        quantity_alt_chart = alt.Chart(sorted_quantity_df).mark_bar().encode(
            x=alt.X("Product Name", sort=None),  # Disable automatic sorting
            y="Quantity"
        ).properties(
            title="Total Sales Per Product From Highest to Lowest"
        )
        # Render the chart using st.altair_chart
        st.altair_chart(quantity_alt_chart, use_container_width=True)

with total_rev_con:
    # TABS for total quantity of sales per product
    st.subheader("Total Revenue Per Product")
    total_rev_data_tab, total_rev_chart_tab = st.tabs(["📒 Data", "📊 Bar Chart"])

    with total_rev_data_tab:
        # Group the dataframe by product name and get the sum of the sell price
        sell_price_per_product = preprocessed_dataset.groupby(["Product Name"])["Sell Price"].sum()
        # Convert the resulting series to a dataframe
        revenue_df = pd.DataFrame(sell_price_per_product).reset_index()
        st.dataframe(revenue_df)

    with total_rev_chart_tab:
        # Sort the dataframe by the Sell Price column in descending order
        sorted_rev_df = revenue_df.sort_values("Sell Price", ascending=False)
        # Create the Altair bar chart
        rev_alt_chart = alt.Chart(sorted_rev_df).mark_bar().encode(
            x=alt.X("Product Name", sort=None),  # Disable automatic sorting
            y="Sell Price"
        ).properties(
            title="Total Revenue Per Product From Highest to Lowest"
        )
        # Render the chart using st.altair_chart
        st.altair_chart(rev_alt_chart, use_container_width=True)

with top_sales_con:
    # TABS for top 30 most sold products
    st.subheader("Top 30 Most Sold Products")
    top_sales_data_tab, top_sales_chart_tab = st.tabs(["📒 Data", "📊 Bar Chart"])

    with top_sales_data_tab:
        # Get the top 30 most sold products
        top_30_products_sales = quantity_df.sort_values("Quantity", ascending=False).head(30).reset_index()
        top_30_products_sales.index += 1 # Start with index 1 instead 0
        top_30_products_sales = top_30_products_sales.drop("index", axis=1) # Remove Index column
        st.dataframe(top_30_products_sales)

    # Plot in bar graph
    with top_sales_chart_tab:
        top_sales_alt_chart = alt.Chart(top_30_products_sales).mark_bar().encode(
            x=alt.X("Product Name", sort=None),  # Disable automatic sorting
            y="Quantity"
        )
        st.altair_chart(top_sales_alt_chart, use_container_width=True)

with top_rev_con:
    # TABS for top 30 products with highest revenue
    st.subheader("Top 30 Products With Highest Revenue")
    top_rev_data_tab, top_rev_chart_tab = st.tabs(["📒 Data", "📊 Bar Chart"])

    with top_rev_data_tab:
        # Get the top 30 products with highest revenue
        top_30_products_rev = revenue_df.sort_values("Sell Price", ascending=False).head(30).reset_index()
        top_30_products_rev.index += 1 # Start with index 1 instead 0
        top_30_products_rev = top_30_products_rev.drop("index", axis=1) # Remove Index column
        st.dataframe(top_30_products_rev)

    # Plot in bar graph
    with top_rev_chart_tab:
        top_rev_alt_chart = alt.Chart(top_30_products_rev).mark_bar().encode(
            x=alt.X("Product Name", sort=None),  # Disable automatic sorting
            y="Sell Price"
        )
        st.altair_chart(top_sales_alt_chart, use_container_width=True)

with top_cat_con:
    st.subheader("Top Sales & Revenue Data Per Category")

    # Get unique preduct names from the dataset
    categories = sorted(preprocessed_dataset["Product Category"].unique())

    # Create a select box for selecting the category
    selected_category = st.selectbox("Select Category", categories)

    # Filter the dataset for the selected category
    category_data = preprocessed_dataset[preprocessed_dataset["Product Category"] == selected_category]

    # TABS for top 20 most sold products per category
    st.write("**Top 20 most sold products per category**")
    cat_sales_data_tab, cat_sales_chart_tab = st.tabs(["📒 Data", "📊 Bar Chart"])

    with cat_sales_data_tab:
        # Group the filtered dataset by product name and get the sum of the quantity
        quantity_per_product_cat = category_data.groupby("Product Name")["Quantity"].sum()
        # Sort the quantity per product in descending order and get the top 20
        top_10_products_sales_cat = quantity_per_product_cat.sort_values(ascending=False).head(20).reset_index()
        top_10_products_sales_cat.index += 1 # Start with index 1 instead 0
        st.dataframe(top_10_products_sales_cat)

    # Plot in bar graph
    with cat_sales_chart_tab:
        top_sales_cat_alt_chart = alt.Chart(top_10_products_sales_cat).mark_bar().encode(
            x=alt.X("Product Name", sort=None),  # Disable automatic sorting
            y="Quantity"
        )
        st.altair_chart(top_sales_cat_alt_chart, use_container_width=True)

    # TABS for top 20 products with highest revenue per category
    st.write("**Top 20 products with highest revenue per category**")
    cat_rev_data_tab, cat_rev_chart_tab = st.tabs(["📒 Data", "📊 Bar Chart"])

    with cat_rev_data_tab:
        # Group the filtered dataset by product name and get the sum of the quantity
        quantity_per_product_cat = category_data.groupby("Product Name")["Sell Price"].sum()
        # Sort the quantity per product in descending order and get the top 20
        top_10_products_rev_cat = quantity_per_product_cat.sort_values(ascending=False).head(20).reset_index()
        top_10_products_rev_cat.index += 1 # Start with index 1 instead 0
        st.dataframe(top_10_products_rev_cat)

    # Plot in bar graph
    with cat_rev_chart_tab:
        top_rev_cat_alt_chart = alt.Chart(top_10_products_rev_cat).mark_bar().encode(
            x=alt.X("Product Name", sort=None),  # Disable automatic sorting
            y="Sell Price"
        )
        st.altair_chart(top_rev_cat_alt_chart, use_container_width=True)

with cat_rank_con:
    # TABS for top 30 most sold products
    st.subheader("Category Sales Ranking")
    cat_rank_data_tab, cat_rank_chart_tab = st.tabs(["📒 Data", "📊 Bar Chart"])

    with cat_rank_data_tab:
        category_sales = preprocessed_dataset.groupby("Product Category")["Quantity"].sum().reset_index()
        category_sales = category_sales.sort_values("Quantity", ascending=False).reset_index()
        category_sales.index += 1
        category_sales = category_sales.drop("index", axis=1)
        st.dataframe(category_sales)

    # Plot in bar graph
    with cat_rank_chart_tab:
        cat_rank_alt_chart = alt.Chart(category_sales).mark_bar().encode(
            x=alt.X("Product Category", sort=None),  # Disable automatic sorting
            y="Quantity"
        )
        st.altair_chart(cat_rank_alt_chart, use_container_width=True)
