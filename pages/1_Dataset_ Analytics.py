import streamlit as st
import pandas as pd
import numpy as np
import os

# Set page title and icon
st.set_page_config(
    page_title="Dataset Analytics",
    page_icon="📈",
)

st.sidebar.header("Dataset Analytics")

# Main content
st.title("Dataset Analytics 📈")

# Load the previously uploaded dataset (if exists)
uploaded_dataset = None
if os.path.exists("uploaded_dataset.csv"):
    uploaded_dataset = pd.read_csv("uploaded_dataset.csv", index_col=None)

# Drop unwanted columns
if uploaded_dataset is not None:
    uploaded_dataset.drop(['Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7'], axis=1, inplace=True)

# Replace all occurrences of '#REF!' with NaN (because of auto-fill category in Google Sheet)
uploaded_dataset.replace('#REF!', np.nan, inplace=True)

# Drop all rows that contain NaN values (All rows that have a single NaN value will be dropped)
uploaded_dataset.dropna(inplace=True)

# Show Cleaned Dataset
st.subheader("Cleaned Dataset")
st.write("All rows with at least 1 empty cell are removed in the uploaded dataset.")
st.dataframe(uploaded_dataset, width=700)


# TABS for total quantity of sales per product
st.subheader("Total Sales Per Product")
total_sales_data_tab, total_sales_chart_tab = st.tabs(["📒 Data", "📊 Bar Chart"])

with total_sales_data_tab:
    # Group the dataframe by product name and get the sum of the quantity
    quantity_per_product = uploaded_dataset.groupby(["Product Name"])["Quantity"].sum()
    # Convert the resulting series to a dataframe
    quantity_df = pd.DataFrame(quantity_per_product).reset_index()
    st.dataframe(quantity_df, width=400)

with total_sales_chart_tab:
    st.bar_chart(quantity_df, x="Product Name", y="Quantity")


# TABS for total quantity of sales per product
st.subheader("Total Revenue Per Product")
total_rev_data_tab, total_rev_chart_tab = st.tabs(["📒 Data", "📊 Bar Chart"])

with total_rev_data_tab:
    # Group the dataframe by product name and get the sum of the sell price
    sell_price_per_product = uploaded_dataset.groupby(["Product Name"])["Sell Price"].sum()
    # Convert the resulting series to a dataframe
    revenue_df = pd.DataFrame(sell_price_per_product).reset_index()
    st.dataframe(revenue_df)

with total_rev_chart_tab:
    st.bar_chart(revenue_df, x="Product Name", y="Sell Price")


# TABS for top 30 most sold products
st.subheader('Top 30 Most Sold Products')
top_sales_data_tab, top_sales_chart_tab = st.tabs(["📒 Data", "📊 Bar Chart"])

with top_sales_data_tab:
    # Get the top 30 most sold products
    top_30_products_sales = quantity_df.sort_values("Quantity", ascending=False).head(30).reset_index()
    top_30_products_sales.index += 1 # Start with index 1 instead 0
    top_30_products_sales = top_30_products_sales.drop('index', axis=1) # Remove Index column
    st.dataframe(top_30_products_sales)

# Plot in bar graph
with top_sales_chart_tab:
    st.bar_chart(top_30_products_sales, x="Product Name", y="Quantity")


# TABS for top 30 products with highest revenue
st.subheader('Top 30 Products With Highest Revenue')
top_rev_data_tab, top_rev_chart_tab = st.tabs(["📒 Data", "📊 Bar Chart"])

with top_rev_data_tab:
    # Get the top 30 products with highest revenue
    top_30_products_rev = revenue_df.sort_values("Sell Price", ascending=False).head(30).reset_index()
    top_30_products_rev.index += 1 # Start with index 1 instead 0
    top_30_products_rev = top_30_products_rev.drop('index', axis=1) # Remove Index column
    st.dataframe(top_30_products_rev)

# Plot in bar graph
with top_rev_chart_tab:
    st.bar_chart(top_30_products_rev, x="Product Name", y="Sell Price")


st.subheader("Top Sales & Revenue Data Per Category")
categories = [
    "Adult Vitamins and Supplements",
    "All Hypertension",
    "All Medical Supplies",
    "Anthelmintic",
    "Anti Tb",
    "Antibiotics",
    "Antigout/hyperthyroids",
    "Antihistamines",
    "Antipsychotics",
    "Antivertigo",
    "Baby Products",
    "Coffee Teas and Milks",
    "Contraceptives",
    "Cosmetics",
    "Cream and Ointments",
    "Drinks",
    "Foods",
    "Galenicals",
    "Gastro Drugs",
    "Herbal Medicines",
    "Hygiene",
    "Maintainance",
    "Milk Supplements",
    "Ob Products",
    "Others",
    "Otic/ophthalmic",
    "Pain Relievers",
    "Pedia Bottles",
    "Respiratory"
]

# Create a select box for selecting the category
selected_category = st.selectbox("Select Category", categories)

# Filter the dataset for the selected category
category_data = uploaded_dataset[uploaded_dataset["Product Category"] == selected_category.upper()]

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
    st.bar_chart(top_10_products_sales_cat, x="Product Name", y="Quantity")


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
    st.bar_chart(top_10_products_rev_cat, x="Product Name", y="Sell Price")


st.subheader("Product Sales Trend Over Time")

# Get unique preduct names from the dataset
product_names = uploaded_dataset["Product Name"].unique()

# Select box to choose a product
selected_product = st.selectbox("Select a Product", product_names)

# Filter the dataset for the selected product
product_sales_dataset = uploaded_dataset[uploaded_dataset["Product Name"] == selected_product]

# Convert the "Date Sold" column to datetime format
product_sales_dataset["Date Sold"] = pd.to_datetime(product_sales_dataset["Date Sold"], dayfirst=True)

# Create a new DataFrame with the dates as the index
indexed_dataset = product_sales_dataset.set_index("Date Sold")

# Resample the DataFrame to include missing dates with 0 sales
resampled_dataset = indexed_dataset.resample("D").sum().fillna(0)

# Sort the resampled dataset by date in ascending order
resampled_dataset = resampled_dataset.sort_index()

# Line graph of sales for the selected product
st.line_chart(resampled_dataset["Quantity"])


st.subheader("Product Sales Trend Over Time: MULTISELECT")

# Multiselect box to choose products
selected_products = st.multiselect("Select Products", product_names)

# Filter the dataset for the selected products
# for product in selected_products:
#     products_sales_dataset = uploaded_dataset[uploaded_dataset["Product Name"] == product]
products_sales_dataset = uploaded_dataset[uploaded_dataset["Product Name"].isin(selected_products)]

# Convert the "Date Sold" column to datetime format
products_sales_dataset["Date Sold"] = pd.to_datetime(products_sales_dataset["Date Sold"], dayfirst=True)

# Create a new DataFrame with the dates as the index
multi_indexed_dataset = products_sales_dataset.set_index("Date Sold")

# Resample the DataFrame to include missing dates with 0 sales for each product
resampled_datasets = []
for product in selected_products:
    multi_resampled_data = multi_indexed_dataset[multi_indexed_dataset["Product Name"] == product].resample("D").sum().fillna(0)
    resampled_datasets.append(multi_resampled_data)

st.dataframe(products_sales_dataset)
