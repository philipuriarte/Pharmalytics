import streamlit as st
import pandas as pd
import altair as alt
import os
from utils import total_analytics, top_analytics, altair_chart


def process_data(selected_products, product_sales_dataset, time_interval, date_range):
    processed_datasets = []
    for product in selected_products:
        resampled_data = product_sales_dataset[product_sales_dataset["Product Name"] == product].resample(time_interval).sum().fillna(0)
        resampled_data = resampled_data.drop("Product Category", axis=1)
        resampled_data["Product Name"] = product
        
        expanded_data = pd.DataFrame(data=date_range, columns=["Date Sold"])
        expanded_data = expanded_data.merge(resampled_data, on="Date Sold", how="left").fillna(0)
        expanded_data["Product Name"] = product
        
        processed_datasets.append(expanded_data)
    
    processed_dataset = pd.concat(processed_datasets)

    return processed_dataset


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
preprocessed_dataset_exists = os.path.exists(preprocessed_dataset_path)

if not preprocessed_dataset_exists:
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

    # Setup Multiselect box to choose products
    product_names = sorted(preprocessed_dataset["Product Name"].unique())
    preselected_products = ["Biogesic 500mg", "Bioflu 10mg/2mg/500mg", "Cetirizine 10mg", "Losartan 50mg"]
    selected_products = st.multiselect("Select products", product_names, default=preselected_products, max_selections=5)

    # Radio buttons to choose the time interval
    time_interval = st.radio("Select Time Interval", ["Daily", "Weekly", "Monthly"])

    if time_interval == "Daily":
        time_interval = "D"
    elif time_interval == "Weekly":
        time_interval = "W-MON"
    elif time_interval == "Monthly":
        time_interval = "M"

    # Get the minimum and maximum dates from the filtered dataset and set to beginning and end of the months respectively
    min_date = pd.Timestamp(preprocessed_dataset.index.min().date().replace(day=1))
    max_date = preprocessed_dataset.index.max().date() + pd.offsets.MonthEnd(0)    
    date_range = pd.date_range(start=min_date, end=max_date, freq=time_interval)

    # Filter the dataset for the selected products
    product_sales_dataset = preprocessed_dataset[preprocessed_dataset["Product Name"].isin(selected_products)]

    if len(selected_products) > 0:
        processed_dataset = process_data(selected_products, product_sales_dataset, time_interval, date_range)

        sales_trend_chart = alt.Chart(processed_dataset).mark_line().encode(
            x=alt.X("Date Sold:T", axis=alt.Axis(format="%b %d, %Y")),  # Format x-axis to display month, day, year
            y="Quantity:Q",
            color="Product Name:N",
            tooltip=["Date Sold:T", "Product Name:N", "Quantity:Q"]  # Include date, product name, and quantity in tooltip
        )
        
        st.altair_chart(sales_trend_chart, use_container_width=True)

        with st.expander("See More Information"):
            st.write("**Aggregated Data for Selected Products**")
            st.write(selected_products)
            st.dataframe(processed_dataset)
    else:
        st.warning("No data available for the selected products.")

with total_sales_con:
    st.subheader("Total Sales Per Product")
    total_sales_data_tab, total_sales_chart_tab = st.tabs(["📒 Data", "📊 Bar Chart"])

    with total_sales_data_tab:
        sales_df = total_analytics(preprocessed_dataset, "Quantity")
        st.dataframe(sales_df)

    with total_sales_chart_tab:
        sorted_sales_df = sales_df.sort_values("Quantity", ascending=False)
        sales_chart = altair_chart(sorted_sales_df, "Product Name", "Quantity")
        st.altair_chart(sales_chart, use_container_width=True)

with total_rev_con:
    st.subheader("Total Revenue Per Product")
    total_rev_data_tab, total_rev_chart_tab = st.tabs(["📒 Data", "📊 Bar Chart"])

    with total_rev_data_tab:
        revenue_df = total_analytics(preprocessed_dataset, "Sell Price")
        st.dataframe(revenue_df)

    with total_rev_chart_tab:
        sorted_rev_df = revenue_df.sort_values("Sell Price", ascending=False)
        revenue_chart = altair_chart(sorted_rev_df, "Product Name", "Sell Price")
        st.altair_chart(revenue_chart, use_container_width=True)

with top_sales_con:
    st.subheader("Top 30 Products With Highest Sales")
    top_sales_data_tab, top_sales_chart_tab = st.tabs(["📒 Data", "📊 Bar Chart"])

    with top_sales_data_tab:
        top_products_sales = top_analytics(sales_df, "Quantity", 30)
        st.dataframe(top_products_sales)

    with top_sales_chart_tab:
        top_sales_chart = altair_chart(top_products_sales, "Product Name", "Quantity")
        st.altair_chart(top_sales_chart, use_container_width=True)

with top_rev_con:
    st.subheader("Top 30 Products With Highest Revenue")
    top_rev_data_tab, top_rev_chart_tab = st.tabs(["📒 Data", "📊 Bar Chart"])

    with top_rev_data_tab:
        top_products_rev = top_analytics(revenue_df, "Sell Price", 30)
        st.dataframe(top_products_rev)

    with top_rev_chart_tab:
        top_revenue_chart = altair_chart(top_products_rev, "Product Name", "Sell Price")
        st.altair_chart(top_revenue_chart, use_container_width=True)

with top_cat_con:
    st.subheader("Top Sales & Revenue Data Per Category")

    # Get unique preduct names from the dataset
    categories = sorted(preprocessed_dataset["Product Category"].unique())

    # Create a select box for selecting the category
    selected_category = st.selectbox("Select Category", categories)

    # Filter the dataset for the selected category
    category_data = preprocessed_dataset[preprocessed_dataset["Product Category"] == selected_category]

    st.write("**Top 20 products with highest sales per category**")
    cat_sales_data_tab, cat_sales_chart_tab = st.tabs(["📒 Data", "📊 Bar Chart"])

    with cat_sales_data_tab:
        # Group the filtered dataset by product name and get the sum of the quantity
        cat_sales_df = category_data.groupby("Product Name")["Quantity"].sum()
        top_products_sales_cat = top_analytics(cat_sales_df, None, 20)
        st.dataframe(top_products_sales_cat)

    with cat_sales_chart_tab:
        top_sales_cat_chart = altair_chart(top_products_sales_cat, "Product Name", "Quantity")
        st.altair_chart(top_sales_cat_chart, use_container_width=True)

    st.write("**Top 20 products with highest revenue per category**")
    cat_rev_data_tab, cat_rev_chart_tab = st.tabs(["📒 Data", "📊 Bar Chart"])

    with cat_rev_data_tab:
        # Group the filtered dataset by product name and get the sum of the quantity
        cat_rev_df = category_data.groupby("Product Name")["Sell Price"].sum()
        top_products_rev_cat = top_analytics(cat_rev_df, None, 20)
        st.dataframe(top_products_rev_cat)

    with cat_rev_chart_tab:
        top_rev_cat_chart = altair_chart(top_products_rev_cat, "Product Name", "Sell Price")
        st.altair_chart(top_rev_cat_chart, use_container_width=True)

with cat_rank_con:
    st.subheader("Category Ranking")
    
    st.write("**Top categories with highest sales per category**")
    cat_sales_rank_data_tab, cat_sales_rank_chart_tab = st.tabs(["📒 Data", "📊 Bar Chart"])

    with cat_sales_rank_data_tab:
        category_sales_ranking = preprocessed_dataset.groupby("Product Category")["Quantity"].sum().reset_index()
        category_sales_ranking = top_analytics(category_sales_ranking, "Quantity", len(category_sales_ranking))
        st.dataframe(category_sales_ranking)

    with cat_sales_rank_chart_tab:
        cat_sales_rank_chart = altair_chart(category_sales_ranking, "Product Category", "Quantity")
        st.altair_chart(cat_sales_rank_chart, use_container_width=True)
    
    st.write("**Top categories with highest revenue per category**")
    cat_rev_rank_data_tab, cat_rev_rank_chart_tab = st.tabs(["📒 Data", "📊 Bar Chart"])

    with cat_rev_rank_data_tab:
        category_rev_ranking = preprocessed_dataset.groupby("Product Category")["Sell Price"].sum().reset_index()
        category_rev_ranking = top_analytics(category_rev_ranking, "Sell Price", len(category_rev_ranking))
        st.dataframe(category_rev_ranking)

    with cat_rev_rank_chart_tab:
        cat_rev_rank_chart = altair_chart(category_rev_ranking, "Product Category", "Sell Price")
        st.altair_chart(cat_rev_rank_chart, use_container_width=True)
