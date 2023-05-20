import streamlit as st
import pandas as pd
import numpy as np

# Set page title and icon
st.set_page_config(
    page_title="Dataset Analytics",
    page_icon="📈",
)

# Main content
st.title("Dataset Analytics 📈")
st.write("*option to open this screen will only show one a dataset file has been uploaded*")
st.write("*will show sales data analytics user can enter input: choose from revenue/quantity of sales, per day/week/month/season and select specific day, month, etc.*")
