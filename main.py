import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image as img


with st.sidebar:
    logo = img.open("logo.png")
    st.image(logo)
    st.markdown("***")
    screen = st.radio("Navigation", ["Home", "Accounting Analytics", "Sales Predictions"])
    st.info("Pharmalytics is a sales prediction application developed for FirstMed Pharmacy.")


if screen == "Home":
    st.title("Home")
    file = st.file_uploader("Upload Pharmacy Sales Dataset")
    if file:
        dataset = pd.read_csv(file, index_col=None)
        st.write("Dataset Preview:")
        st.dataframe(dataset)

elif screen == "Accounting Analytics":
    st.title("Accounting Analytics")
    st.write("*option to open this screen will only show one a dataset file has been uploaded*")
    st.write("*will show sales data analytics user can enter input: choose from revenue/quantity of sales, per day/week/month/season and select specific day, month, etc.*")

elif screen == "Sales Predictions":
    st.title("Sales Predictions")
    st.write("*option to open this screen will only show one a dataset file has been uploaded*")
    st.write("*will show sales predictions from trained SARIMA model*")
