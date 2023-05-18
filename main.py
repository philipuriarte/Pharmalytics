import streamlit as st
import pandas as pd
import numpy as np


with st.sidebar:
    st.image()
    st.title("Pharmalytics")
    screen = st.radio("Navigation", ["Home", "Accounting Analytics", "Sales Prediction"])
    st.info("Pharmalytics is a sales prediction application developed for FirstMed Pharmacy.")

st.title("Sales Prediction System for Firstmed Pharmacy")
st.write("Welcome to Pharmalytics!")
