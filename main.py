import streamlit as st
import pandas as pd
import numpy as np


with st.sidebar:
    st.title("Pharmalytics")
    screen = st.radio("Navigation", ["Home", "Accounting Analytics", "Sales Prediction"])

st.title("Sales Prediction System for Firstmed Pharmacy")
st.write("Welcome to Pharmalytics!")
