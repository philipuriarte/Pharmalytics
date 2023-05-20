import streamlit as st
import pandas as pd
from streamlit_extras.app_logo import add_logo


# Set page title and icon
st.set_page_config(
    page_title="Home",
    page_icon="🏠",
)

# Add the logo to the sidebar
add_logo("logo.png", height=10)

# Main content
st.title("Home")
st.write("#### Pharmalytics is a sales prediction application developed for FirstMed Pharmacy.")
file = st.file_uploader("Upload Pharmacy Sales Dataset")
if file:
    dataset = pd.read_csv(file, index_col=None)
    st.write("Dataset Preview:")
    st.dataframe(dataset)
