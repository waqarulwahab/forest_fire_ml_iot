import streamlit as st
from plots.cards import *


def header(read_csv):
    num_rows = len(read_csv)
    data_shape  = read_csv.shape
    null_values = read_csv.isnull().sum().sum()

    col1, col2, col3, col4 = st.columns([1,1,1,1])
    with col1:
        title = "Total Columns"
        cards(title, data_shape[1])
    with col2:
        title = "Total Rows"
        cards(title, data_shape[0])
    with col3:
        title = "Null Values"
        cards(title, null_values)
    with col4:
        title = "Selected Model"
        selected_model = "Extra-Tress Regressor"
        cards(title, selected_model)