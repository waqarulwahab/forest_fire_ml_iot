import streamlit as st
import pandas as pd

from logics.train_model import *
from plots.outliers import *


def outliers_with_tuning(y_test, y_pred):
    pass




def  outliers_without_tuning(y_test, y_pred):
        st.divider()
        lower_percentile = st.sidebar.slider('Select lower percentile', 0, 100, 25, key="lower_percentile")
        upper_percentile = st.sidebar.slider('Select upper percentile', 0, 100, 75, key="upper_percentile")
        residuals, lower_bound, upper_bound = calculate_residuals(y_test, y_pred, 
                                                                    lower_percentile, upper_percentile)
        residuals_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Residuals': residuals})
        residuals_df['Outlier'] = (residuals_df['Residuals'] < lower_bound) | (residuals_df['Residuals'] > upper_bound)

        data = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred,
            'Outlier': residuals_df['Outlier']
        })

        col1 , col2 = st.columns([1,1])
        with col1:
            outliers_scatter_plot(data)
        with col2:
            residuals_vs_actual_plot(residuals_df)

        col1 , col2 = st.columns([1,1])
        with col1:
            residuals_distribution_plot(residuals_df)
        with col2:
            outliers_percentage_plot(residuals_df)