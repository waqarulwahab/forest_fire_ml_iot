import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


#______________________________________________Without TUNE Outliers______________________________________________

def outliers_scatter_plot(data):
    fig = px.scatter(
        data,
        x='Actual',
        y='Predicted',
        color=data['Outlier'].map({True: 'Outlier', False: 'Not Outlier'}),
        labels={'color': 'Outlier Status'},
        title='Actual vs Predicted with Outliers',
        color_discrete_map={'Outlier': 'red', 'Not Outlier': 'green'},
        size_max=10 
    )
    fig.update_traces(marker=dict(size=12))
    fig.add_shape(
        type="line",
        x0=data['Actual'].min(), y0=data['Actual'].min(),
        x1=data['Actual'].max(), y1=data['Actual'].max(),
        line=dict(dash="dash")
    )
    st.plotly_chart(fig)


def residuals_distribution_plot(residuals_df):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=residuals_df[residuals_df['Outlier'] == False]['Residuals'],
        name='Not Outlier',
        marker_color='green',
        opacity=0.75
    ))
    fig.add_trace(go.Histogram(
        x=residuals_df[residuals_df['Outlier'] == True]['Residuals'],
        name='Outlier',
        marker_color='red',
        opacity=0.75
    ))
    fig.update_layout(
        title='Distribution of Residuals with Outliers',
        xaxis_title='Residuals',
        yaxis_title='Count',
        barmode='overlay'
    )
    fig.update_traces(opacity=0.75)
    st.plotly_chart(fig)



def residuals_vs_actual_plot(residuals_df):
    fig = px.scatter(
        residuals_df,
        x='Actual',
        y='Residuals',
        color=residuals_df['Outlier'].map({True: 'Outlier', False: 'Not Outlier'}),
        labels={'color': 'Outlier Status'},
        title='Actual vs Residuals with Outliers',
        color_discrete_map={'Outlier': 'red', 'Not Outlier': 'green'},
        size_max=10
    )
    fig.update_traces(marker=dict(size=12))
    fig.add_shape(
        type="line",
        x0=residuals_df['Actual'].min(), y0=0,
        x1=residuals_df['Actual'].max(), y1=0,
        line=dict(dash="dash")
    )
    st.plotly_chart(fig)

def outliers_percentage_plot(residuals_df):
    total_points   = len(residuals_df)
    outliers_count = residuals_df['Outlier'].sum()   
    outliers_percentage = (outliers_count / total_points) * 100
    data = {
        'Type': ['Outliers', 'Non-Outliers'],
        'Count': [outliers_count, total_points - outliers_count]
    }
    df = pd.DataFrame(data)
    fig = px.pie(df, values='Count', names='Type', title='Percentage of Outliers', 
                 color_discrete_map={'Outliers': 'red', 'Non-Outliers': 'green'})
    fig.update_traces(marker=dict(colors=['red', 'green']))
    st.plotly_chart(fig)



#______________________________________________With TUNE Outliers______________________________________________



def plot_tune_outliers(data):
    y_test = data['y_test']
    y_test_pred = data['y_test_pred']
    
    residuals = y_test - y_test_pred
    outliers = np.abs(residuals) > 1.5 * np.std(residuals)
    
    fig = go.Figure()
    
    # Actual values (green)
    fig.add_trace(go.Scatter(
        x=y_test[~outliers],
        y=y_test_pred[~outliers],
        mode='markers',
        marker=dict(color='green'),
        name='Actual'
    ))
    
    # Predicted values (red)
    fig.add_trace(go.Scatter(
        x=y_test[outliers],
        y=y_test_pred[outliers],
        mode='markers',
        marker=dict(color='red'),
        name='Predicted'
    ))
    
    fig.update_layout(
        title='Actual vs Predicted',
        xaxis_title='Actual',
        yaxis_title='Predicted'
    )
    
    st.plotly_chart(fig)

# Function to create a pie chart for correct and incorrect predictions
def plot_prediction_accuracy(data):
    y_test = data['y_test']
    y_test_pred = data['y_test_pred']
    
    correct_predictions = np.sum(np.isclose(y_test, y_test_pred, atol=1.5 * np.std(y_test - y_test_pred)))
    incorrect_predictions = len(y_test) - correct_predictions
    
    labels = ['Correct Predictions', 'Incorrect Predictions']
    values = [correct_predictions, incorrect_predictions]
    colors = ['green', 'red']
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3, marker=dict(colors=colors))])
    fig.update_layout(
        title_text='Prediction Accuracy',
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    st.plotly_chart(fig)







