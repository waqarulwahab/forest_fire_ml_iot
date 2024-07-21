import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def overfitting_plot_bar(y_train, y_test, y_train_pred, y_test_pred):
    # Print the types and contents of the inputs
    print("y_train:", y_train, type(y_train))
    print("y_test:", y_test, type(y_test))
    print("y_train_pred:", y_train_pred, type(y_train_pred))
    print("y_test_pred:", y_test_pred, type(y_test_pred))
    
    # Convert inputs to numpy arrays
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    y_train_pred = np.asarray(y_train_pred)
    y_test_pred = np.asarray(y_test_pred)

    # Ensure the inputs are 1-dimensional arrays
    if y_train.ndim != 1 or y_test.ndim != 1 or y_train_pred.ndim != 1 or y_test_pred.ndim != 1:
        raise ValueError("All inputs must be 1-dimensional array-like objects.")
    
    # Calculate metrics
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    
    # Create DataFrame for metrics
    metrics = pd.DataFrame({
        'Dataset': ['Training', 'Test'],
        'R-squared': [r2_train, r2_test],
        'MSE': [mse_train, mse_test]
    })
    
    # Plot bar chart
    fig = px.bar(metrics, x='Dataset', y=['R-squared', 'MSE'], barmode='group',
                 title='Training vs Test Performance Metrics')
    st.plotly_chart(fig)

def overfitting_plot_line(y_train, y_test, y_train_pred, y_test_pred):
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    metrics = pd.DataFrame({
        'Metric': ['R-squared', 'R-squared', 'MSE', 'MSE'],
        'Dataset': ['Training', 'Test', 'Training', 'Test'],
        'Value': [r2_train, r2_test, mse_train, mse_test]
    })
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=metrics['Dataset'][metrics['Metric'] == 'R-squared'],
                             y=metrics['Value'][metrics['Metric'] == 'R-squared'],
                             mode='lines+markers',
                             name='R-squared'))
    fig.add_trace(go.Scatter(x=metrics['Dataset'][metrics['Metric'] == 'MSE'],
                             y=metrics['Value'][metrics['Metric'] == 'MSE'],
                             mode='lines+markers',
                             name='MSE'))
    fig.update_layout(
        title='Training vs Test Performance Metrics',
        xaxis_title='Dataset',
        yaxis_title='Value',
        legend_title='Metric'
    )
    st.plotly_chart(fig)


def after_overfitting_plot_(r2_train_scores, r2_test_scores, mse_train_scores, mse_test_scores):
    if isinstance(r2_train_scores, np.float64):
        r2_train_scores = [r2_train_scores]
    if isinstance(r2_test_scores, np.float64):
        r2_test_scores = [r2_test_scores]
    if isinstance(mse_train_scores, np.float64):
        mse_train_scores = [mse_train_scores]
    if isinstance(mse_test_scores, np.float64):
        mse_test_scores = [mse_test_scores]

    metrics = pd.DataFrame({
        'Metric': ['R-squared'] * 2 * len(r2_train_scores) + ['MSE'] * 2 * len(mse_train_scores),
        'Dataset': ['Training'] * len(r2_train_scores) + ['Test'] * len(r2_test_scores) +
                   ['Training'] * len(mse_train_scores) + ['Test'] * len(mse_test_scores),
        'Value': r2_train_scores + r2_test_scores + mse_train_scores + mse_test_scores
    })
    min_mse_train = min(mse_train_scores)
    min_mse_test = min(mse_test_scores)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=metrics['Dataset'][metrics['Metric'] == 'R-squared'],
                             y=metrics['Value'][metrics['Metric'] == 'R-squared'],
                             mode='lines+markers',
                             name='R-squared'))
    fig.add_trace(go.Scatter(x=metrics['Dataset'][metrics['Metric'] == 'MSE'],
                             y=metrics['Value'][metrics['Metric'] == 'MSE'],
                             mode='lines+markers',
                             name='MSE'))
    fig.add_shape(
        type="line",
        x0='Training', y0=min_mse_train,
        x1='Test', y1=min_mse_test,
        line=dict(dash="dash", color="red")
    )
    fig.update_layout(
        title='Training vs Test Performance Metrics with K-Fold Cross-Validation',
        xaxis_title='Dataset',
        yaxis_title='Value',
        legend_title='Metric'
    )
    st.plotly_chart(fig)

def over_fitting_without_tune(metrics_train, metrics_test):
    metrics_names = list(metrics_train.keys())
    train_values = list(metrics_train.values())
    test_values = list(metrics_test.values())

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=metrics_names,
        y=train_values,
        name='Train',
        marker_color='indianred'
    ))

    fig.add_trace(go.Bar(
        x=metrics_names,
        y=test_values,
        name='Test',
        marker_color='lightsalmon'
    ))

    fig.update_layout(
        title='Model Performance Metrics',
        xaxis=dict(title='Metrics'),
        yaxis=dict(title='Scores'),
        barmode='group'
    )

    st.plotly_chart(fig)



#___________________________________________________________________________________________________

def tune_overfitting(mse_train_score, mse_test_score):
    # Create subplots
    fig = go.Figure()

    # Add training and testing MSE scores
    fig.add_trace(go.Scatter(x=np.arange(len(mse_train_score)), y=mse_train_score,
                            mode='lines+markers', name='Training MSE'))
    fig.add_trace(go.Scatter(x=np.arange(len(mse_test_score)), y=mse_test_score,
                            mode='lines+markers', name='Testing MSE'))

    # Update layout
    fig.update_layout(
        title="Training vs Testing MSE",
        xaxis_title="Epoch",
        yaxis_title="MSE",
        legend_title="Legend"
    )

    # Show plot
    st.plotly_chart(fig)