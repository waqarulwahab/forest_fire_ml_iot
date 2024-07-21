import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go



def learning_rate_line_plot(train_scores, test_scores):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, 101)), y=train_scores, mode='lines', name='Train Score'))
    fig.add_trace(go.Scatter(x=list(range(1, 101)), y=test_scores, mode='lines', name='Test Score'))

    # Find the point of minimum difference between train and test scores
    differences = [abs(train - test) for train, test in zip(train_scores, test_scores)]
    min_diff_index = differences.index(min(differences)) + 1

    # Highlight the area of minimum difference
    fig.add_vline(x=min_diff_index, line=dict(color="red", width=2), annotation_text="Min Difference", annotation_position="top right")

    fig.update_layout(
        title='Learning Curve for Extra Trees Regressor',
        xaxis_title='Number of Trees',
        yaxis_title='R^2 Score'
    )

    st.plotly_chart(fig)

def tune_learning_rate(scores_df):
    # Determine the best fold based on a specific metric, e.g., highest R2 Test score
    best_fold = scores_df['R2 Test'].idxmax() + 1

    fig = px.line(scores_df, x='Fold', y=['R2 Train', 'R2 Test', 'MSE Train', 'MSE Test', 'RMSE Test', 'MAE Test'],
                  title='Model Performance Metrics Across Folds')
    
    # Add a red vertical line at the best fold
    fig.add_shape(
        go.layout.Shape(
            type="line",
            x0=best_fold,
            y0=scores_df[['R2 Train', 'R2 Test', 'MSE Train', 'MSE Test', 'RMSE Test', 'MAE Test']].min().min(),
            x1=best_fold,
            y1=scores_df[['R2 Train', 'R2 Test', 'MSE Train', 'MSE Test', 'RMSE Test', 'MAE Test']].max().max(),
            line=dict(color="Red", width=2),
        )
    )

    fig.update_layout(yaxis_title='Score', xaxis_title='Fold')
    st.plotly_chart(fig)
