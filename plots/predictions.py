import streamlit as st
import pandas as pd
import plotly.express as px


def prepare_results_df(y_test, y_test_pred):
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_test_pred
    })
    results_df['Correct'] = results_df['Actual'] == results_df['Predicted']
    results_df['Color'] = results_df['Correct'].map({True: 'green', False: 'red'})
    return results_df


def test_predictions(results_df, title):
    fig = px.scatter(
        results_df, 
        x=results_df.index, 
        y='Actual', 
        color='Color', 
        color_discrete_map={'green': 'green', 'red': 'red'},
        title = title,
        labels={"x": "Index", "Actual": "Actual Values"},
        hover_data=['Predicted']
    )
    fig.update_traces(marker=dict(size=10))
    st.plotly_chart(fig)


def correct_incorrect_pie_chart(correct_predictions, incorrect_predictions, title):
    data = {
        'Category': ['Correct Predictions', 'Incorrect Predictions'],
        'Count': [correct_predictions, incorrect_predictions]
    }
    color_discrete_map = {'Correct Predictions': 'green', 'Incorrect Predictions': 'red'}
    fig = px.pie(data, values='Count', names='Category', title= title, 
                 color='Category', color_discrete_map=color_discrete_map)
    st.plotly_chart(fig)


def correct_incorrect_pie_chart_combined(train_correct, train_incorrect, test_correct, test_incorrect):
    data = {
        'Category': [
            'Train Correct Predictions', 'Train Incorrect Predictions',
            'Test Correct Predictions', 'Test Incorrect Predictions'
        ],
        'Count': [train_correct, train_incorrect, test_correct, test_incorrect]
    }
    color_discrete_map = {
        'Train Correct Predictions': 'lightgreen', 
        'Train Incorrect Predictions': 'darkred',
        'Test Correct Predictions': 'green',
        'Test Incorrect Predictions': 'red'
    }
    fig = px.pie(data, values='Count', names='Category', title='Prediction Results',
                 color='Category', color_discrete_map=color_discrete_map)
    st.plotly_chart(fig)

def correct_incorrect_line_plot(train_correct, train_incorrect, test_correct, test_incorrect):
    data = {
        'Category': [
            'Train Correct Predictions', 'Train Incorrect Predictions',
            'Test Correct Predictions', 'Test Incorrect Predictions'
        ],
        'Count': [train_correct, train_incorrect, test_correct, test_incorrect]
    }
    fig = px.line(data, x='Category', y='Count', title='Prediction Results', markers=True,
                  labels={'Count': 'Number of Predictions'})
    fig.update_traces(line=dict(color='blue'), marker=dict(color='blue'))
    st.plotly_chart(fig)