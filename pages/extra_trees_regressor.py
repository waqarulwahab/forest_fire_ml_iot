import pandas as pd
from plots.cards import *
from plots.model_performance_plots import *


import streamlit as st
import pickle
from sklearn.ensemble import ExtraTreesRegressor
from io import BytesIO
import joblib
from logics.train_model import *



from plots.overfitting import *
from plots.predictions import *
from plots.learning_rate import *

from logics.logics import *
from logics.models import *
from logics.header import *
from logics.load_data import *
from logics.outliers import *


st.sidebar.page_link("main.py",                        label="OVERVIEW", icon="🏠")
st.sidebar.page_link("pages/extra_trees_regressor.py", label="MODELS",   icon="⚙️")
st.sidebar.page_link("pages/about_dataset.py",         label="DATASET",  icon="📖")


selected_kfolds = None

read_csv = load_data()

header(read_csv)
st.divider()

st.sidebar.divider()

option = st.sidebar.selectbox(
    'Select an option:',
    ('With Tuning', 'Without Tuning')
)
if option == "Without Tuning":
    option = option

elif option == "With Tuning":
    option = option
    col1, col2 = st.sidebar.columns([1,1])
    with col1:
        selected_kfolds = st.number_input("Select KFold", key='select_kfold', min_value=1, step=1)
    with col2:
        selected_nsplits = st.number_input("Select NSplits", key='selected_nsplits', min_value=1, step=1)


model_names, training_models = models()
col1, col2 = st.sidebar.columns([1,2])
with col1:
    model_names_button = st.checkbox("Models", key="Model_Names_Button")
with col2:
    show_performances_button = st.checkbox("Show Performances", key="Model_Performances_Button")


if model_names_button:
    st.sidebar.write(model_names)

try:
    if show_performances_button:    
        model_performances_file = 'CSV Files/model_performance.csv'
        model_performances      = pd.read_csv(model_performances_file)
        st.write(model_performances)
except:
    pass

model_performances = pd.read_csv('CSV Files/model_performance.csv')
all_model_performance(model_performances)
col1, col2 = st.columns([1,1])
with col1:
    extra_tree_regressor_pie(model_performances)
with col2:
    extra_tree_regressor_bar(model_performances)


if option == "Without Tuning":
    check_outliers           = st.sidebar.toggle("Check Outliers",    key="check_outliers")
    overfitting              = st.sidebar.toggle("Overfitting"   ,    key="overfitting")
    predictions              = st.sidebar.toggle("Predictions"   ,    key="prediction")
    learning_rate            = st.sidebar.toggle("Learning Rates",    key="learning_rates")

    with open('pickle_file/extra_tress_results.pkl', 'rb') as f:
        X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, y_pred, train_scores, test_scores= pickle.load(f)
    X_train      = np.array(X_train,      copy=False, subok=True)
    X_test       = np.array(X_test,       copy=False, subok=True)
    y_train      = np.array(y_train,      copy=False, subok=True)
    y_test       = np.array(y_test,       copy=False, subok=True)
    y_train_pred = np.array(y_train_pred, copy=False, subok=True)
    y_test_pred  = np.array(y_test_pred,  copy=False, subok=True)
    y_pred       = np.array(y_pred,       copy=False, subok=True)
    train_scores = np.asarray(train_scores)
    test_scores  = np.asarray(test_scores)

    if check_outliers:
        outliers_without_tuning(y_test, y_pred)

    if overfitting:
        st.divider()
        col1, col2 = st.columns([1,1])
        with col1:
            overfitting_plot_bar(y_train, y_test, y_train_pred, y_test_pred)
        with col2:
            overfitting_plot_line(y_train, y_test, y_train_pred, y_test_pred)


        metrics_train = calculate_metrics(y_train, y_train_pred)
        metrics_test  = calculate_metrics(y_test, y_test_pred)

        over_fitting_without_tune(metrics_train, metrics_test)

        if any(metrics_test[key] > metrics_train[key] for key in metrics_train):
            st.warning('Your model might be overfitting. Tune Your Model.')
        else:
            st.success('No signs of overfitting detected. Your model generalizes well.')


    if predictions:

        st.divider()
        train_correct   = sum(y_train == y_train_pred)
        train_incorrect = sum(y_train != y_train_pred)
        test_correct    = sum(y_test  == y_test_pred)
        test_incorrect  = sum(y_test  != y_test_pred)

        col1, col2 = st.columns([1,1])
        with col1:
            correct_incorrect_line_plot(train_correct, train_incorrect, test_correct, test_incorrect)
        with col2:
            correct_incorrect_pie_chart_combined(train_correct, train_incorrect, test_correct, test_incorrect)
        results_df_test = prepare_results_df(y_test, y_test_pred)
        col1, col2 = st.columns([1,1])
        with col1:
            title = "Test Prediction"
            test_predictions(results_df_test, title)
        with col2:
            test_correct_predictions   = results_df_test['Correct'].sum()
            test_incorrect_predictions = len(results_df_test) - test_correct_predictions
            title = "Predictions Percentage"
            correct_incorrect_pie_chart(test_correct_predictions, test_incorrect_predictions, title)

        results_df_train = prepare_results_df(y_train, y_train_pred)
        col1, col2 = st.columns([1,1])
        with col1:
            title = "Train Prediction"
            test_predictions(results_df_train, title)
        with col2:
            train_correct_predictions   = results_df_train['Correct'].sum()
            train_incorrect_predictions = len(results_df_train) - train_correct_predictions
            title = "Predictions Percentage"
            correct_incorrect_pie_chart(train_correct_predictions, train_incorrect_predictions, title)

    if learning_rate:
        st.divider()
        learning_rate_line_plot(train_scores, test_scores)






elif option == "With Tuning":
    predictions   = st.sidebar.toggle("Predictions"   ,    key="prediction")
    learning_rate = st.sidebar.toggle("Learning Rates",    key="learning_rates")

    with open(f'pickle_file/fold_{selected_kfolds}_data.pkl', 'rb') as f:
        data = pickle.load(f)

    (mse, rmse, r2, mae, r2_train_score,
    r2_test_score, mse_train_score,
    mse_test_score, y_train_pred, 
    y_test_pred, y_pred, X_train, 
    X_test, y_train, y_test, model) = data

    mse             = np.asarray(mse)
    rmse            = np.asarray(rmse)
    r2              = np.asarray(r2)
    mae             = np.asarray(mae)
    r2_train_score  = np.asarray(r2_train_score)
    r2_test_score   = np.asarray(r2_test_score)
    mse_train_score = np.asarray(mse_train_score)
    mse_test_score  = np.asarray(mse_test_score)
    y_train_pred    = np.asarray(y_train_pred)
    y_test_pred     = np.asarray(y_test_pred)
    y_pred          = np.asarray(y_pred)
    X_train         = np.asarray(X_train)
    X_test          = np.asarray(X_test)
    y_train         = np.asarray(y_train)
    y_test          = np.asarray(y_test)

    if predictions:
        st.divider()
        col1, col2 = st.columns([1,1])
        with col1:
            plot_tune_outliers(data)
        with col2:
            plot_prediction_accuracy(data)
    


    if learning_rate:
        df = pd.read_csv("CSV Files/kfold_scores.csv")
        tune_learning_rate(df)
