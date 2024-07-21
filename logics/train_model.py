import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
import pickle
import os


def calculate_residuals(y_test, y_pred, lower_percentile, upper_percentile):
    residuals = y_test - y_pred
    Q1 = np.percentile(residuals, lower_percentile)
    Q3 = np.percentile(residuals, upper_percentile)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return residuals, lower_bound, upper_bound


def calculate_metrics(y_true, y_pred):
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'R²' : r2_score(y_true, y_pred)
    }
    return metrics


print("TEsting")

def train_all_models(read_csv, models):
    df = read_csv

    columns_to_keep = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'area']
    df_selected = df[columns_to_keep]

    df_selected = df_selected.dropna()

    scaler = StandardScaler()
    features = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind']
    df_selected[features] = scaler.fit_transform(df_selected[features])

    selected_features = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind']
    X = df_selected[selected_features]
    y = df_selected['area']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    

    results  = {}
    outliers = {}

    extra_trees_model_name = 'Extra Trees'

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        results[name] = {
            'Model': name,
            'MSE'  : mse,
            'RMSE' : rmse,
            'R2'   : r2,
            'MAE'  : mae
        }

        if name == extra_trees_model_name:
            outliers[name] = {
                'Actual': y_test, 
                'Predicted': y_pred,
            }
        
    return results, outliers

def extra_tress_without_tune(read_csv):
    df = read_csv
    columns_to_keep = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'area']
    df_selected = df[columns_to_keep]
    df_selected = df_selected.dropna()
    scaler = StandardScaler()
    features = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind']
    df_selected[features] = scaler.fit_transform(df_selected[features])
    selected_features = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind']
    X = df_selected[selected_features]
    y = df_selected['area']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = ExtraTreesRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Predict the target values for the training and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)

    # Calculate train and test scores for learning curve
    train_scores = []
    test_scores  = []
    for n_trees in range(1, 101):
        model = ExtraTreesRegressor(n_estimators=n_trees, random_state=42)
        model.fit(X_train, y_train)
        train_scores.append(model.score(X_train, y_train))
        test_scores.append(model.score(X_test, y_test))
        print(f"{n_trees}% Done!")

    with open('pickle_file/extra_tress_results.pkl', 'wb') as f:
        pickle.dump((X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, y_pred, train_scores, test_scores), f)





def extra_trees_with_tune(read_csv, select_kfold, n_splits):
    df = read_csv
    columns_to_keep = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'area']
    df_selected = df[columns_to_keep]
    df_selected = df_selected.dropna()
    
    scaler = StandardScaler()
    features = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind']
    df_selected[features] = scaler.fit_transform(df_selected[features])
    
    X = df_selected[features]
    y = df_selected['area']
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    r2_train_scores = []
    r2_test_scores = []
    mse_train_scores = []
    mse_test_scores = []
    rmse_test_scores = []
    mae_test_scores = []

    fold = select_kfold
    
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model = ExtraTreesRegressor(random_state=42)
        model.fit(X_train, y_train)
        
        y_pred       = model.predict(X_test)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        r2_train_score = r2_score(y_train, y_train_pred)
        r2_test_score = r2_score(y_test, y_test_pred)
        mse_train_score = mean_squared_error(y_train, y_train_pred)
        mse_test_score = mean_squared_error(y_test, y_test_pred)
        rmse_test_score = np.sqrt(mse_test_score)
        mae_test_score = mean_absolute_error(y_test, y_test_pred)
        
        mse  = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_test, y_pred)
        mae  = mean_absolute_error(y_test, y_pred)


        r2_train_scores.append(r2_train_score)
        r2_test_scores.append(r2_test_score)
        mse_train_scores.append(mse_train_score)
        mse_test_scores.append(mse_test_score)
        rmse_test_scores.append(rmse_test_score)
        mae_test_scores.append(mae_test_score)
        
        with open(f'pickle_file/fold_{i+1}_data.pkl', 'wb') as f:
            pickle.dump({
                'r2_train_score' : r2_train_score,
                'r2_test_score'  : r2_test_score,

                'mse_train_score': mse_train_score,
                'mse_test_score' : mse_test_score,

                'mse'            : mse,
                'rmse'           : rmse,
                'r2'             : r2,
                'mae'            : mae,

                'y_train_pred'   : y_train_pred,
                'y_test_pred'    : y_test_pred,
                'y_pred'         : y_pred,

                'X_train'        : X_train,
                'X_test'         : X_test,
                'y_train'        : y_train,
                'y_test'         : y_test,

                'model'          : model
            }, f)
        fold += 1
        print(f"\t{i}\tKFold Done!")
    
    scores_df = pd.DataFrame({
        'Fold': list(range(1, n_splits + 1)),
        'R2 Train': r2_train_scores,
        'R2 Test': r2_test_scores,
        'MSE Train': mse_train_scores,
        'MSE Test': mse_test_scores,
        'RMSE Test': rmse_test_scores,
        'MAE Test': mae_test_scores
    })
    scores_df.to_csv('CSV Files/kfold_scores.csv', index=False)


# def extra_tress_with_tune(read_csv, select_kfold, n_splits):
#     df = read_csv
#     columns_to_keep = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'area']
#     df_selected = df[columns_to_keep]
#     df_selected = df_selected.dropna()
#     scaler = StandardScaler()
#     features = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind']
#     df_selected[features] = scaler.fit_transform(df_selected[features])
#     selected_features = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind']
#     X = df_selected[selected_features]
#     y = df_selected['area']
    
#     kf = KFold(n_splits= n_splits, shuffle=True, random_state=42)
    
#     fold = select_kfold

#     # Lists to store the scores for each fold
#     r2_train_scores = []
#     r2_test_scores = []
#     mse_train_scores = []
#     mse_test_scores = []
#     rmse_test_scores = []
#     mae_test_scores = []


#     for i, train_index, test_index in kf.split(X):
#         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#         y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
#         model = ExtraTreesRegressor(random_state=42)
#         model.fit(X_train, y_train)

#         y_pred       = model.predict(X_test)
#         y_train_pred = model.predict(X_train)
#         y_test_pred  = model.predict(X_test)
        
#         r2_train_score  = r2_score(y_train, y_train_pred)
#         r2_test_score   = r2_score(y_test, y_test_pred)
#         mse_train_score = mean_squared_error(y_train, y_train_pred)
#         mse_test_score  = mean_squared_error(y_test, y_test_pred)

#         mse  = mean_squared_error(y_test, y_pred)
#         rmse = np.sqrt(mse)
#         r2   = r2_score(y_test, y_pred)
#         mae  = mean_absolute_error(y_test, y_pred)

#         with open(f'pickle_file/fold_{i+1}_data.pkl', 'wb') as f:
#             pickle.dump({
#                 'r2_train_score' : r2_train_score,
#                 'r2_test_score'  : r2_test_score,

#                 'mse_train_score': mse_train_score,
#                 'mse_test_score' : mse_test_score,

#                 'mse'            : mse,
#                 'rmse'           : rmse,
#                 'r2'             : r2,
#                 'mae'            : mae,

#                 'y_train_pred'   : y_train_pred,
#                 'y_test_pred'    : y_test_pred,
#                 'y_pred'         : y_pred,

#                 'X_train'        : X_train,
#                 'X_test'         : X_test,
#                 'y_train'        : y_train,
#                 'y_test'         : y_test,

#                 'model'          : model
#             }, f)
#         fold += 1
#         print(f"\t{i}\tKFold Done!" )

#     # Create a DataFrame to store all the scores
#     scores_df = pd.DataFrame({
#         'Fold': list(range(1, n_splits + 1)),
#         'R2 Train': r2_train_scores,
#         'R2 Test': r2_test_scores,
#         'MSE Train': mse_train_scores,
#         'MSE Test': mse_test_scores,
#         'RMSE Test': rmse_test_scores,
#         'MAE Test': mae_test_scores
#     })
#     scores_df.to_csv(f'CSV Files/kfold_scores.csv', index=False)


# mse, rmse, r2, mae, r2_train_score, r2_test_score, mse_train_score, mse_test_score, y_train_pred, y_test_pred, y_pred, X_train, X_test, y_train, y_test, model