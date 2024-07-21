from streamlit_echarts import st_echarts


def all_model_performance(all_model_per_data):
    model = all_model_per_data['Model'].tolist()
    mse   = [round(val, 1) for val in all_model_per_data['MSE'].tolist()]
    rmse  = [round(val, 1) for val in all_model_per_data['RMSE'].tolist()]
    mae   = [round(val, 1) for val in all_model_per_data['MAE'].tolist()]
    r2    = [round(val, 1) for val in all_model_per_data['R2'].tolist()]

    options = {
        "title": {
            "text": "All Regressor Models"
        },
        "tooltip": {
            "trigger": "axis"
        },
        "legend": {
            "data": ["MSE", "RMSE", "MAE", "R2"]
        },
        "grid": {
            "left": "3%",
            "right": "4%",
            "bottom": "3%",
            "containLabel": True
        },
        "toolbox": {
            "feature": {
                "saveAsImage": {}
            }
        },
        "xAxis": {
            "type": "category",
            "boundaryGap": False,
            "data": model,
            "axisLabel": {
                "rotate": 45,  # Rotate labels 45 degrees
                "fontWeight": "bold",  # Make labels bold
                "fontSize": 12  # Increase font size
            }
        },
        "yAxis": {
            "type": "value"
        },
        "dataZoom": [
            {
                "type": "slider",
                "start": 0,
                "end": 100
            },
            {
                "type": "inside",
                "start": 0,
                "end": 100
            }
        ],
        "series": [
            {
                "name": "MSE",
                "type": "line",
                "data": mse
            },
            {
                "name": "RMSE",
                "type": "line",
                "data": rmse
            },
            {
                "name": "MAE",
                "type": "line",
                "data": mae
            },
            {
                "name": "R2",
                "type": "line",
                "data": r2
            },
        ]
    }
    st_echarts(options=options, height=400, width=1350),



def extra_tree_regressor_pie(all_model_per_data):
    etr_data = all_model_per_data[all_model_per_data['Model'] == 'Extra Trees']
    
    mse   = round(etr_data['MSE'].values[0], 1)
    rmse  = round(etr_data['RMSE'].values[0], 1)
    mae   = round(etr_data['MAE'].values[0], 1)
    r2    = round(etr_data['R2'].values[0], 1)

    data = [
        {"value": mse, "name": "MSE"},
        {"value": rmse, "name": "RMSE"},
        {"value": mae, "name": "MAE"},
        {"value": r2, "name": "R2"},
    ]

    options = {
        "title": {
            "subtext": "Extra-Tree-Regressor Model Performance Metrics By PIE",
            "left": "center"
        },
        "tooltip": {
            "trigger": "item"
        },
        "legend": {
            "orient": "horizontal",
            "top": "bottom",
            "data": ["MSE", "RMSE", "MAE", "R2"]
        },
        "series": [
            {
                "name": "Metrics",
                "type": "pie",
                "radius": "50%",
                "data": data,
                "emphasis": {
                    "itemStyle": {
                        "shadowBlur": 10,
                        "shadowOffsetX": 0,
                        "shadowColor": "rgba(0, 0, 0, 0.5)"
                    }
                }
            }
        ]
    }
    st_echarts(options=options, height=400, width=600)

def extra_tree_regressor_bar(all_model_per_data):
    etr_data = all_model_per_data[all_model_per_data['Model'] == 'Extra Trees']
    
    mse   = round(etr_data['MSE'].values[0], 1)
    rmse  = round(etr_data['RMSE'].values[0], 1)
    mae   = round(etr_data['MAE'].values[0], 1)
    r2    = round(etr_data['R2'].values[0], 1)


    metrics = ["MSE", "RMSE", "MAE", "R2"]
    values = [mse, rmse, mae, r2]

    options = {
        "title": {
            "subtext": "Extra-Tree-Regressor Model Performance Metrics By BARS",
            "left": "center"
        },
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {
                "type": "shadow"
            }
        },
        "legend": {
            "orient": "horizontal",
            "top": "bottom",
            "data": metrics
        },
        "xAxis": {
            "type": "category",
            "data": metrics
        },
        "yAxis": {
            "type": "value"
        },
        "series": [
            {
                "name": "Metrics",
                "type": "bar",
                "data": values,
                "itemStyle": {
                    "color": "#5470C6",
                    "emphasis": {
                        "shadowBlur": 10,
                        "shadowOffsetX": 0,
                        "shadowColor": "rgba(0, 0, 0, 0.5)"
                    }
                }
            }
        ]
    }
    st_echarts(options=options, height=400, width=600)