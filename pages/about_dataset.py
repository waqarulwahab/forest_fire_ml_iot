import pandas as pd
from plots.cards import *
from plots.model_performance_plots import *


import streamlit as st

from logics.train_model import *


from plots.outliers import *
from plots.overfitting import *
from logics.logics import *
from logics.models import *
from logics.header import *



st.sidebar.page_link("main.py",                        label="OVERVIEW", icon="🏠")
st.sidebar.page_link("pages/extra_trees_regressor.py", label="MODELS",   icon="⚙️")
st.sidebar.page_link("pages/about_dataset.py",         label="DATASET",  icon="📖")



upload_data        = pd.read_csv('complete_forest_fire_data.csv')

num_rows = len(upload_data)
st.sidebar.divider()

show_dataframe_head   = st.sidebar.toggle("DF Head",   key="dataframe_head")
display_full_data     = st.sidebar.toggle("Dataframe", key="full_dataframe")

data_shape  = upload_data.shape
null_values = upload_data.isnull().sum().sum()

st.sidebar.divider()
col1, col2 = st.sidebar.columns([1,1])
with col1:
    start_row = st.number_input("Starting Row:", min_value=0, max_value=num_rows-1, value=0)
with col2:
    end_row   = st.number_input("Ending Row:",   min_value=0, max_value=num_rows-1, value=num_rows-1)

columns          = upload_data.columns.tolist()
selected_columns = st.sidebar.multiselect(f"Values in :", columns, default=columns)

selected_data = upload_data.iloc[start_row:end_row+1]
selected_data = selected_data[selected_columns]



header(selected_data)


data = pd.DataFrame(selected_data)

col1, col2 = st.columns([1,5])
with col1:
    plot_type = st.selectbox('Select plot type', ['Scatter Plot', 'Line Chart', 'Bar Chart', 'Density Plot', 'Box Plot'])
with col2:
    pass
col1, col2, col3 = st.columns([1,1,5])
with col1:
    x_axis = st.selectbox('Select X-axis', data.columns)
with col2:
    y_axis = st.selectbox('Select Y-axis', data.columns)

# Plot based on user selection
if plot_type == 'Scatter Plot':
    fig = px.scatter(data, x=x_axis, y=y_axis, title=f'{plot_type} of {x_axis} vs {y_axis}')
elif plot_type == 'Line Chart':
    fig = px.line(data, x=x_axis, y=y_axis, title=f'{plot_type} of {x_axis} vs {y_axis}')
elif plot_type == 'Bar Chart':
    fig = px.bar(data, x=x_axis, y=y_axis, title=f'{plot_type} of {x_axis} vs {y_axis}')
elif plot_type == 'Density Plot':
    fig = px.density_contour(data, x=x_axis, y=y_axis, title=f'{plot_type} of {x_axis} vs {y_axis}')
elif plot_type == 'Box Plot':
    fig = px.box(data, x=x_axis, y=y_axis, title=f'{plot_type} of {x_axis} vs {y_axis}')
# Display the plot
st.plotly_chart(fig)


col1, col2 = st.columns([1,1])
with col1:
    # Select the numerical columns for the heatmap
    numerical_columns = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'area']
    # Calculate the correlation matrix
    correlation_matrix = selected_data[numerical_columns].corr()
    # Generate the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=numerical_columns,
        y=numerical_columns,
        colorscale='Viridis',
        hoverongaps=False,
        zmin=-1, zmax=1,
        colorbar=dict(title="Correlation")
    ))
    # Customize the layout
    fig.update_layout(
        title='Heatmap of Numerical Features Correlation',
        xaxis_title='Features',
        yaxis_title='Features',
        width=800, # Increase the width
        height=800 # Increase the height
    )
    # Display the heatmap in Streamlit
    st.plotly_chart(fig)
with col2:
    pass








if show_dataframe_head:
    st.write(upload_data.head(5))
if display_full_data:
    st.write(upload_data)