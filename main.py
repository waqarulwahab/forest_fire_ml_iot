import streamlit as st
import pandas as pd

from logics.train_model import *
from logics.models import *
from logics.load_data import *


st.set_page_config(layout="wide")

def main():

    st.sidebar.page_link("main.py",                        label="OVERVIEW", icon="🏠")
    st.sidebar.page_link("pages/extra_trees_regressor.py", label="MODELS",   icon="⚙️")
    st.sidebar.page_link("pages/about_dataset.py",         label="DATASET",  icon="📖")

    st.title("FOREST FIRE PREDICTION SYSTEM, using IOT and ML")
    abstract = """
                Forest fires pose significant threats worldwide, impacting human habitats and 
                ecosystems through smoke, air pollution, and property damage. Early detection is crucial to mitigate these 
                effects. This paper proposes a system utilizing a wireless sensor network to detect forest fires at their initial 
                stage. Artificial intelligence algorithms are used to fuse data from various sources, including fire hotspots, 
                meteorological conditions, terrain, vegetation, and socioeconomic data. The dataset, sourced from Kaggle, 
                is divided into a 70% training set and a 30% test set. After testing several algorithms, the Extra Tree 
                Regressor model was selected for formal data processing due to its superior performance. The system 
                includes a machine learning regression model to enhance detection accuracy and is powered by 
                rechargeable batteries with a secondary solar supply, enabling long-term standalone operation. Sensor node 
                design and placement are optimized for harsh forest environments. The paper reviews IoT, machine 
                learning, and deep learning approaches for wildfire detection and spread prediction, providing a 
                comprehensive evaluation and comparison of existing methodologies.
                """
    st.write(abstract)


    read_csv = load_data()
    model_names, training_models = models()

    
    col1, col2 = st.columns([1,5])
    with col1:
        all_models = st.button("Train All Models", key="train_all_models")
    with col2:
        without_tune = st.button("Extra Trees", key="extra_trees_without_tine")

    st.divider()
    col1, col2 = st.columns([1,5])
    with col1:
        with_tune    = st.toggle("K-Fold Tuning",  key="extra_trees_with_tune")
    with col2:
        if with_tune:
            customize_tuning = st.toggle("Customize Your Model", key="cutomize_model")
    
    try:
        if customize_tuning:
            col1, col2, col3 = st.columns([1,1,8])
            with col1:
                select_kfold = st.number_input("No. of KFolds", key='kfold',    min_value=1, step=1)
            with col2:
                n_splits     = st.number_input("N Splits",      key='n_splits', min_value=1, step=1)
            with col3:
                pass

            if st.toggle("Train Now."):
                extra_trees_with_tune(read_csv, select_kfold, n_splits)
                st.success("Model Train Successfully")
    except:
        pass

    try:
        if all_models:
            trained_models, outliers_data = train_all_models(read_csv, training_models)
            trained_models = pd.DataFrame(trained_models).T
            outliers_data  = pd.DataFrame(outliers_data)
            trained_models.to_csv('CSV Files/model_performance.csv', index=False)
            outliers_data.to_csv('CSV Files/outliers_data.csv', index=False)
            st.success("Models Train Successfully")
    except:
        pass

    try:
        if without_tune:
            extra_tress_without_tune(read_csv)
            st.success("Model Train Successfully")
    except:
        pass


if '__main__':
    main()



