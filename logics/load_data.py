import pandas as pd

def load_data():
    file_path = "complete_forest_fire_data.csv"
    read_csv  = pd.read_csv(file_path)
    return read_csv