import pickle
import re
import glob

def load_fold_data(fold_number):
    with open(f'pickle_file/fold_{fold_number}_data.pkl', 'rb') as f:
        fold_data = pickle.load(f)
    return fold_data