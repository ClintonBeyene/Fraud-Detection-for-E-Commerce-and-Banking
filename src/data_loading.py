# Import necessay libraries
import pandas as pd
import numpy as np
import os

_DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/raw"))
def load_data(file_path):
    '''
    This function loads the data from the given dataframe and returns the data in a suitable format for the
    model. 
    
    parameter (str): Path to dataset.

    Returns: 
    ------------
    pd.Dataframe
    '''
    return pd.read_csv(os.path.join(_DATASET_DIR, file_path))