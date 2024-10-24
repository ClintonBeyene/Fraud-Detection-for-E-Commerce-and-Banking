# Import necessary libraries
import pandas as pd
import numpy as np
import os
import logging
import sys

# Add the parent directory to the system path
sys.path.append(os.path.join(os.path.abspath('../')))

# Ensure the logs directory exists
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../logs"))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Set up logging
logging.basicConfig(
    filename=os.path.join(log_dir, 'data_loading.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

_DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))

def load_data(file_path):
    '''
    This function loads the data from the given dataframe and returns the data in a suitable format for the model.
    
    Parameter:
    ------------
    file_path (str): Path to dataset.
    
    Returns:
    ------------
    pd.DataFrame
    '''
    try:
        logging.info('Loading pandas dataframe')
        return pd.read_csv(os.path.join(_DATASET_DIR, file_path))
    except Exception as e:
        logging.error(f'Error loading data: {e}')

