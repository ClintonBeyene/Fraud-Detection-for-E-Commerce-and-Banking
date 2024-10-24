import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import struct
import socket
import logging 
import os 
import sys

# Add the parent directory to the system path
sys.path.append(os.path.join(os.path.abspath('../')))

# Ensure the logs directory exists
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../logs"))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Set up logging
logging.basicConfig(
    filename=os.path.join(log_dir, 'utils.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def ip_to_int(ip):
    try:
        return struct.unpack("!I", socket.inet_aton(ip))[0]
    except struct.error:
        return None

def calculate_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate transaction features
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with additional transaction-based features
    """
    # Check for required columns
    required_columns = ['user_id', 'purchase_time', 'purchase_value']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("The input DataFrame is missing one or more required columns.")
    
    # Ensure 'purchase_time' is datetime type
    if df['purchase_time'].dtype!= 'datetime64[ns]':
        try:
            df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        except ValueError:
            raise ValueError("The 'purchase_time' column cannot be converted to datetime type.")
    
    # Ensure 'purchase_value' is numeric type
    if df['purchase_value'].dtype not in ['int64', 'float64']:
        try:
            df['purchase_value'] = pd.to_numeric(df['purchase_value'])
        except ValueError:
            raise ValueError("The 'purchase_value' column cannot be converted to numeric type.")
    
    try:
        # Transaction frequency
        df['transaction_count'] = df.groupby('user_id')['user_id'].transform('count')
        
        # Transaction velocity (transactions per day)
        df['user_activity_period_days'] = df.groupby('user_id')['purchase_time'].transform(lambda x: (x.max() - x.min()).total_seconds() / 86400)
        df['transaction_velocity'] = df['transaction_count'] / df['user_activity_period_days'].clip(lower=1)
        
        # Average purchase value per user
        df['avg_purchase'] = df.groupby('user_id')['purchase_value'].transform('mean')
        
        return df
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None