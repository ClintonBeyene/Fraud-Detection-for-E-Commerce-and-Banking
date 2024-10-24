import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import OneHotEncoder, StandardScaler
import dask.dataframe as dd
import dask.array as da

import logging 
import os 
import sys
from src.utils import *

# Add the parent directory to the system path
sys.path.append(os.path.join(os.path.abspath('../')))

# Ensure the logs directory exists
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../logs"))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Set up logging
logging.basicConfig(
    filename=os.path.join(log_dir, 'geolocations_analysis.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def create_device_browser_combination(df):
    df['device_browser_combination'] = df['device_id'] + '_' + df['browser']
    return df

def create_country_source_combination(df):
    df['country_source_combination'] = df['country'] + '_' + df['source']
    return df


def extract_datetime_features(df: pd.DataFrame, datetime_columns: list) -> pd.DataFrame:
    """
    Extracts datetime features and drops the original datetime columns from the DataFrame.

    Args:
    df: Input DataFrame with datetime columns.
    datetime_columns: List of datetime columns to process.

    Returns:
    DataFrame with extracted datetime features and without the original datetime columns.
    """
    for col in datetime_columns:
        df[col] = pd.to_datetime(df[col])
        df[f'{col}_hour'] = df[col].dt.hour
        df[f'{col}_day'] = df[col].dt.dayofweek
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_month'] = df[col].dt.month
    
    return df



def analyze_fraud_patterns(df: pd.DataFrame):
    """
    Analyze and plot fraud patterns
    
    Args:
    df: Input DataFrame with 'class' column (0 for non-fraud, 1 for fraud)
    """
    if 'class' not in df.columns and 'Class' not in df.columns:
        raise ValueError("DataFrame must contain 'class' or 'Class' column")
        
    target_col = 'class' if 'class' in df.columns else 'Class'
    
    # Create age groups
    bins = [0, 17, 24, 34, 44, 54, 64, 100]  # Define age bins
    labels = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=True)
    
    # Create subplots with modified layout
    fig = make_subplots(rows=3, cols=2,
                        subplot_titles=('Fraud by Hour of Day',
                                       'Fraud Distribution',
                                       'Transaction Count by Age Group',
                                       'Total Purchase Value by Class',
                                       'Total Purchase Value by Age Group',
                                       'Fraud by Day of Week'))
    
    # Fraud by Hour of Day
    if 'purchase_time_hour' in df.columns:
        hourly_fraud = df[df[target_col] == 1]['purchase_time_hour'].value_counts().sort_index()
        fig.add_trace(
            go.Scatter(x=hourly_fraud.index, y=hourly_fraud.values,
                       mode='lines+markers',
                       line=dict(color='purple', width=2),
                       marker=dict(size=8)),
            row=1, col=1
        )
    
    # Fraud Distribution
    fraud_dist = df[target_col].value_counts(normalize=True) * 100
    fig.add_trace(
        go.Bar(x=['Non-Fraud', 'Fraud'], y=fraud_dist.values,
               marker_color=px.colors.sequential.Viridis[:2]),
        row=1, col=2
    )
    
    # Transaction Count by Age Group
    transaction_count_age = df['age_group'].value_counts()
    fig.add_trace(
        go.Bar(x=transaction_count_age.index, y=transaction_count_age.values,
               marker_color=px.colors.sequential.Viridis[2]),
        row=2, col=1
    )
    
    # Total Purchase Value by Class
    if 'purchase_value' in df.columns:
        total_purchase = df.groupby(target_col, observed=True)['purchase_value'].sum()
        fig.add_trace(
            go.Bar(x=['Non-Fraud', 'Fraud'], y=total_purchase.values,
                marker_color=px.colors.sequential.Viridis[3:5]),
            row=2, col=2
        )
    
    # Average Purchase Value by Age Group
    if 'purchase_value' in df.columns:
        avg_purchase_age = df.groupby('age_group', observed=True)['purchase_value'].sum()
        fig.add_trace(
            go.Bar(x=avg_purchase_age.index, y=avg_purchase_age.values,
                marker_color=px.colors.sequential.Viridis[5]),
            row=3, col=1
        )
        
    # Fraud by Day of Week
    if 'purchase_time_day' in df.columns:
        daily_fraud = df[df[target_col] == 1]['purchase_time_day'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                   y=daily_fraud.values,
                   marker_color=px.colors.sequential.Viridis[6]),
            row=3, col=2
        )
    
    # Update layout with titles and axis labels
    fig.update_layout(
        height=1000, 
        width=1000, 
        showlegend=False,
        title_text='Fraud Analysis Overview',
        title_x=0.5,
        xaxis_title='Category',
        yaxis_title='Total Purchase Value',
        font=dict(size=12)
    )
    
    # Show the figure
    fig.show()
    # Save the image
    fig.write_image("../Visual/analyzing_fraud_patterns.png")

