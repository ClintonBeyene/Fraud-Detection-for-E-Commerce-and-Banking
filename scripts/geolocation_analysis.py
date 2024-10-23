import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import struct
import socket
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

def convert_ip_to_int(df, ip_column):
    df[f'{ip_column}_int'] = df[ip_column].apply(lambda x: ip_to_int(str(int(x))) if pd.notna(x) else None)
    df[f'{ip_column}_int'] = df[f'{ip_column}_int'].astype('int64')
    return df

def sort_ip_data(df, col):
    df.sort_values(col, inplace=True)
    return df

def merge_ip_data(fraud_df, ip_address_to_country):
    fraud_df_sorted = fraud_df.sort_values('ip_address_int')
    ip_country_cleaned = ip_address_to_country[['lower_bound_ip_address_int', 'upper_bound_ip_address_int', 'country']]
    merged_df = pd.merge_asof(fraud_df_sorted, ip_country_cleaned, left_on='ip_address_int', right_on='lower_bound_ip_address_int', direction='backward')
    merged_df = merged_df[(merged_df['ip_address_int'] >= merged_df['lower_bound_ip_address_int']) & (merged_df['ip_address_int'] <= merged_df['upper_bound_ip_address_int'])]
    merged_df.drop(['lower_bound_ip_address_int', 'upper_bound_ip_address_int'], axis=1, inplace=True)
    return merged_df

def calculate_fraud_rate_by_country(df):
    """
    Calculate fraud rate by country.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing fraud data with country information.
    
    Returns:
    pd.DataFrame: DataFrame with the fraud rate by country.
    """
    country_fraud = df.groupby('country')['class'].mean().reset_index()
    return country_fraud


def plot_fraud_rate_by_country(country_fraud):
    """
    Plot the fraud rate by country on a world map using Plotly.
    
    Parameters:
    country_fraud (pd.DataFrame): DataFrame containing fraud rates by country.
    
    Returns:
    plotly.graph_objects.Figure: Plotly figure object.
    """
    fig = px.choropleth(
        country_fraud,
        locations='country',
        locationmode='country names',
        color='class',
        hover_name='country',
        hover_data=['class'],
        color_continuous_scale='Viridis', 
        title='Global Fraud Rate by Country', 
        range_color=[0, 1]
    )
    
    fig.update_layout(
        coloraxis_colorbar_title="Fraud Rate (%)", 
        title={'text': 'Global Fraud Rate by Country', 'x': 0.5, 'xanchor': 'center'}  
    )
    
    return fig


def user_distribution_by_country(fraud_data_with_country):
    # Calculate User Distribution by Country
    country_dist = fraud_data_with_country.groupby('country')['user_id'].count().reset_index()

    # Rename columns for clarity
    country_dist.columns = ['country', 'user_count']

    # Create the figure
    fig_user_dist = go.Figure()

    # Add bar trace
    fig_user_dist.add_trace(go.Bar(
        x=country_dist['country'],
        y=country_dist['user_count'],
        marker=dict(
            color=country_dist['user_count'],  # Use the 'user_count' column for color
            colorscale='Viridis',  # Use the scale Viridis, Plasma, Inferno, Magma, or Cividis.
            colorbar=dict(title='Number of Users')  # Add a colorbar
        ),
        name='User  Count'
    ))

    # Update layout for y-axis ticks and dropdown menu
    fig_user_dist.update_layout(
        title='User  Distribution by Country',
        xaxis_title='Country',
        yaxis_title='Number of Users',
        yaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=5000
        ),
        updatemenus=[
            dict(
                buttons=[
                    dict(label='All', method='update', args=[{'visible': [True]}]),
                    *[
                        dict(label=f'{i}-{i + 200}',
                            method='update',
                            args=[{'y': [country_dist['user_count'][(country_dist['user_count'] > i) & (country_dist['user_count'] <= i + 200)]],
                                        'x': [country_dist['country'][(country_dist['user_count'] > i) & (country_dist['user_count'] <= i + 200)]]}])
                        for i in range(0, 600, 200)  # Create ranges 0-200, 200-400, ..., 600-1000
                    ] + [
                        dict(label='600-1000',
                            method='update',
                            args=[{'y': [country_dist['user_count'][(country_dist['user_count'] > 600) & (country_dist['user_count'] <= 1000)]],
                                        'x': [country_dist['country'][(country_dist['user_count'] > 600) & (country_dist['user_count'] <= 1000)]]}]),
                        dict(label='1000-2000',
                            method='update',
                            args=[{'y': [country_dist['user_count'][(country_dist['user_count'] > 1000) & (country_dist['user_count'] <= 2000)]],
                                        'x': [country_dist['country'][(country_dist['user_count'] > 1000) & (country_dist['user_count'] <= 2000)]]}]),
                        dict(label='2000-5000',
                            method='update',
                            args=[{'y': [country_dist['user_count'][(country_dist['user_count'] > 2000) & (country_dist['user_count'] <= 5000)]],
                                        'x': [country_dist['country'][(country_dist['user_count'] > 2000) & (country_dist['user_count'] <= 5000)]]}]),
                        dict(label='5000-60000',
                            method='update',
                            args=[{'y': [country_dist['user_count'][country_dist['user_count'] > 5000]],
                                    'x': [country_dist['country'][country_dist['user_count'] > 5000]]}])  # Range 5000-60000
                    ]
                ],
                direction="down",
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top"
            )
        ]
    )

    # Display the user distribution plot
    fig_user_dist.show()
