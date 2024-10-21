import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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
    filename=os.path.join(log_dir, 'univariate_analysis.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def class_distribution(df, column):
    try:
        logging.debug('Plotting class distribution')
        plt.figure(figsize=(8,6))
        sns.countplot(x=column, data=df)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks([0, 1], ['No Fraud', 'Fraud'])
        logging.info('Class distribution successfully plotted')
        plt.show()
    
    except Exception as e:
        logging.error('Error plotting class distribution: %s', e)
        return None

def summary_statistics(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    '''
    Calculates and displays summary statistics for the given columns in the dataframe.
    
    Parameters:
    ----------
    df : pd.DataFrame
        The dataframe containing the data.
    columns : list
        List of columns for which to calculate summary statistics.
    
    Returns:
    ----------
    pd.DataFrame
        A dataframe containing the summary statistics for the given columns.
    '''
    logging.info('Calculating summary statistics')
    return df[columns].describe()

def numerical_distribution(df, numerical_columns, style='whitegrid', figsize=(15, 15), bins=30):
    '''
    Plot the numerical distribution of the specified columns.
    
    Parameters:
    df (DataFrame): The dataframe containing the data.
    numerical_columns (list): A list of numerical column names.
    style (str): The plot style (default: 'whitegrid').
    figsize (tuple): The figure size (default: (15, 15)).
    bins (int): The number of bins for the histogram (default: 30).
    '''
    logging.info('Plotting numerical distribution')
    sns.set_style(style)
    fig, axes = plt.subplots(nrows=(len(numerical_columns) + 2) // 3, ncols=3, figsize=figsize)
    axes = axes.flatten()
    
    # Custom color palette
    palette = sns.color_palette("deep")
    
    for i, column in enumerate(numerical_columns):
        sns.histplot(df[column], kde=True, ax=axes[i], bins=bins, color=palette[i % len(palette)])
        axes[i].set_title(f'Distribution of {column}', fontsize=14)
        axes[i].set_xlabel(column, fontsize=12)
        axes[i].set_ylabel('Frequency', fontsize=12)
        axes[i].grid(True)
        axes[i].tick_params(axis='both', labelsize=10)
        
        # Rotate x-axis labels if necessary
        if len(df[column].unique()) > 10:
            axes[i].tick_params(axis='x', rotation=45)
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    # Add a title and subtitle
    fig.suptitle('Numerical Distribution of Variables', fontsize=18)
    fig.subplots_adjust(top=0.9)
    
    plt.tight_layout()
    logging.info('Numerical distribution plotted')
    plt.show()

def categorical_distribution(df, categorical_columns, style='whitegrid', figsize=(8, 12)):
    logging.info('Plotting categorical distribution')
    sns.set_style(style)
    n_plots = len(categorical_columns)
    fig, axes = plt.subplots(nrows=n_plots, ncols=1, figsize=figsize)
    
    # Using a bold color palette
    palette = sns.color_palette("bright", n_plots)  # Change to "deep" or "Set1" for more options
    
    for i, (column, ax) in enumerate(zip(categorical_columns, axes)):
        counts = df[column].value_counts()
        counts.plot(kind='bar', ax=ax, color=palette[i], alpha=0.6)  # Set alpha to 1 for bold colors
        
        # Adding value annotations
        for index, value in enumerate(counts):
            ax.text(index, value, f'{value} ({value/counts.sum()*100:.1f}%)', 
                    ha='center', va='bottom', fontsize=10)
        
        ax.set_title(f'Categorical Distribution of {column}', fontsize=18)
        ax.set_xlabel('Category', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  # Rotate x-axis labels
        ax.set_ylim(0, counts.max() * 1.1)  # Set y-axis limit

    plt.tight_layout()
    plt.show()

def outlier_detection(df, numerical_columns, style='whitegrid', ncols=3, figsize=(10,8), bins=30):
    # set the style
    logging.info('Plotting outlier detection')
    sns.set_style(style)
    palette = sns.color_palette("Set2")

    fig, axes = plt.subplots(nrows=(len(numerical_columns) + 2) // ncols, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    # Create a box plot for each numeric columns
    for i, column in enumerate(numerical_columns):
        sns.boxplot(data=df[column], ax=axes[i], color=palette[i % len(palette)])
        axes[i].set_title(f'Box plot for {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Value')
        axes[i].grid(True)
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()
    logging.info('Outlier detection plotted')
    plt.show()


def analyze_fraud_data(df, date_columns):
    try:
        logging.info('Converting specified datae column')
        
        # Convert specified date columns to datetime
        for col in date_columns:
            logging.info('Converting date column to datetime')
            df[col] = pd.to_datetime(df[col])
        
        # Extract date information from date columns
        for i, col in enumerate(date_columns):
            logging.info('Extracting date information')
            df[f'{col}_date'] = df[col].dt.date
        
        # Group by date and count transactions
        logging.info('Grouping by date and counting transactions')
        purchase_counts = df.groupby(f'{date_columns[1]}_date').size()
        
        # Plot the time series
        logging.info('Plotting time series')
        plt.figure(figsize=(12, 6))
        purchase_counts.plot(title='Purchase Frequency')
        plt.xlabel('Date')
        plt.ylabel('Number of Transactions')
        plt.grid(True)
        plt.show()
        
        # Extract hour from the second date column (assumed to be 'purchase_time')
        logging.info('Extracting hour from date column')
        df['purchase_hour'] = df[date_columns[1]].dt.hour
        
        # Create a pivot table for hours and days of the week
        logging.info('Creating pivot table')
        pivot_table = pd.pivot_table(df, values='class', index='purchase_hour', columns=df[date_columns[1]].dt.dayofweek, aggfunc='count')
        
        # Plot heatmap
        logging.info('Plotting heatmap')
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot_table, annot=True, cmap='Blues')
        plt.title('Hourly Purchase Frequency by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Hour of Day')
        plt.show()
    
    except Exception as e:
        logging.error('Error analyzing fraud data: %s', e)
