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
    filename=os.path.join(log_dir, 'bivariate_analysis.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def bivariate_kdeplot(data, x, hue):
    try:
        logging.debug('Bibariate analysis')
        sns.kdeplot(data=data, x=x, hue=hue)
        plt.title(f'KDE Plot of {x} by {hue}')
        plt.xlabel(x)
        plt.ylabel('Density')
        # Display the plot
        plt.show()
        logging.info('Successfully ploted')
    except Exception as e:
        logging.error(f'Failed to plot: %s', e)
        return None
    
def violinplot(data, x, hue):
    # We can change the dimensions of our plot with this line of code. Make sure to choose a size that highlights the features of the data.
    plt.figure(figsize=(14, 8))
    
    # Log the inputs
    logging.info(f"Creating violin plot with data for {x} grouped by {hue}")
    
    # Plot a Violin plot
    sns.violinplot(data=data, x=x, hue=hue)
    
    # Show the plot
    plt.show()
    logging.info("Violin plot displayed successfully")