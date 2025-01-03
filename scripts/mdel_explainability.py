# Import necessary libraries
import numpy as np
import shap
import lime
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
import logging
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
plt.ioff()
from scipy.special import expit as sigmoid

# Add the parent directory to the system path
sys.path.append(os.path.join(os.path.abspath('..')))

# Manually set the logs directory
log_dir = os.path.abspath(os.path.join(os.getcwd(), "../logs"))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Set up logging
logging.basicConfig(
    filename=os.path.join(log_dir, 'model_explainability.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def shap_explainability(model, X_train, X_test):
    # Turn off interactive mode in Matplotlib
    plt.ioff()

    # Create a SHAP explainer
    masker = shap.maskers.Independent(X_train)
    shap_explainer = shap.Explainer(model.predict_proba, masker)
    
    # Get SHAP values for the test data
    with shap.utils.show_progress(False):
        shap_values = shap_explainer(X_test)
    
    # SHAP summary plot
    shap.plots.beeswarm(shap_values[:, :, 0], show=False)
    plt.title("SHAP Summary Plot")
    plt.show()
    
    # SHAP force plot
    shap.plots.force(shap_values[0, :, 0], matplotlib=True, show=False)
    plt.title("SHAP Force Plot for the First Instance")
    plt.show()
    
    # SHAP dependence plot
    shap.plots.scatter(shap_values[:, :, 0], color=shap_values, show=False)


def shap_explainability_neural(model, X_train, X_test):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow's verbose output
    # Turn off interactive mode in matplotlib 
    plt.ioff()

    # Create a SHAP explainer
    masker = shap.maskers.Independent(X_train)
    model_proba = lambda x: sigmoid(model.predict(x, verbose=0))
    shap_explainer = shap.Explainer(model_proba, masker, algorithm='permutation', nsamples=100, verbose=0)
    
    # Get SHAP values for the test data
    shap_values = shap_explainer(X_test)
    
    # SHAP summary plot
    shap.plots.beeswarm(shap_values, show=False)
    plt.title("SHAP Summary Plot")
    plt.show()
    
    # SHAP force plot
    shap.plots.force(shap_values[0, :], matplotlib=True, show=False)
    plt.title("SHAP Force Plot for the First Instance")
    plt.show()
    
    # SHAP dependence plot
    shap.plots.scatter(shap_values[:, 0], ylabel="SHAP value\n(higher means more likely to fraud)", show=False)
    plt.title("SHAP Dependence Plot for the First Feature")
    plt.show()
    
def lime_explainability(model, X_train, X_test):
    # Convert X_train to a pandas DataFrame
    X_train_df = pd.DataFrame(X_train)
    
    # Create a LIME explainer
    lime_explainer = LimeTabularExplainer(X_train_df.values, feature_names=X_train_df.columns, class_names=["Not Fraud", "Fraud"], discretize_continuous=True)
    
    # Get LIME explanations for the test data
    lime_explanation = lime_explainer.explain_instance(X_test[0], model.predict_proba, num_features=5)
    
    # LIME feature importance plot
    lime_explanation.as_pyplot_figure()
    plt.title("LIME Feature Importance Plot for the First Instance")
    plt.show()


def lime_explainability_neural(model, X_train, X_test):
    # Convert X_train to a pandas DataFrame
    X_train_df = pd.DataFrame(X_train)
    
    # Create a LIME explainer
    lime_explainer = LimeTabularExplainer(X_train_df.values, feature_names=X_train_df.columns, class_names=["Not Fraud", "Fraud"], discretize_continuous=True)
    
    # Define a function to predict probabilities
    def predict_proba(X):
        predictions = model.predict(X)
        return np.array([[(1 - pred[0]), pred[0]] for pred in predictions])
    
    # Get LIME explanations for the test data
    lime_explanation = lime_explainer.explain_instance(X_test[0], predict_proba, num_features=5)
    
    # LIME feature importance plot
    lime_explanation.as_pyplot_figure()
    plt.title("LIME Feature Importance Plot for the First Instance")
    plt.show()
