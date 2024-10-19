**Fraud Detection Project**
==========================

**Table of Contents**
-----------------

1. [Introduction](#introduction)
2. [Task 1: Data Analysis and Preprocessing](#task-1-data-analysis-and-preprocessing)
3. [Task 2: Model Building and Training](#task-2-model-building-and-training)
4. [Task 3: Model Explainability](#task-3-model-explainability)
5. [Task 4: Model Deployment and API Development](#task-4-model-deployment-and-api-development)
6. [Task 5: Build a Dashboard with Flask and Dash](#task-5-build-a-dashboard-with-flask-and-dash)

**Introduction**
---------------

This project aims to build a comprehensive fraud detection system using machine learning and data analysis techniques. The project consists of five tasks:

* Task 1: Data Analysis and Preprocessing
* Task 2: Model Building and Training
* Task 3: Model Explainability
* Task 4: Model Deployment and API Development
* Task 5: Build a Dashboard with Flask and Dash

**Task 1: Data Analysis and Preprocessing**
-----------------------------------------

### Handle Missing Values

* Impute or drop missing values
* Remove duplicates
* Correct data types

### Exploratory Data Analysis (EDA)

* Univariate analysis
* Bivariate analysis

### Merge Datasets for Geolocation Analysis

* Convert IP addresses to integer format
* Merge Fraud_Data.csv with IpAddress_to_Country.csv

### Feature Engineering

* Transaction frequency and velocity for Fraud_Data.csv
* Time-Based features for Fraud_Data.csv (hour_of_day, day_of_week)

### Normalization and Scaling

* Encode Categorical Features

**Task 2: Model Building and Training**
-----------------------------------------

### Data Preparation

* Feature and Target Separation [‘Class’(creditcard), ‘class’(Fraud_Data)]
* Train-Test Split

### Model Selection

* Use several models to compare performance, including:
    + Logistic Regression
    + Decision Tree
    + Random Forest
    + Gradient Boosting
    + Multi-Layer Perceptron (MLP)
    + Convolutional Neural Network (CNN)
    + Recurrent Neural Network (RNN)
    + Long Short-Term Memory (LSTM)

### Model Training and Evaluation

* Training models for both credit card and fraud-data datasets

### MLOps Steps

* Versioning and Experiment Tracking using MLflow

**Task 3: Model Explainability**
-------------------------------

### Using SHAP for Explainability

* SHAP values provide a unified measure of feature importance
* SHAP Plots:
    + Summary Plot: Provides an overview of the most important features
    + Force Plot: Visualizes the contribution of features for a single prediction
    + Dependence Plot: Shows the relationship between a feature and the model output

### Using LIME for Explainability

* LIME explains individual predictions by approximating the model locally with an interpretable model
* LIME Plots:
    + Feature Importance Plot: Shows the most influential features for a specific prediction

**Task 4: Model Deployment and API Development**
----------------------------------------------

### Setting Up the Flask API

* Create the Flask Application
* Create a new directory for your project
* Create a Python script serve_model.py to serve the model using Flask
* Create a requirements.txt file to list dependencies

### API Development

* Define API Endpoints
* Test the API

### Dockerizing the Flask Application

* Create a Dockerfile in the same directory
* Build and Run the Container

### Integrate logging

* Use Flask-Logging to log incoming requests, errors, and fraud predictions

**Task 5: Build a Dashboard with Flask and Dash**
----------------------------------------------

### Create an interactive dashboard using Dash

* Use Dash to handle the frontend visualizations
* Add a Flask Endpoint that reads fraud data from a CSV file and serves summary statistics and fraud trends through API endpoints

### Dashboard Insights

* Display total transactions, fraud cases, and fraud percentages in simple summary boxes
* Display a line chart showing the number of detected fraud cases over time
* Analyze where fraud is occurring geographically
* Show a bar chart comparing the number of fraud cases across different devices and browsers
* Show a chart comparing the number of fraud cases across different devices and browsers