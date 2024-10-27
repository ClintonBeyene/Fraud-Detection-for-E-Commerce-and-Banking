# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn import svm
import lightgbm as lgb
from lightgbm import LGBMClassifier
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Input, Reshape, Flatten
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from mlflow import *
from mlflow.sklearn import *
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay, confusion_matrix, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

# Set pandas options
pd.set_option('display.max_columns', 100)

# Define constants
RFC_METRIC = 'gini'
NUM_ESTIMATORS = 100
NO_JOBS = 4
VALID_SIZE = 0.20
TEST_SIZE = 0.20
NUMBER_KFOLDS = 5
RANDOM_STATE = 2018
MAX_ROUNDS = 1000
EARLY_STOP = 50
OPT_ROUNDS = 1000
VERBOSE_EVAL = 50
IS_LOCAL = False

import logging
import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.join(os.path.abspath('..')))

# Manually set the logs directory
log_dir = os.path.abspath(os.path.join(os.getcwd(), "../logs"))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Set up logging
logging.basicConfig(
    filename=os.path.join(log_dir, 'model_building_and_evaluation.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def model_neural_network(classifier, x_train, y_train, x_test, y_test):
    classifier.fit(x_train, y_train, epochs=10, batch_size=32)
    prediction = classifier.predict(x_test)
    print("ROC_AUC Score : ", '{0:.2%}'.format(roc_auc_score(y_test, (prediction > 0.5).astype(int))))
    fpr, tpr, _ = roc_curve(y_test, (prediction > 0.5).astype(int))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC AUC Plot')
    plt.show()

def model(classifier, x_train, y_train, x_test, y_test):
    if isinstance(classifier, Sequential):
        classifier.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
        prediction = classifier.predict(x_test)
        prediction_proba = prediction  # For Sequential models, use prediction directly as proba
        print("ROC_AUC Score: ", '{0:.2%}'.format(roc_auc_score(y_test, (prediction > 0.5).astype(int))))
    else:
        classifier.fit(x_train, y_train)
        prediction = classifier.predict(x_test)
        if hasattr(classifier, 'predict_proba'):
            prediction_proba = classifier.predict_proba(x_test)[:, 1]
        else:
            prediction_proba = prediction
        print("ROC_AUC Score: ", '{0:.2%}'.format(roc_auc_score(y_test, prediction_proba)))
    
    fpr, tpr, _ = roc_curve(y_test, prediction_proba)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

def model_other(classifier, x_train, y_train, x_test, y_test):
    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test)
    
    if hasattr(classifier, 'predict_proba'):
        prediction_proba = classifier.predict_proba(x_test)
        print("ROC_AUC Score : ", '{0:.2%}'.format(roc_auc_score(y_test, prediction_proba[:, 1])))
        fpr, tpr, _ = roc_curve(y_test, prediction_proba[:, 1])
    else:
        print("ROC_AUC Score is not available for this classifier")
        fpr, tpr, _ = roc_curve(y_test, prediction)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

def model_evaluation(classifier, x_test, y_test):
    if hasattr(classifier, 'predict_proba'):
        prediction_proba = classifier.predict_proba(x_test)
        prediction = np.argmax(prediction_proba, axis=1)
    else:
        prediction = (classifier.predict(x_test) > 0.5).astype(int)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, prediction)
    names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    counts = [value for value in cm.flatten()]
    percentages = ['{0:.2%}'.format(value) for value in cm.flatten() / np.sum(cm)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names, counts, percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cm, annot=labels, cmap='Blues', fmt='')
    
    # Classification Report
    print(classification_report(y_test, prediction))

def cross_validation_score(classifier, x_train, y_train):
    if isinstance(classifier, Sequential):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_index, val_index in kf.split(x_train):
            X_train_fold, X_val_fold = x_train[train_index], x_train[val_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            classifier.fit(X_train_fold, y_train_fold, epochs=10, batch_size=32)
            scores.append(classifier.evaluate(X_val_fold, y_val_fold)[1])
        print("Cross Validation Score: ", np.mean(scores))
    else:
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        scores = cross_val_score(classifier, x_train, y_train, cv=cv, scoring='roc_auc')
        print("Cross Validation Score: ", '{0:.2%}'.format(scores.mean()))
