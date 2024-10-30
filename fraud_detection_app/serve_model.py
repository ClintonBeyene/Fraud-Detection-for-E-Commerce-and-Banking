from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
import plotly.express as px
from dotenv import load_dotenv
import os
import warnings
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from config import Config
from models import db, User, FraudData
from datetime import datetime
from flask_migrate import Migrate
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize database
db.init_app(app)

# Load data
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data/fraud_data_with_country.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model/random_forest_model.pkl')

# Load the data
data = pd.read_csv(DATA_PATH)
data['signup_time'] = pd.to_datetime(data['signup_time'])
data['purchase_time'] = pd.to_datetime(data['purchase_time'])


def feature_engineering(data):
    # Create a copy of the data to avoid modifying the original
    data = data.copy()

    # Convert timestamp strings to datetime objects
    data['signup_time'] = pd.to_datetime(data['signup_time'])
    data['purchase_time'] = pd.to_datetime(data['purchase_time'])

    # Extract time features
    data['signup_time_hour'] = data['signup_time'].dt.hour
    data['signup_time_day'] = data['signup_time'].dt.day
    data['signup_time_month'] = data['signup_time'].dt.month
    data['signup_time_year'] = data['signup_time'].dt.year

    data['purchase_time_hour'] = data['purchase_time'].dt.hour
    data['purchase_time_day'] = data['purchase_time'].dt.day
    data['purchase_time_month'] = data['purchase_time'].dt.month
    data['purchase_time_year'] = data['purchase_time'].dt.year
    
    data['purchase_value'] = pd.to_numeric(data['purchase_value'])
        
    # Transaction frequency
    data['transaction_count'] = data.groupby('user_id')['user_id'].transform('count')
        
    # Transaction velocity (transactions per day)
    data['user_activity_period_days'] = data.groupby('user_id')['purchase_time'].transform(lambda x: (x.max() - x.min()).total_seconds() / 86400)
    data['transaction_velocity'] = data['transaction_count'] / data['user_activity_period_days'].clip(lower=1)
        
    # Average purchase value per user
    data['avg_purchase'] = data.groupby('user_id')['purchase_value'].transform('mean')
    
    # Calculate time difference
    data['time_to_purchase'] = (data['purchase_time'] - data['signup_time']).dt.total_seconds()

    # Column seconds_since_signup
    data["seconds_since_signup"] = (data.purchase_time - data.signup_time).dt.total_seconds()

    # Column "quick_purchase"
    data["quick_purchase"] = data.seconds_since_signup.apply(lambda x : 1 if x < 30 else 0)
    
    # Handle age groups
    age_group_columns = [
        'age_group_18-24', 'age_group_25-34', 'age_group_35-44',
        'age_group_45-54', 'age_group_55-64', 'age_group_65+'
    ]
    
    # Initialize all age group columns to 0
    for col in age_group_columns:
        data[col] = 0
    
    # Determine age group and set appropriate column to 1
    age = data['age'].iloc[0]
    if 18 <= age <= 24:
        data['age_group_18-24'] = 1
    elif 25 <= age <= 34:
        data['age_group_25-34'] = 1
    elif 35 <= age <= 44:
        data['age_group_35-44'] = 1
    elif 45 <= age <= 54:
        data['age_group_45-54'] = 1
    elif 55 <= age <= 64:
        data['age_group_55-64'] = 1
    else:
        data['age_group_65+'] = 1


    # Create dummy variables for sex (corrected version)
    data['sex_F'] = 0  # Initialize both columns to 0
    data['sex_M'] = 0
    
    # Set the appropriate sex column to 1 based on input
    current_sex = data['sex'].iloc[0]  # Get the sex value from input
    if current_sex == 'Female':
        data['sex_F'] = 1
    elif current_sex == 'Male':
        data['sex_M'] = 1

    # Define the exact list of countries from your model
    all_countries = [
        'Afghanistan', 'Albania', 'Algeria', 'Angola', 'Antigua and Barbuda', 'Argentina',
        'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh',
        'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bermuda', 'Bhutan',
        'Bolivia', 'Bonaire; Sint Eustatius; Saba', 'Bosnia and Herzegowina', 'Botswana',
        'Brazil', 'British Indian Ocean Territory', 'Brunei Darussalam', 'Bulgaria',
        'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon', 'Canada', 'Cape Verde',
        'Cayman Islands', 'Chile', 'China', 'Colombia', 'Congo',
        'Congo The Democratic Republic of The', 'Costa Rica', "Cote D'ivoire",
        'Croatia (LOCAL Name: Hrvatska)', 'Cuba', 'Curacao', 'Cyprus', 'Czech Republic',
        'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt',
        'El Salvador', 'Estonia', 'Ethiopia', 'European Union', 'Faroe Islands', 'Fiji',
        'Finland', 'France', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Gibraltar',
        'Greece', 'Guadeloupe', 'Guam', 'Guatemala', 'Haiti', 'Honduras', 'Hong Kong',
        'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran (ISLAMIC Republic Of)', 'Iraq',
        'Ireland', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya',
        'Korea Republic of', 'Kuwait', 'Kyrgyzstan', "Lao People's Democratic Republic",
        'Latvia', 'Lebanon', 'Lesotho', 'Libyan Arab Jamahiriya', 'Liechtenstein',
        'Lithuania', 'Luxembourg', 'Macau', 'Macedonia', 'Madagascar', 'Malawi', 'Malaysia',
        'Maldives', 'Malta', 'Mauritius', 'Mexico', 'Moldova Republic of', 'Monaco',
        'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nauru',
        'Nepal', 'Netherlands', 'New Caledonia', 'New Zealand', 'Nicaragua', 'Niger',
        'Nigeria', 'Norway', 'Oman', 'Pakistan', 'Palestinian Territory Occupied',
        'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland',
        'Portugal', 'Puerto Rico', 'Qatar', 'Reunion', 'Romania', 'Russian Federation',
        'Rwanda', 'Saint Kitts and Nevis', 'Saint Martin', 'San Marino', 'Saudi Arabia',
        'Senegal', 'Serbia', 'Seychelles', 'Singapore', 'Slovakia (SLOVAK Republic)',
        'Slovenia', 'South Africa', 'South Sudan', 'Spain', 'Sri Lanka', 'Sudan',
        'Sweden', 'Switzerland', 'Syrian Arab Republic', 'Taiwan; Republic of China (ROC)',
        'Tajikistan', 'Tanzania United Republic of', 'Thailand', 'Trinidad and Tobago',
        'Tunisia', 'Turkey', 'Turkmenistan', 'Uganda', 'Ukraine', 'United Arab Emirates',
        'United Kingdom', 'United States', 'Uruguay', 'Uzbekistan', 'Vanuatu',
        'Venezuela', 'Viet Nam', 'Virgin Islands (U.S.)', 'Yemen', 'Zambia', 'Zimbabwe'
    ]

    # Create country dummy variables
    for country in all_countries:
        data[f'country_{country}'] = (data['country'] == country).astype(int)

    # Drop unnecessary columns
    columns_to_drop = [
        "user_id", "signup_time", "purchase_time",
        "ip_address", "device_id", "source", " browser", "age"
    ]
    data = data.drop(columns=columns_to_drop, errors='ignore')

    # Define the expected order of features
    expected_order = [
        'purchase_value', 'signup_time_hour', 'signup_time_day', 'signup_time_year', 'signup_time_month',
        'purchase_time_hour', 'purchase_time_day', 'purchase_time_year', 'purchase_time_month',
        'transaction_count', 'user_activity_period_days', 'transaction_velocity', 'avg_purchase',
        'time_to_purchase', 'seconds_since_signup', 'quick_purchase', 'sex_F', 'sex_M',
    ] + [f'country_{country}' for country in all_countries] + age_group_columns 

    # Reorder the columns to match the expected order
    data = data[expected_order]

    # Normalize numeric features
    numeric_cols = data.select_dtypes(include=['int64', 'float64', 'int32']).columns
    numeric_cols = [col for col in numeric_cols if not col.startswith(('country_', 'sex_', 'age_group_'))]
    
    if len(numeric_cols) > 0:
        rob_scaler = RobustScaler()
        data[numeric_cols] = rob_scaler.fit_transform(data[numeric_cols])

    return data

# Routes
@app.route('/')
def home():
    return render_template('home.html')


# Initialize Dash app
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/dashboard/', 
                     external_stylesheets=['/static/css/dashboard.css'])

# Dash layout
dash_app.layout = html.Div([
    html.Div([
        html.H1('Fraud Detection Dashboard', className='dashboard-title'),
        
        # Summary Statistics Cards
        html.Div([
            html.Div([
                html.H3('Total Transactions'),
                html.P(id='total-transactions')
            ], className='stat-card'),
            html.Div([
                html.H3('Average Purchase Value'),
                html.P(id='avg-purchase-value')
            ], className='stat-card'),
            html.Div([
                html.H3('Fraud Rate'),
                html.P(id='fraud-rate')
            ], className='stat-card')
        ], className='stats-container'),

        # Filters
        html.Div([
            html.Div([
                html.Label('Date Range:'),
                dcc.DatePickerRange(
                    id='date-range',
                    start_date=data['signup_time'].min(),
                    end_date=data['signup_time'].max()
                )
            ], className='filter-item'),
            html.Div([
                html.Label('Source:'),
                dcc.Dropdown(
                    id='source-filter',
                    options=[{'label': src, 'value': src} for src in data['source'].unique()],
                    multi=True
                )
            ], className='filter-item')
        ], className='filters-container'),

        # Charts Grid
        html.Div([
            html.Div([
                html.Div([
                    html.H3('Purchase Value Distribution'),
                    dcc.Graph(id='purchase-dist-chart')
                ], className='chart-card'),
                html.Div([
                    html.H3('Fraud by Time of Day'),
                    dcc.Graph(id='time-pattern-chart')
                ], className='chart-card')
            ], className='chart-row'),
            html.Div([
                html.Div([
                    html.H3('Country Distribution'),
                    dcc.Graph(id='country-chart')
                ], className='chart-card'),
                html.Div([
                    html.H3('Age Distribution'),
                    dcc.Graph(id='age-chart')
                ], className='chart-card')
            ], className='chart-row')
        ], className='charts-grid')
    ], className='dashboard-container')
])

# Dash callbacks
@dash_app.callback(
    [Output('total-transactions', 'children'),
     Output('avg-purchase-value', 'children'),
     Output('fraud-rate', 'children')],
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('source-filter', 'value')]
)
def update_stats(start_date, end_date, sources):
    filtered_df = data.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['signup_time'] >= start_date) & 
            (filtered_df['signup_time'] <= end_date)
        ]
    
    if sources:
        filtered_df = filtered_df[filtered_df['source'].isin(sources)]
    
    total = len(filtered_df)
    avg_purchase = filtered_df['purchase_value'].mean()
    fraud_rate = (filtered_df['class'].mean() * 100)
    
    return f"{total:,}", f"${avg_purchase:.2f}", f"{fraud_rate:.2f}%"

@dash_app.callback(
    Output('purchase-dist-chart', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('source-filter', 'value')]
)
def update_purchase_distribution(start_date, end_date, sources):
    filtered_df = data.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['signup_time'] >= start_date) & 
            (filtered_df['signup_time'] <= end_date)
        ]
    
    if sources:
        filtered_df = filtered_df[filtered_df['source'].isin(sources)]
    
    fig = px.histogram(
        filtered_df,
        x='purchase_value',
        color='class',
        nbins=50,
        title='Purchase Value Distribution',
        labels={'purchase_value': 'Purchase Value ($)', 'class': 'Class'}
    )
    return fig

@dash_app.callback(
    Output('time-pattern-chart', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('source-filter', 'value')]
)
def update_time_patterns(start_date, end_date, sources):
    filtered_df = data.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['signup_time'] >= start_date) & 
            (filtered_df['signup_time'] <= end_date)
        ]
    
    if sources:
        filtered_df = filtered_df[filtered_df['source'].isin(sources)]
    
    filtered_df['hour'] = pd.to_datetime(filtered_df['purchase_time']).dt.hour
    hourly_data = filtered_df.groupby(['hour', 'class']).size().reset_index(name='count')
    
    fig = px.line(
        hourly_data,
        x='hour',
        y='count',
        color='class',
        title='Transactions by Hour of Day',
        labels={'hour': 'Hour of Day', 'count': 'Number of Transactions'}
    )
    return fig

@dash_app.callback(
    Output('country-chart', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('source-filter', 'value')]
)
def update_country_distribution(start_date, end_date, sources):
    filtered_df = data.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['signup_time'] >= start_date) & 
            (filtered_df['signup_time'] <= end_date)
        ]
    
    if sources:
        filtered_df = filtered_df[filtered_df['source'].isin(sources)]
    
    country_data = filtered_df.groupby(['country', 'class']).size().reset_index(name='count')
    
    fig = px.bar(
        country_data,
        x='country',
        y='count',
        color='class',
                title='Transactions by Country',
        labels={'country': 'Country', 'count': 'Number of Transactions'}
    )
    return fig

@dash_app.callback(
    Output('age-chart', 'figure'),
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('source-filter', 'value')]
)
def update_age_distribution(start_date, end_date, sources):
    filtered_df = data.copy()
    
    if start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['signup_time'] >= start_date) & 
            (filtered_df['signup_time'] <= end_date)
        ]
    
    if sources:
        filtered_df = filtered_df[filtered_df['source'].isin(sources)]
    
    fig = px.histogram(
        filtered_df,
        x='age',
        color='class',
        nbins=30,
        title='Age Distribution',
        labels={'age': 'Age', 'class': 'Class'}
    )
    return fig


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            # Extract data from form
            transaction_data = {
                'user_id': request.form['user_id'],
                'signup_time': datetime.strptime(request.form['signup_time'], '%Y-%m-%dT%H:%M'),
                'purchase_time': datetime.strptime(request.form['purchase_time'], '%Y-%m-%dT%H:%M'),
                'purchase_value': float(request.form['purchase_value']),
                'device_id': request.form['device_id'],
                'source': request.form['source'],
                'browser': request.form['browser'],
                'sex': request.form['sex'],
                'age': int(request.form['age']),
                'ip_address': request.form['ip_address'],
                'country': request.form['country']
            }

            # Create a new transaction
            new_transaction = FraudData(**transaction_data)
            
            # Save to database
            db.session.add(new_transaction)
            db.session.commit()

            # Prepare data for prediction
            pred_data = pd.DataFrame([transaction_data])
            pred_data = feature_engineering(pred_data)
            
            # Load model and make prediction
            model = joblib.load(MODEL_PATH)
            prediction = model.predict(pred_data)[0]
            probability = float(model.predict_proba(pred_data)[0][1])

            # Update the transaction with the prediction
            new_transaction.is_fraud = bool(prediction)
            db.session.commit()

            result = "Potential fraud" if prediction else "Not fraud"
            flash(f'Transaction registered successfully. Fraud check result: {result} (Probability: {probability:.2f})', 'info')
            return redirect(url_for('register'))

        except Exception as e:
            db.session.rollback()
            flash(f'An error occurred: {str(e)}', 'danger')
            return redirect(url_for('register'))

    return render_template('register.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', dash_app=dash_app.index())


# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

migrate = Migrate(app, db)


if __name__ == '__main__':
    app.run(debug=True)
