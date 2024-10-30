from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(100), unique=True, nullable=False)
    signup_time = db.Column(db.DateTime, nullable=False)
    purchase_time = db.Column(db.DateTime, nullable=False)
    purchase_value = db.Column(db.Float, nullable=False)
    device_id = db.Column(db.String(100), nullable=False)
    source = db.Column(db.String(100), nullable=False)
    browser = db.Column(db.String(100), nullable=False)
    sex = db.Column(db.String(1), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    ip_address = db.Column(db.String(50), nullable=False)
    country = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<User {self.user_id}>'


class FraudData(db.Model):
    __tablename__ = 'fraud_data'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(50), nullable=False)
    signup_time = db.Column(db.DateTime, nullable=False)
    purchase_time = db.Column(db.DateTime, nullable=False)
    purchase_value = db.Column(db.Float, nullable=False)
    device_id = db.Column(db.String(50), nullable=False)
    source = db.Column(db.String(50), nullable=False)
    browser = db.Column(db.String(50), nullable=False)
    sex = db.Column(db.String(1), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    ip_address = db.Column(db.String(50), nullable=False)
    country = db.Column(db.String(50), nullable=False)
    is_fraud = db.Column(db.Boolean, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<FraudData {self.id}>'