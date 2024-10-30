CREATE TABLE fraud_data (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    signup_time TIMESTAMP NOT NULL,
    purchase_time TIMESTAMP NOT NULL,
    purchase_value FLOAT NOT NULL,
    device_id VARCHAR(50) NOT NULL,
    source VARCHAR(50) NOT NULL,
    browser VARCHAR(50) NOT NULL,
    sex VARCHAR(1) NOT NULL,
    age INTEGER NOT NULL,
    ip_address VARCHAR(50) NOT NULL,
    country VARCHAR(50) NOT NULL,
    is_fraud BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);