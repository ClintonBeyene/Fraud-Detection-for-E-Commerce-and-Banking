from serve_model import app, db
from models import User, FraudData

def init_database():
    with app.app_context():
        # Create all tables
        db.create_all()

        # Add some initial data if needed
        if User.query.count() == 0:
            admin = User(username='admin', email='admin@example.com')
            db.session.add(admin)
            db.session.commit()

if __name__ == '__main__':
    init_database()
    print("Database initialized successfully!")