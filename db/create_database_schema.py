from db.database import create_tables

def create_database_schema():
    """Create database schema using SQLAlchemy."""
    create_tables()
    print("Database tables created successfully")