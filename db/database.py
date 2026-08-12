# db/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import Config
from db.models import Base


engine = create_engine(f"sqlite:///{Config.DATABASE_URL}", echo=False, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)


def create_tables() -> None:
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)


# def get_db():
#     """Yield a database session (FastAPI-style helper)."""
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()