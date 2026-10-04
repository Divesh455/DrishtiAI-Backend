import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


# ---------------------------------------------------------
# Load backend/.env
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = BASE_DIR / "backend" / ".env"

load_dotenv(ENV_FILE)


# ---------------------------------------------------------
# Database URL
# ---------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. "
        "Please add your Neon PostgreSQL connection string "
        "to backend/.env"
    )


# ---------------------------------------------------------
# SQLAlchemy Engine
# ---------------------------------------------------------

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
)


# ---------------------------------------------------------
# Session
# ---------------------------------------------------------

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


# ---------------------------------------------------------
# Base model
# ---------------------------------------------------------

Base = declarative_base()


# ---------------------------------------------------------
# Database session dependency
# ---------------------------------------------------------

def get_db():
    """
    Provides a database session for FastAPI endpoints.
    """

    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------
# Connection test
# ---------------------------------------------------------

def test_database_connection():
    """
    Test connection to Neon PostgreSQL.
    """

    from sqlalchemy import text

    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            result.fetchone()

        return True

    except Exception as exc:
        print(f"Database connection failed: {exc}")
        return False