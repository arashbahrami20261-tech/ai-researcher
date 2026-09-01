"""
Database engine + session setup.

SQLite for the MVP (zero setup — it's just a file). Switching to Postgres
later is a one-line change here (`DATABASE_URL`), not a rewrite, because
every other file talks to SQLAlchemy's ORM layer, never to SQLite directly.
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import Base

DATABASE_URL = "sqlite:///ai_researcher.db"

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


def init_db() -> None:
    """Create all tables if they don't already exist."""
    Base.metadata.create_all(engine)
