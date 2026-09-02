"""
Database engine + session setup.

SQLite for now (zero setup — it's just a file). Switching to Postgres later
is a one-line change to DATABASE_URL here, not a rewrite, because every
other file talks to SQLAlchemy's ORM layer, never to SQLite directly.
`migrations/env.py` imports DATABASE_URL from this file, so there is a
single source of truth for where the database lives.
"""

from __future__ import annotations

import os

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from database.models import Base

# Overridable via environment so tests and a future Postgres deployment can
# point elsewhere without editing code.
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///ai_researcher.db")

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


def ensure_migrated() -> None:
    """
    Check that Alembic migrations have been applied, and refuse to run if not.

    This replaces the old `init_db()` call in the CLI. `create_all()` can
    create a missing table but can never *alter* an existing one, so once the
    database holds real research data, a schema change would silently do
    nothing and the app would fail with a confusing error much later.
    Failing loudly here, with the exact command to fix it, is better.
    """
    tables = inspect(engine).get_table_names()

    if "alembic_version" not in tables:
        raise RuntimeError(
            "the database has not been migrated. Run:  alembic upgrade head"
        )


def init_db() -> None:
    """
    Create all tables directly from the models, bypassing migrations.

    Kept only for tests, which build a throwaway in-memory database where
    migration history is irrelevant. Do not call this against a real
    database — use `alembic upgrade head` instead.
    """
    Base.metadata.create_all(engine)
