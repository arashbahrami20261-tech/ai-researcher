"""
Alembic environment configuration.

Why this file matters: without migrations, the only way to create tables is
`Base.metadata.create_all()`, which can create a table but can never *alter*
one. The moment a column needs to change on a database that already holds
real research data, create_all() silently does nothing and the app breaks.
Migrations record each schema change as a versioned, reversible step.
"""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# Alembic runs from the project root but doesn't add it to the import path
# itself, so `database.models` would fail to import without this line.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import DATABASE_URL
from database.models import Base

config = context.config

# Inject the URL from database/db.py rather than reading it from alembic.ini,
# so there is a single source of truth for where the database lives.
config.set_main_option("sqlalchemy.url", DATABASE_URL)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# This is what lets `alembic revision --autogenerate` compare the SQLAlchemy
# models against the live database and write the difference for you.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Generate SQL text without connecting to a database."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # SQLite cannot ALTER most columns in place. "batch mode" makes
        # Alembic rebuild the table instead, which is the only way column
        # changes work on SQLite at all.
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect to the database and apply migrations directly."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
