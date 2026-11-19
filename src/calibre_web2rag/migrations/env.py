"""Alembic environment: runs the migrations against DATABASE_URL.

calibre_web2rag.db.migrate() sets ``sqlalchemy.url`` on the config; nothing
here reads alembic.ini.
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine, pool, text

# The pipeline's sources share one database; each keeps its own version table.
VERSION_TABLE = "calibre_web2rag_alembic_version"

url = context.config.get_main_option("sqlalchemy.url")
if not url:
    raise RuntimeError(
        "sqlalchemy.url is not set; run migrations through calibre_web2rag.db.migrate()"
    )

engine = create_engine(url, poolclass=pool.NullPool)
with engine.connect() as connection:
    context.configure(connection=connection, version_table=VERSION_TABLE)
    with context.begin_transaction():
        # Every run migrates first; when two runs start together on a fresh
        # database, the second waits here, then finds the schema current.
        connection.execute(
            text("SELECT pg_advisory_xact_lock(hashtext(:key))"), {"key": VERSION_TABLE}
        )
        context.run_migrations()
