import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our database models and configuration
from database.base import Base

# Import all models so they are available for autogenerate
from database.models import *
from database.session import DATABASE_URL
from database.types import GUID

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Override database URL from environment
if DATABASE_URL:
    config.set_main_option("sqlalchemy.url", DATABASE_URL)
else:
    # Use default SQLite URL for migrations when no DATABASE_URL is set
    config.set_main_option("sqlalchemy.url", "sqlite:///./freeagentics.db")

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def render_item(type_, obj, autogen_context):
    """Custom rendering for UUID type in migrations."""
    if type_ == "type" and isinstance(obj, type(GUID)):
        # Import the custom GUID type in the migration
        autogen_context.imports.add("from database.types import GUID")
        return "GUID()"
    return False


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_item=render_item,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
