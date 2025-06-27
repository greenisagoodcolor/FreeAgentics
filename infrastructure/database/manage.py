"""
Database management script for FreeAgentics.

This script provides utilities for:
- Creating and dropping databases
- Running migrations
- Seeding test data
- Database health checks
"""

import os
import sys
from pathlib import Path

import click
import psycopg2
from alembic import command
from alembic.config import Config
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from .connection import DATABASE_URL, engine
from .seed import seed_demo_data, seed_development_data


def get_db_config():
    """Extract database configuration from DATABASE_URL."""
    url_parts = DATABASE_URL.replace("postgresql://", "").split("@")
    user_pass = url_parts[0].split(":")
    host_port_db = url_parts[1].split("/")
    host_port = host_port_db[0].split(":")
    return {
        "user": user_pass[0],
        "password": user_pass[1] if len(user_pass) > 1 else "",
        "host": host_port[0],
        "port": host_port[1] if len(host_port) > 1 else "5432",
        "database": host_port_db[1] if len(host_port_db) > 1 else "freeagentics_dev",
    }


@click.group()
def cli():
    """FreeAgentics database management commands."""
    pass


@cli.command()
@click.option("--force", is_flag=True, help="Force create (drop if exists)")
def create_db(force):
    """Create the database."""
    config = get_db_config()
    conn = psycopg2.connect(
        host=config["host"],
        port=config["port"],
        user=config["user"],
        password=config["password"],
        database="postgres",
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    try:
        if force:
            click.echo(f"Dropping database {config['database']} if exists...")
            cur.execute(f"DROP DATABASE IF EXISTS {config['database']}")
        click.echo(f"Creating database {config['database']}...")
        cur.execute(f"CREATE DATABASE {config['database']}")
        click.echo("Database created successfully!")
    except psycopg2.errors.DuplicateDatabase:
        click.echo(f"Database {config['database']} already exists. Use --force to recreate.")
    finally:
        cur.close()
        conn.close()


@cli.command()
@click.confirmation_option(prompt= (
    "Are you sure you want to drop the database?"))
def drop_db():
    """Drop the database."""
    config = get_db_config()
    conn = psycopg2.connect(
        host=config["host"],
        port=config["port"],
        user=config["user"],
        password=config["password"],
        database="postgres",
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    try:
        click.echo(f"Dropping database {config['database']}...")
        cur.execute(f"DROP DATABASE IF EXISTS {config['database']}")
        click.echo("Database dropped successfully!")
    finally:
        cur.close()
        conn.close()


@cli.command()
def migrate():
    """Run database migrations."""
    click.echo("Running database migrations...")
    alembic_cfg = Config(Path(__file__).parent / "alembic.ini")
    command.upgrade(alembic_cfg, "head")
    click.echo("Migrations completed successfully!")


@cli.command()
@click.option("--revision", default= (
    "base", help="Target revision (default: base)"))
def rollback(revision):
    """Rollback database migrations."""
    click.echo(f"Rolling back to revision: {revision}...")
    alembic_cfg = Config(Path(__file__).parent / "alembic.ini")
    command.downgrade(alembic_cfg, revision)
    click.echo("Rollback completed successfully!")


@cli.command()
def status():
    """Show current migration status."""
    alembic_cfg = Config(Path(__file__).parent / "alembic.ini")
    command.current(alembic_cfg, verbose=True)


@cli.command()
@click.option("--env", type= (
    click.Choice(["development", "demo", "test"]), default="development"))
def seed(env):
    """Seed the database with test data."""
    click.echo(f"Seeding database for {env} environment...")
    if env == "development":
        seed_development_data()
    elif env == "demo":
        seed_demo_data()
    elif env == "test":
        click.echo("Test data seeding not implemented yet.")
    click.echo("Database seeded successfully!")


@cli.command()
def check():
    """Check database connection and health."""
    click.echo("Checking database connection...")
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.scalar()
            click.echo(f"✓ Connected to PostgreSQL: {version}")
            result = conn.execute(
                text(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = (
                        'public' AND table_name = 'alembic_version')")
                )
            )
            has_migrations = result.scalar()
            if has_migrations:
                result = (
                    conn.execute(text("SELECT version_num FROM alembic_version")))
                current_version = result.scalar()
                click.echo(f"✓ Migrations applied. Current version: {current_version}")
            else:
                click.echo("✗ No migrations applied yet. Run 'python manage.py migrate'")
            result = conn.execute(
                text(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
                )
            )
            table_count = result.scalar()
            click.echo(f"✓ Database has {table_count} tables")
    except Exception as e:
        click.echo(f"✗ Database connection failed: {e}")
        sys.exit(1)


@cli.command()
def init():
    """Initialize database (create + migrate + seed)."""
    click.echo("Initializing database...")
    ctx = click.get_current_context()
    ctx.invoke(create_db, force=True)
    ctx.invoke(migrate)
    if os.getenv("DATABASE_SEED_DATA", "false").lower() == "true":
        env = os.getenv("ENVIRONMENT", "development")
        ctx.invoke(seed, env=env)
    click.echo("Database initialization complete!")


if __name__ == "__main__":
    cli()
