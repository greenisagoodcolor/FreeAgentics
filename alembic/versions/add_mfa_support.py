"""Add MFA support.

Revision ID: add_mfa_support
Revises: c42749d3c630
Create Date: 2025-01-16 12:00:00.000000

"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op  # type: ignore[attr-defined]

# revision identifiers, used by Alembic.
revision = "add_mfa_support"
down_revision = "c42749d3c630"
branch_labels = None
depends_on = None


def upgrade():
    """Add MFA support tables and columns."""
    # Create MFA settings table
    op.create_table(
        "mfa_settings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("totp_secret", sa.Text(), nullable=True),
        sa.Column("backup_codes", sa.Text(), nullable=True),
        sa.Column("hardware_keys", sa.Text(), nullable=True),
        sa.Column("is_enabled", sa.Boolean(), nullable=True),
        sa.Column("enrollment_date", sa.DateTime(), nullable=True),
        sa.Column("last_used", sa.DateTime(), nullable=True),
        sa.Column("failed_attempts", sa.Integer(), nullable=True),
        sa.Column("locked_until", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id"),
    )

    # Create index for user_id lookups
    op.create_index("ix_mfa_settings_user_id", "mfa_settings", ["user_id"])

    # Create MFA audit log table for enhanced security monitoring
    op.create_table(
        "mfa_audit_log",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("event_type", sa.String(length=50), nullable=False),
        sa.Column("method", sa.String(length=50), nullable=True),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("ip_address", sa.String(length=45), nullable=True),
        sa.Column("user_agent", sa.String(length=500), nullable=True),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for audit log
    op.create_index("ix_mfa_audit_log_user_id", "mfa_audit_log", ["user_id"])
    op.create_index(
        "ix_mfa_audit_log_timestamp", "mfa_audit_log", ["timestamp"]
    )
    op.create_index(
        "ix_mfa_audit_log_event_type", "mfa_audit_log", ["event_type"]
    )

    # Add MFA-related columns to users table (if it exists)
    # This assumes you have a users table; adjust as needed
    try:
        op.add_column(
            "users", sa.Column("mfa_enabled", sa.Boolean(), default=False)
        )
        op.add_column(
            "users", sa.Column("mfa_enforced", sa.Boolean(), default=False)
        )
        op.add_column(
            "users",
            sa.Column("last_mfa_verification", sa.DateTime(), nullable=True),
        )
    except Exception:
        # Table might not exist or columns might already exist
        pass


def downgrade():
    """Remove MFA support."""
    # Remove MFA audit log table
    op.drop_table("mfa_audit_log")

    # Remove MFA settings table
    op.drop_table("mfa_settings")

    # Remove MFA columns from users table
    try:
        op.drop_column("users", "last_mfa_verification")
        op.drop_column("users", "mfa_enforced")
        op.drop_column("users", "mfa_enabled")
    except Exception:
        # Columns might not exist
        pass
