"""Add agent creation fields for natural language agent creation

Revision ID: 007
Revises: 006
Create Date: 2025-07-31 01:30:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add new fields to agents table for natural language agent creation."""

    # Add agent_type enum
    agent_type_enum = sa.Enum(
        "advocate", "analyst", "critic", "creative", "moderator", name="agenttype"
    )
    agent_type_enum.create(op.get_bind())

    # Add new columns to agents table
    op.add_column(
        "agents",
        sa.Column(
            "agent_type",
            sa.Enum("advocate", "analyst", "critic", "creative", "moderator", name="agenttype"),
            nullable=True,
        ),
    )
    op.add_column("agents", sa.Column("system_prompt", sa.Text(), nullable=True))
    op.add_column(
        "agents",
        sa.Column(
            "personality_traits", sa.JSON(), server_default=sa.text("'{}'::json"), nullable=True
        ),
    )
    op.add_column(
        "agents",
        sa.Column("creation_source", sa.String(length=50), server_default="manual", nullable=True),
    )
    op.add_column("agents", sa.Column("source_prompt", sa.Text(), nullable=True))


def downgrade() -> None:
    """Remove agent creation fields."""

    # Remove columns
    op.drop_column("agents", "source_prompt")
    op.drop_column("agents", "creation_source")
    op.drop_column("agents", "personality_traits")
    op.drop_column("agents", "system_prompt")
    op.drop_column("agents", "agent_type")

    # Drop enum type
    op.execute("DROP TYPE agenttype")
