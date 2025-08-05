"""add_agent_conversation_schema

Revision ID: e92552ddd3d6
Revises: add_user_settings_001
Create Date: 2025-08-01 01:13:52.341775

This migration adds the agent conversation database schema as specified in Task 39.
Creates tables for:
- agent_conversation_sessions: Multi-agent conversation sessions
- agent_conversation_messages: Messages within conversations
- agent_conversations: Many-to-many relationship between agents and conversations

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# Import GUID type for proper UUID handling
try:
    from database.types import GUID
except ImportError:
    # Fallback for migration environments where imports might fail
    from sqlalchemy_utils import UUIDType as GUID


# revision identifiers, used by Alembic.
revision: str = "e92552ddd3d6"
down_revision: Union[str, Sequence[str], None] = "add_user_settings_001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema to add agent conversation tables."""

    # Create agent_conversation_sessions table
    op.create_table(
        "agent_conversation_sessions",
        sa.Column("id", GUID(), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("title", sa.String(255), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "status",
            sa.Enum(
                "pending", "active", "completed", "failed", "cancelled", name="conversationstatus"
            ),
            nullable=False,
            default="pending",
        ),
        sa.Column("message_count", sa.Integer(), nullable=False, default=0),
        sa.Column("agent_count", sa.Integer(), nullable=False, default=0),
        sa.Column("max_turns", sa.Integer(), nullable=False, default=5),
        sa.Column("current_turn", sa.Integer(), nullable=False, default=0),
        sa.Column("llm_provider", sa.String(50), nullable=True),
        sa.Column("llm_model", sa.String(100), nullable=True),
        sa.Column("config", sa.JSON(), nullable=False, default={}),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
        sa.Column("user_id", sa.String(255), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for agent_conversation_sessions
    op.create_index(
        "ix_agent_conversation_sessions_user_id", "agent_conversation_sessions", ["user_id"]
    )
    op.create_index(
        "ix_agent_conversation_sessions_status", "agent_conversation_sessions", ["status"]
    )
    op.create_index(
        "ix_agent_conversation_sessions_created_at", "agent_conversation_sessions", ["created_at"]
    )

    # Create agent_conversations association table (many-to-many)
    op.create_table(
        "agent_conversations",
        sa.Column("agent_id", GUID(), nullable=False),
        sa.Column("conversation_id", GUID(), nullable=False),
        sa.Column("role", sa.String(50), nullable=False, default="participant"),
        sa.Column("joined_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("left_at", sa.DateTime(), nullable=True),
        sa.Column("message_count", sa.Integer(), nullable=False, default=0),
        sa.ForeignKeyConstraint(
            ["agent_id"],
            ["agents.id"],
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["agent_conversation_sessions.id"],
        ),
        sa.PrimaryKeyConstraint("agent_id", "conversation_id"),
    )

    # Create indexes for agent_conversations
    op.create_index("ix_agent_conversations_agent_id", "agent_conversations", ["agent_id"])
    op.create_index(
        "ix_agent_conversations_conversation_id", "agent_conversations", ["conversation_id"]
    )
    op.create_index("ix_agent_conversations_joined_at", "agent_conversations", ["joined_at"])

    # Create agent_conversation_messages table
    op.create_table(
        "agent_conversation_messages",
        sa.Column("id", GUID(), nullable=False),
        sa.Column("conversation_id", GUID(), nullable=False),
        sa.Column("agent_id", GUID(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("message_order", sa.Integer(), nullable=False),
        sa.Column("turn_number", sa.Integer(), nullable=False, default=1),
        sa.Column("role", sa.String(50), nullable=False, default="assistant"),
        sa.Column("message_type", sa.String(50), nullable=False, default="text"),
        sa.Column("message_metadata", sa.JSON(), nullable=False, default={}),
        sa.Column("is_processed", sa.Boolean(), nullable=False, default=False),
        sa.Column("processing_time_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["agent_id"],
            ["agents.id"],
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["agent_conversation_sessions.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for agent_conversation_messages for optimal query performance
    op.create_index(
        "ix_agent_conversation_messages_conversation_id",
        "agent_conversation_messages",
        ["conversation_id"],
    )
    op.create_index(
        "ix_agent_conversation_messages_agent_id", "agent_conversation_messages", ["agent_id"]
    )
    op.create_index(
        "ix_agent_conversation_messages_created_at", "agent_conversation_messages", ["created_at"]
    )

    # Composite indexes for common query patterns as recommended by Addy Osmani
    op.create_index(
        "ix_agent_conversation_messages_conv_order",
        "agent_conversation_messages",
        ["conversation_id", "message_order"],
    )
    op.create_index(
        "ix_agent_conversation_messages_agent_created",
        "agent_conversation_messages",
        ["agent_id", "created_at"],
    )
    op.create_index(
        "ix_agent_conversation_messages_turn_order",
        "agent_conversation_messages",
        ["turn_number", "message_order"],
    )


def downgrade() -> None:
    """Downgrade schema to remove agent conversation tables."""

    # Drop indexes first
    op.drop_index("ix_agent_conversation_messages_turn_order", "agent_conversation_messages")
    op.drop_index("ix_agent_conversation_messages_agent_created", "agent_conversation_messages")
    op.drop_index("ix_agent_conversation_messages_conv_order", "agent_conversation_messages")
    op.drop_index("ix_agent_conversation_messages_created_at", "agent_conversation_messages")
    op.drop_index("ix_agent_conversation_messages_agent_id", "agent_conversation_messages")
    op.drop_index("ix_agent_conversation_messages_conversation_id", "agent_conversation_messages")

    # Drop agent_conversation_messages table
    op.drop_table("agent_conversation_messages")

    # Drop agent_conversations association table indexes
    op.drop_index("ix_agent_conversations_joined_at", "agent_conversations")
    op.drop_index("ix_agent_conversations_conversation_id", "agent_conversations")
    op.drop_index("ix_agent_conversations_agent_id", "agent_conversations")

    # Drop agent_conversations association table
    op.drop_table("agent_conversations")

    # Drop agent_conversation_sessions indexes
    op.drop_index("ix_agent_conversation_sessions_created_at", "agent_conversation_sessions")
    op.drop_index("ix_agent_conversation_sessions_status", "agent_conversation_sessions")
    op.drop_index("ix_agent_conversation_sessions_user_id", "agent_conversation_sessions")

    # Drop agent_conversation_sessions table
    op.drop_table("agent_conversation_sessions")

    # Drop the enum type
    op.execute("DROP TYPE IF EXISTS conversationstatus")
