"""Initial database schema

Revision ID: d3e985c2dfb8
Revises:
Create Date: 2025-06-18 09:53:31.460728

"""

from collections.abc import Sequence
from typing import Optional, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "d3e985c2dfb8"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial database schema"""

    # Create custom enums
    op.execute("CREATE TYPE agentstatus AS ENUM ('active', 'inactive', 'suspended', 'terminated')")
    op.execute("CREATE TYPE conversationtype AS ENUM ('direct', 'group', 'broadcast', 'system')")

    # Create agents table
    op.create_table(
        "agents",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("uuid", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("type", sa.String(length=100), nullable=False),
        sa.Column(
            "status",
            postgresql.ENUM("active", "inactive", "suspended", "terminated", name="agentstatus"),
            nullable=False,
        ),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("state", sa.JSON(), nullable=True),
        sa.Column("beliefs", sa.JSON(), nullable=True),
        sa.Column("location", sa.String(length=15), nullable=True),
        sa.Column("energy_level", sa.Float(), nullable=True),
        sa.Column("experience_points", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=True),
        sa.Column("last_active_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_agents_id"), "agents", ["id"], unique=False)
    op.create_index(op.f("ix_agents_uuid"), "agents", ["uuid"], unique=True)
    op.create_index("idx_agent_location", "agents", ["location"], unique=False)
    op.create_index("idx_agent_type_status", "agents", ["type", "status"], unique=False)

    # Create conversations table
    op.create_table(
        "conversations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("uuid", sa.String(length=36), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column(
            "type",
            postgresql.ENUM("direct", "group", "broadcast", "system", name="conversationtype"),
            nullable=True,
        ),
        sa.Column("meta_data", sa.JSON(), nullable=True),
        sa.Column("context", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=True),
        sa.Column("last_message_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_conversations_id"), "conversations", ["id"], unique=False)
    op.create_index(op.f("ix_conversations_uuid"), "conversations", ["uuid"], unique=True)

    # Create coalitions table
    op.create_table(
        "coalitions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("uuid", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("type", sa.String(length=50), nullable=True),
        sa.Column("goal", sa.JSON(), nullable=True),
        sa.Column("rules", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=True),
        sa.Column("value_pool", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("activated_at", sa.DateTime(), nullable=True),
        sa.Column("disbanded_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_coalitions_id"), "coalitions", ["id"], unique=False)
    op.create_index(op.f("ix_coalitions_uuid"), "coalitions", ["uuid"], unique=True)
    op.create_index("idx_coalition_status_type", "coalitions", ["status", "type"], unique=False)

    # Create system_logs table
    op.create_table(
        "system_logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("level", sa.String(length=20), nullable=False),
        sa.Column("component", sa.String(length=100), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("agent_id", sa.Integer(), nullable=True),
        sa.Column("conversation_id", sa.Integer(), nullable=True),
        sa.Column("coalition_id", sa.Integer(), nullable=True),
        sa.Column("data", sa.JSON(), nullable=True),
        sa.Column("error_trace", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["agent_id"], ["agents.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["coalition_id"], ["coalitions.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_system_logs_id"), "system_logs", ["id"], unique=False)
    op.create_index(op.f("ix_system_logs_timestamp"), "system_logs", ["timestamp"], unique=False)
    op.create_index(
        "idx_log_component_timestamp",
        "system_logs",
        ["component", "timestamp"],
        unique=False,
    )
    op.create_index("idx_log_timestamp_level", "system_logs", ["timestamp", "level"], unique=False)

    # Create knowledge_graphs table
    op.create_table(
        "knowledge_graphs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("uuid", sa.String(length=36), nullable=False),
        sa.Column("owner_id", sa.Integer(), nullable=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("type", sa.String(length=50), nullable=True),
        sa.Column("nodes", sa.JSON(), nullable=True),
        sa.Column("edges", sa.JSON(), nullable=True),
        sa.Column("meta_data", sa.JSON(), nullable=True),
        sa.Column("is_public", sa.Boolean(), nullable=True),
        sa.Column("access_list", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=True),
        sa.ForeignKeyConstraint(["owner_id"], ["agents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_knowledge_graphs_id"), "knowledge_graphs", ["id"], unique=False)
    op.create_index(op.f("ix_knowledge_graphs_uuid"), "knowledge_graphs", ["uuid"], unique=True)
    op.create_index(
        "idx_knowledge_owner_type",
        "knowledge_graphs",
        ["owner_id", "type"],
        unique=False,
    )

    # Create conversation_participants table
    op.create_table(
        "conversation_participants",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("conversation_id", sa.Integer(), nullable=True),
        sa.Column("agent_id", sa.Integer(), nullable=True),
        sa.Column("role", sa.String(length=50), nullable=True),
        sa.Column("joined_at", sa.DateTime(), server_default=sa.text("now()"), nullable=True),
        sa.Column("left_at", sa.DateTime(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.ForeignKeyConstraint(["agent_id"], ["agents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("conversation_id", "agent_id", name="uq_conversation_agent"),
    )
    op.create_index(
        "idx_participant_active",
        "conversation_participants",
        ["conversation_id", "is_active"],
        unique=False,
    )

    # Create coalition_members table
    op.create_table(
        "coalition_members",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("coalition_id", sa.Integer(), nullable=True),
        sa.Column("agent_id", sa.Integer(), nullable=True),
        sa.Column("role", sa.String(length=50), nullable=True),
        sa.Column("contribution", sa.Float(), nullable=True),
        sa.Column("share", sa.Float(), nullable=True),
        sa.Column("joined_at", sa.DateTime(), server_default=sa.text("now()"), nullable=True),
        sa.Column("left_at", sa.DateTime(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.ForeignKeyConstraint(["agent_id"], ["agents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["coalition_id"], ["coalitions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("coalition_id", "agent_id", name="uq_coalition_agent"),
    )
    op.create_index(
        "idx_coalition_member_active",
        "coalition_members",
        ["coalition_id", "is_active"],
        unique=False,
    )

    # Create messages table
    op.create_table(
        "messages",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("conversation_id", sa.Integer(), nullable=True),
        sa.Column("sender_id", sa.Integer(), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("type", sa.String(length=50), nullable=True),
        sa.Column("meta_data", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("edited_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["sender_id"], ["agents.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_messages_id"), "messages", ["id"], unique=False)
    op.create_index(
        "idx_message_conversation_created",
        "messages",
        ["conversation_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    """Drop all tables and types"""
    # Drop tables in reverse order due to foreign key constraints
    op.drop_index("idx_message_conversation_created", table_name="messages")
    op.drop_index(op.f("ix_messages_id"), table_name="messages")
    op.drop_table("messages")

    op.drop_index("idx_coalition_member_active", table_name="coalition_members")
    op.drop_table("coalition_members")

    op.drop_index("idx_participant_active", table_name="conversation_participants")
    op.drop_table("conversation_participants")

    op.drop_index("idx_knowledge_owner_type", table_name="knowledge_graphs")
    op.drop_index(op.f("ix_knowledge_graphs_uuid"), table_name="knowledge_graphs")
    op.drop_index(op.f("ix_knowledge_graphs_id"), table_name="knowledge_graphs")
    op.drop_table("knowledge_graphs")

    op.drop_index("idx_log_timestamp_level", table_name="system_logs")
    op.drop_index("idx_log_component_timestamp", table_name="system_logs")
    op.drop_index(op.f("ix_system_logs_timestamp"), table_name="system_logs")
    op.drop_index(op.f("ix_system_logs_id"), table_name="system_logs")
    op.drop_table("system_logs")

    op.drop_index("idx_coalition_status_type", table_name="coalitions")
    op.drop_index(op.f("ix_coalitions_uuid"), table_name="coalitions")
    op.drop_index(op.f("ix_coalitions_id"), table_name="coalitions")
    op.drop_table("coalitions")

    op.drop_index(op.f("ix_conversations_uuid"), table_name="conversations")
    op.drop_index(op.f("ix_conversations_id"), table_name="conversations")
    op.drop_table("conversations")

    op.drop_index("idx_agent_type_status", table_name="agents")
    op.drop_index("idx_agent_location", table_name="agents")
    op.drop_index(op.f("ix_agents_uuid"), table_name="agents")
    op.drop_index(op.f("ix_agents_id"), table_name="agents")
    op.drop_table("agents")

    # Drop enums
    op.execute("DROP TYPE IF EXISTS conversationtype")
    op.execute("DROP TYPE IF EXISTS agentstatus")
