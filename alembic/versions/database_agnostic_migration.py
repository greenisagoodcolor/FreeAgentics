"""Database-agnostic migration for cross-platform compatibility.

Revision ID: database_agnostic_001
Revises: performance_indexes_001
Create Date: 2025-01-13 12:00:00.000000

"""

from typing import Optional, Sequence, Union

from alembic import op  

# revision identifiers, used by Alembic.
revision: str = "database_agnostic_001"
down_revision: Optional[Union[str, Sequence[str]]] = "performance_indexes_001"
branch_labels: Optional[Union[str, Sequence[str]]] = None
depends_on: Optional[Union[str, Sequence[str]]] = None


def upgrade() -> None:
    """Add database-agnostic indexes and constraints."""
    # Add performance indexes using database-agnostic syntax
    try:
        # Agents table indexes
        op.create_index("idx_agents_status_active", "agents", ["status"])
        op.create_index("idx_agents_template_type", "agents", ["template"])
        op.create_index(
            "idx_agents_last_active_desc", "agents", ["last_active"]
        )

        # Coalitions table indexes
        op.create_index(
            "idx_coalitions_status_active", "coalitions", ["status"]
        )
        op.create_index("idx_coalitions_type_category", "coalitions", ["type"])

        # Agent-Coalition association indexes
        op.create_index(
            "idx_agent_coalition_agent_lookup", "agent_coalition", ["agent_id"]
        )
        op.create_index(
            "idx_agent_coalition_coalition_lookup",
            "agent_coalition",
            ["coalition_id"],
        )

        # Conversations table indexes (if exists)
        try:
            op.create_index(
                "idx_conversations_agent_id", "conversations", ["agent_id"]
            )
            op.create_index(
                "idx_conversations_session_id", "conversations", ["session_id"]
            )
            op.create_index(
                "idx_conversations_created_at", "conversations", ["created_at"]
            )
        except Exception:
            # Table might not exist in all environments
            pass

        # GMN specifications indexes (if exists)
        try:
            op.create_index(
                "idx_gmn_specifications_agent_id",
                "gmn_specifications",
                ["agent_id"],
            )
            op.create_index(
                "idx_gmn_specifications_is_active",
                "gmn_specifications",
                ["is_active"],
            )
        except Exception:
            # Table might not exist in all environments
            pass

    except Exception as e:
        # Log the error but don't fail the migration
        print(f"Warning: Some indexes could not be created: {e}")


def downgrade() -> None:
    """Remove database-agnostic indexes."""
    try:
        # Remove agent indexes
        op.drop_index("idx_agents_status_active", table_name="agents")
        op.drop_index("idx_agents_template_type", table_name="agents")
        op.drop_index("idx_agents_last_active_desc", table_name="agents")

        # Remove coalition indexes
        op.drop_index("idx_coalitions_status_active", table_name="coalitions")
        op.drop_index("idx_coalitions_type_category", table_name="coalitions")

        # Remove association indexes
        op.drop_index(
            "idx_agent_coalition_agent_lookup", table_name="agent_coalition"
        )
        op.drop_index(
            "idx_agent_coalition_coalition_lookup",
            table_name="agent_coalition",
        )

        # Remove conversation indexes (if exists)
        try:
            op.drop_index(
                "idx_conversations_agent_id", table_name="conversations"
            )
            op.drop_index(
                "idx_conversations_session_id", table_name="conversations"
            )
            op.drop_index(
                "idx_conversations_created_at", table_name="conversations"
            )
        except Exception:
            pass

        # Remove GMN indexes (if exists)
        try:
            op.drop_index(
                "idx_gmn_specifications_agent_id",
                table_name="gmn_specifications",
            )
            op.drop_index(
                "idx_gmn_specifications_is_active",
                table_name="gmn_specifications",
            )
        except Exception:
            pass

    except Exception as e:
        # Log the error but don't fail the migration
        print(f"Warning: Some indexes could not be removed: {e}")
