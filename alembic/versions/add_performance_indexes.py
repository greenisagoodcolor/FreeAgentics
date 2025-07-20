"""Add performance indexes for common queries.

Revision ID: performance_indexes_001
Revises: 2a8b9c3d4e5f
Create Date: 2025-01-13 12:00:00.000000

"""

from typing import Optional, Sequence, Union

import sqlalchemy as sa  

from alembic import op  

# revision identifiers, used by Alembic.
revision: str = "performance_indexes_001"
down_revision: Optional[str] = "2a8b9c3d4e5f"
branch_labels: Optional[Union[str, Sequence[str]]] = None
depends_on: Optional[Union[str, Sequence[str]]] = None


def upgrade() -> None:
    """Add performance indexes for common queries."""
    op.create_index(
        "idx_agents_status_template", "agents", ["status", "template"]
    )
    op.create_index("idx_agents_last_active", "agents", ["last_active"])
    op.create_index(
        "idx_agents_inference_count", "agents", ["inference_count"]
    )

    # Coalitions table indexes
    op.create_index("idx_coalitions_status", "coalitions", ["status"])
    op.create_index("idx_coalitions_created_at", "coalitions", ["created_at"])

    # Agent-Coalition association indexes
    op.create_index("idx_agent_coalition_role", "agent_coalition", ["role"])
    op.create_index(
        "idx_agent_coalition_joined_at", "agent_coalition", ["joined_at"]
    )

    # Knowledge graph indexes (if they exist)
    try:
        op.create_index(
            "idx_knowledge_nodes_type", "knowledge_nodes", ["type"]
        )
        op.create_index(
            "idx_knowledge_edges_type", "knowledge_edges", ["type"]
        )
    except Exception:
        # Tables might not exist yet, skip silently
        pass


def downgrade() -> None:
    """Remove performance indexes."""
    op.drop_index("idx_agents_status_template", table_name="agents")
    op.drop_index("idx_agents_last_active", table_name="agents")
    op.drop_index("idx_agents_inference_count", table_name="agents")

    # Remove coalition indexes
    op.drop_index("idx_coalitions_status", table_name="coalitions")
    op.drop_index("idx_coalitions_created_at", table_name="coalitions")

    # Remove association indexes
    op.drop_index("idx_agent_coalition_role", table_name="agent_coalition")
    op.drop_index(
        "idx_agent_coalition_joined_at", table_name="agent_coalition"
    )

    # Remove knowledge graph indexes (if they exist)
    try:
        op.drop_index("idx_knowledge_nodes_type", table_name="knowledge_nodes")
        op.drop_index("idx_knowledge_edges_type", table_name="knowledge_edges")
    except Exception:
        # Tables might not exist, skip silently
        pass
