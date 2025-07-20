"""rename knowledge tables to avoid conflicts.

Revision ID: c42749d3c630
Revises: 1b4306802749
Create Date: 2025-07-04 11:13:37.581314

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op  

# revision identifiers, used by Alembic.
revision: str = "c42749d3c630"
down_revision: Union[str, Sequence[str], None] = "1b4306802749"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Rename knowledge tables to avoid conflicts with knowledge_graph storage
    op.rename_table("knowledge_nodes", "db_knowledge_nodes")
    op.rename_table("knowledge_edges", "db_knowledge_edges")


def downgrade() -> None:
    """Downgrade schema."""
    # Revert table renames
    op.rename_table("db_knowledge_edges", "knowledge_edges")
    op.rename_table("db_knowledge_nodes", "knowledge_nodes")
