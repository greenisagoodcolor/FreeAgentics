"""Add prompt tracking tables

Revision ID: add_prompt_tracking
Revises: performance_optimization_indexes
Create Date: 2025-01-17 10:00:00.000000

"""
import uuid

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = 'add_prompt_tracking'
down_revision = 'performance_optimization_indexes'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add tables for prompt processing and conversation tracking."""

    # Create conversations table
    op.create_table(
        'conversations',
        sa.Column(
            'id',
            postgresql.UUID(as_uuid=True),
            nullable=False,
            default=uuid.uuid4,
        ),
        sa.Column('user_id', sa.String(100), nullable=False),
        sa.Column('title', sa.String(200), nullable=True),
        sa.Column(
            'status',
            sa.Enum(
                'active',
                'completed',
                'archived',
                'error',
                name='conversationstatus',
            ),
            nullable=False,
        ),
        sa.Column('context', sa.JSON(), nullable=False, server_default='{}'),
        sa.Column('agent_ids', sa.JSON(), nullable=False, server_default='[]'),
        sa.Column(
            'created_at',
            sa.DateTime(),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )

    # Create indexes for conversations
    op.create_index('idx_conversations_user_id', 'conversations', ['user_id'])
    op.create_index('idx_conversations_status', 'conversations', ['status'])
    op.create_index(
        'idx_conversations_created_at', 'conversations', ['created_at']
    )

    # Create prompts table
    op.create_table(
        'prompts',
        sa.Column(
            'id',
            postgresql.UUID(as_uuid=True),
            nullable=False,
            default=uuid.uuid4,
        ),
        sa.Column(
            'conversation_id', postgresql.UUID(as_uuid=True), nullable=False
        ),
        sa.Column('prompt_text', sa.Text(), nullable=False),
        sa.Column(
            'iteration_count', sa.Integer(), nullable=False, server_default='1'
        ),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('gmn_specification', sa.Text(), nullable=True),
        sa.Column(
            'status',
            sa.Enum(
                'pending',
                'processing',
                'success',
                'partial_success',
                'failed',
                name='promptstatus',
            ),
            nullable=False,
        ),
        sa.Column(
            'response_data', sa.JSON(), nullable=False, server_default='{}'
        ),
        sa.Column(
            'next_suggestions', sa.JSON(), nullable=False, server_default='[]'
        ),
        sa.Column('warnings', sa.JSON(), nullable=False, server_default='[]'),
        sa.Column('errors', sa.JSON(), nullable=False, server_default='[]'),
        sa.Column('processing_time_ms', sa.Float(), nullable=True),
        sa.Column('tokens_used', sa.Integer(), nullable=True),
        sa.Column(
            'created_at',
            sa.DateTime(),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column('processed_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ['conversation_id'],
            ['conversations.id'],
        ),
        sa.ForeignKeyConstraint(
            ['agent_id'],
            ['agents.id'],
        ),
        sa.PrimaryKeyConstraint('id'),
    )

    # Create indexes for prompts
    op.create_index(
        'idx_prompts_conversation_id', 'prompts', ['conversation_id']
    )
    op.create_index('idx_prompts_agent_id', 'prompts', ['agent_id'])
    op.create_index('idx_prompts_status', 'prompts', ['status'])
    op.create_index('idx_prompts_created_at', 'prompts', ['created_at'])

    # Create knowledge_graph_updates table
    op.create_table(
        'knowledge_graph_updates',
        sa.Column(
            'id',
            postgresql.UUID(as_uuid=True),
            nullable=False,
            default=uuid.uuid4,
        ),
        sa.Column('prompt_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('node_id', sa.String(100), nullable=False),
        sa.Column('node_type', sa.String(50), nullable=False),
        sa.Column('operation', sa.String(20), nullable=False),
        sa.Column(
            'properties', sa.JSON(), nullable=False, server_default='{}'
        ),
        sa.Column(
            'applied', sa.Boolean(), nullable=False, server_default='false'
        ),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column(
            'created_at',
            sa.DateTime(),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ['prompt_id'],
            ['prompts.id'],
        ),
        sa.PrimaryKeyConstraint('id'),
    )

    # Create indexes for knowledge_graph_updates
    op.create_index(
        'idx_kg_updates_prompt_id', 'knowledge_graph_updates', ['prompt_id']
    )
    op.create_index(
        'idx_kg_updates_node_type', 'knowledge_graph_updates', ['node_type']
    )
    op.create_index(
        'idx_kg_updates_created_at', 'knowledge_graph_updates', ['created_at']
    )

    # Create prompt_templates table
    op.create_table(
        'prompt_templates',
        sa.Column(
            'id',
            postgresql.UUID(as_uuid=True),
            nullable=False,
            default=uuid.uuid4,
        ),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('category', sa.String(50), nullable=False),
        sa.Column('template_text', sa.Text(), nullable=False),
        sa.Column(
            'default_parameters',
            sa.JSON(),
            nullable=False,
            server_default='{}',
        ),
        sa.Column(
            'example_prompts', sa.JSON(), nullable=False, server_default='[]'
        ),
        sa.Column('suggested_gmn_structure', sa.Text(), nullable=True),
        sa.Column(
            'constraints', sa.JSON(), nullable=False, server_default='{}'
        ),
        sa.Column(
            'usage_count', sa.Integer(), nullable=False, server_default='0'
        ),
        sa.Column('success_rate', sa.Float(), nullable=True),
        sa.Column(
            'is_active', sa.Boolean(), nullable=False, server_default='true'
        ),
        sa.Column(
            'created_at',
            sa.DateTime(),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name', name='uq_prompt_templates_name'),
    )

    # Create indexes for prompt_templates
    op.create_index(
        'idx_prompt_templates_category', 'prompt_templates', ['category']
    )
    op.create_index(
        'idx_prompt_templates_is_active', 'prompt_templates', ['is_active']
    )


def downgrade() -> None:
    """Remove prompt tracking tables."""
    # Drop tables in reverse order due to foreign key constraints
    op.drop_table('prompt_templates')
    op.drop_table('knowledge_graph_updates')
    op.drop_table('prompts')
    op.drop_table('conversations')

    # Drop enums
    op.execute('DROP TYPE IF EXISTS promptstatus')
    op.execute('DROP TYPE IF EXISTS conversationstatus')
