"""Performance optimization indexes for multi-agent systems.

Revision ID: performance_optimization_20250715
Revises:
Create Date: 2025-07-15 15:45:00.000000

"""

import sqlalchemy as sa

from alembic import op  # type: ignore[attr-defined]

# revision identifiers, used by Alembic.
revision = "performance_optimization_20250715"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    """Create performance optimization indexes."""
    # Enable required PostgreSQL extensions
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute("CREATE EXTENSION IF NOT EXISTS btree_gin")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements")

    # Agent performance indexes
    op.create_index(
        "idx_agents_status_created",
        "agents",
        ["status", "created_at"],
        postgresql_concurrently=True,
    )

    op.create_index(
        "idx_agents_last_active",
        "agents",
        ["last_active"],
        postgresql_concurrently=True,
        postgresql_where=sa.text("last_active IS NOT NULL"),
    )

    op.create_index(
        "idx_agents_template_status",
        "agents",
        ["template", "status"],
        postgresql_concurrently=True,
    )

    # Text search index for agent names
    op.execute(
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_name_trgm
        ON agents USING gin (name gin_trgm_ops)
    """
    )

    op.create_index(
        "idx_agents_inference_count",
        "agents",
        [sa.desc("inference_count")],
        postgresql_concurrently=True,
    )

    op.create_index(
        "idx_agents_total_steps",
        "agents",
        [sa.desc("total_steps")],
        postgresql_concurrently=True,
    )

    # Coalition performance indexes
    op.create_index(
        "idx_coalitions_status_created",
        "coalitions",
        ["status", "created_at"],
        postgresql_concurrently=True,
    )

    op.create_index(
        "idx_coalitions_performance_score",
        "coalitions",
        [sa.desc("performance_score")],
        postgresql_concurrently=True,
    )

    op.create_index(
        "idx_coalitions_cohesion_score",
        "coalitions",
        [sa.desc("cohesion_score")],
        postgresql_concurrently=True,
    )

    # Text search index for coalition names
    op.execute(
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_coalitions_name_trgm
        ON coalitions USING gin (name gin_trgm_ops)
    """
    )

    # Agent-Coalition association indexes
    op.create_index(
        "idx_agent_coalition_agent_id",
        "agent_coalition",
        ["agent_id"],
        postgresql_concurrently=True,
    )

    op.create_index(
        "idx_agent_coalition_coalition_id",
        "agent_coalition",
        ["coalition_id"],
        postgresql_concurrently=True,
    )

    op.create_index(
        "idx_agent_coalition_role",
        "agent_coalition",
        ["role"],
        postgresql_concurrently=True,
    )

    op.create_index(
        "idx_agent_coalition_joined_at",
        "agent_coalition",
        ["joined_at"],
        postgresql_concurrently=True,
    )

    op.create_index(
        "idx_agent_coalition_contribution",
        "agent_coalition",
        [sa.desc("contribution_score")],
        postgresql_concurrently=True,
    )

    op.create_index(
        "idx_agent_coalition_trust",
        "agent_coalition",
        [sa.desc("trust_score")],
        postgresql_concurrently=True,
    )

    # Knowledge graph indexes
    op.create_index(
        "idx_knowledge_nodes_type",
        "db_knowledge_nodes",
        ["type"],
        postgresql_concurrently=True,
    )

    op.create_index(
        "idx_knowledge_nodes_creator",
        "db_knowledge_nodes",
        ["creator_agent_id"],
        postgresql_concurrently=True,
    )

    op.create_index(
        "idx_knowledge_nodes_created",
        "db_knowledge_nodes",
        ["created_at"],
        postgresql_concurrently=True,
    )

    op.create_index(
        "idx_knowledge_nodes_updated",
        "db_knowledge_nodes",
        ["updated_at"],
        postgresql_concurrently=True,
    )

    # Composite indexes for common query patterns
    op.create_index(
        "idx_agents_status_active_updated",
        "agents",
        ["status", "last_active", "updated_at"],
        postgresql_concurrently=True,
        postgresql_where=sa.text("status = 'active'"),
    )

    op.create_index(
        "idx_coalitions_active_performance",
        "coalitions",
        ["status", "performance_score", "created_at"],
        postgresql_concurrently=True,
        postgresql_where=sa.text("status = 'active'"),
    )

    # Partial indexes for common filters
    op.create_index(
        "idx_agents_active_only",
        "agents",
        ["id", "name", "created_at"],
        postgresql_concurrently=True,
        postgresql_where=sa.text("status = 'active'"),
    )

    op.create_index(
        "idx_coalitions_active_only",
        "coalitions",
        ["id", "name", "created_at"],
        postgresql_concurrently=True,
        postgresql_where=sa.text("status = 'active'"),
    )

    # JSON field indexes (PostgreSQL-specific)
    op.execute(
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_beliefs_gin
        ON agents USING gin (beliefs)
    """
    )

    op.execute(
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_preferences_gin
        ON agents USING gin (preferences)
    """
    )

    op.execute(
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_metrics_gin
        ON agents USING gin (metrics)
    """
    )

    op.execute(
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_coalitions_objectives_gin
        ON coalitions USING gin (objectives)
    """
    )

    op.execute(
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_coalitions_capabilities_gin
        ON coalitions USING gin (required_capabilities)
    """
    )

    # Create materialized view for performance monitoring
    op.execute(
        """
        CREATE MATERIALIZED VIEW IF NOT EXISTS agent_performance_summary AS
        SELECT
            a.id,
            a.name,
            a.status,
            a.inference_count,
            a.total_steps,
            a.created_at,
            a.last_active,
            COUNT(ac.coalition_id) as coalition_count,
            AVG(ac.contribution_score) as avg_contribution,
            AVG(ac.trust_score) as avg_trust,
            CASE
                WHEN a.last_active IS NULL THEN 0
                WHEN a.last_active < NOW() - INTERVAL '1 hour' THEN 1
                WHEN a.last_active < NOW() - INTERVAL '10 minutes' THEN 2
                ELSE 3
            END as activity_level
        FROM agents a
        LEFT JOIN agent_coalition ac ON a.id = ac.agent_id
        GROUP BY a.id, a.name, a.status, a.inference_count, a.total_steps, a.created_at, a.last_active
        WITH DATA;
    """
    )

    # Create unique index on materialized view
    op.create_index(
        "idx_agent_performance_summary_id",
        "agent_performance_summary",
        ["id"],
        unique=True,
    )

    op.create_index(
        "idx_agent_performance_summary_activity",
        "agent_performance_summary",
        ["activity_level", "inference_count"],
        postgresql_concurrently=True,
    )

    # Create materialized view for coalition performance
    op.execute(
        """
        CREATE MATERIALIZED VIEW IF NOT EXISTS coalition_performance_summary AS
        SELECT
            c.id,
            c.name,
            c.status,
            c.performance_score,
            c.cohesion_score,
            c.created_at,
            c.dissolved_at,
            COUNT(ac.agent_id) as member_count,
            COUNT(CASE WHEN a.status = 'active' THEN 1 END) as active_members,
            AVG(ac.contribution_score) as avg_member_contribution,
            AVG(ac.trust_score) as avg_member_trust,
            SUM(a.inference_count) as total_member_inferences,
            SUM(a.total_steps) as total_member_steps,
            CASE
                WHEN c.dissolved_at IS NOT NULL THEN 0
                WHEN c.status = 'active' AND COUNT(ac.agent_id) > 0 THEN 3
                WHEN c.status = 'forming' THEN 2
                ELSE 1
            END as effectiveness_level
        FROM coalitions c
        LEFT JOIN agent_coalition ac ON c.id = ac.coalition_id
        LEFT JOIN agents a ON ac.agent_id = a.id
        GROUP BY c.id, c.name, c.status, c.performance_score, c.cohesion_score, c.created_at, c.dissolved_at
        WITH DATA;
    """
    )

    # Create unique index on coalition materialized view
    op.create_index(
        "idx_coalition_performance_summary_id",
        "coalition_performance_summary",
        ["id"],
        unique=True,
    )

    op.create_index(
        "idx_coalition_performance_summary_effectiveness",
        "coalition_performance_summary",
        ["effectiveness_level", "performance_score"],
        postgresql_concurrently=True,
    )

    # Create function to refresh materialized views
    op.execute(
        """
        CREATE OR REPLACE FUNCTION refresh_performance_views()
        RETURNS void AS $$
        BEGIN
            REFRESH MATERIALIZED VIEW CONCURRENTLY agent_performance_summary;
            REFRESH MATERIALIZED VIEW CONCURRENTLY coalition_performance_summary;
        END;
        $$ LANGUAGE plpgsql;
    """
    )

    # Create trigger function to update agent last_active
    op.execute(
        """
        CREATE OR REPLACE FUNCTION update_agent_last_active()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.last_active = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """
    )

    # Create trigger for agent activity updates
    op.execute(
        """
        CREATE TRIGGER trigger_agent_activity_update
        BEFORE UPDATE OF inference_count, total_steps ON agents
        FOR EACH ROW
        EXECUTE FUNCTION update_agent_last_active();
    """
    )


def downgrade():
    """Remove performance optimization indexes."""
    # Drop triggers
    op.execute(
        "DROP TRIGGER IF EXISTS trigger_agent_activity_update ON agents"
    )
    op.execute("DROP FUNCTION IF EXISTS update_agent_last_active()")
    op.execute("DROP FUNCTION IF EXISTS refresh_performance_views()")

    # Drop materialized views
    op.execute(
        "DROP MATERIALIZED VIEW IF EXISTS coalition_performance_summary"
    )
    op.execute("DROP MATERIALIZED VIEW IF EXISTS agent_performance_summary")

    # Drop indexes (in reverse order)
    indexes_to_drop = [
        "idx_coalition_performance_summary_effectiveness",
        "idx_coalition_performance_summary_id",
        "idx_agent_performance_summary_activity",
        "idx_agent_performance_summary_id",
        "idx_coalitions_capabilities_gin",
        "idx_coalitions_objectives_gin",
        "idx_agents_metrics_gin",
        "idx_agents_preferences_gin",
        "idx_agents_beliefs_gin",
        "idx_coalitions_active_only",
        "idx_agents_active_only",
        "idx_coalitions_active_performance",
        "idx_agents_status_active_updated",
        "idx_knowledge_nodes_updated",
        "idx_knowledge_nodes_created",
        "idx_knowledge_nodes_creator",
        "idx_knowledge_nodes_type",
        "idx_agent_coalition_trust",
        "idx_agent_coalition_contribution",
        "idx_agent_coalition_joined_at",
        "idx_agent_coalition_role",
        "idx_agent_coalition_coalition_id",
        "idx_agent_coalition_agent_id",
        "idx_coalitions_name_trgm",
        "idx_coalitions_cohesion_score",
        "idx_coalitions_performance_score",
        "idx_coalitions_status_created",
        "idx_agents_total_steps",
        "idx_agents_inference_count",
        "idx_agents_name_trgm",
        "idx_agents_template_status",
        "idx_agents_last_active",
        "idx_agents_status_created",
    ]

    for index_name in indexes_to_drop:
        op.drop_index(index_name, if_exists=True)

    # Note: We don't drop extensions as they might be used elsewhere
