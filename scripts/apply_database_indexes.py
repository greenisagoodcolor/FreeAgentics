#!/usr/bin/env python3
"""
Apply Database Indexes Script
Creates optimized indexes for FreeAgentics database
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError


def create_indexes(engine):
    """Create all recommended indexes for optimal performance."""

    indexes = [
        # Agent table indexes
        {
            "name": "idx_agents_status",
            "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_status ON agents (status)",
            "description": "Speed up agent status queries",
        },
        {
            "name": "idx_agents_template",
            "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_template ON agents (template)",
            "description": "Speed up agent template filtering",
        },
        {
            "name": "idx_agents_last_active",
            "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_last_active ON agents (last_active DESC)",
            "description": "Speed up recent agent queries",
        },
        {
            "name": "idx_agents_created_at",
            "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_created_at ON agents (created_at DESC)",
            "description": "Speed up agent listing by creation date",
        },
        {
            "name": "idx_agents_status_template",
            "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_status_template ON agents (status, template)",
            "description": "Composite index for common filter combinations",
        },
        # Coalition table indexes
        {
            "name": "idx_coalitions_status",
            "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_coalitions_status ON coalitions (status)",
            "description": "Speed up coalition status queries",
        },
        {
            "name": "idx_coalitions_performance_score",
            "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_coalitions_performance_score ON coalitions (performance_score DESC)",
            "description": "Speed up coalition ranking queries",
        },
        {
            "name": "idx_coalitions_created_at",
            "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_coalitions_created_at ON coalitions (created_at DESC)",
            "description": "Speed up coalition listing by date",
        },
        {
            "name": "idx_coalitions_status_performance",
            "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_coalitions_status_performance ON coalitions (status, performance_score DESC)",
            "description": "Composite index for active coalition rankings",
        },
        # Agent-Coalition association table indexes
        {
            "name": "idx_agent_coalition_agent_id",
            "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_coalition_agent_id ON agent_coalition (agent_id)",
            "description": "Speed up agent's coalitions lookup",
        },
        {
            "name": "idx_agent_coalition_coalition_id",
            "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_coalition_coalition_id ON agent_coalition (coalition_id)",
            "description": "Speed up coalition's agents lookup",
        },
        {
            "name": "idx_agent_coalition_joined_at",
            "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_coalition_joined_at ON agent_coalition (joined_at DESC)",
            "description": "Speed up recent coalition joins",
        },
        {
            "name": "idx_agent_coalition_role",
            "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_coalition_role ON agent_coalition (role)",
            "description": "Speed up role-based queries",
        },
        # Knowledge graph indexes
        {
            "name": "idx_knowledge_nodes_type",
            "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_nodes_type ON db_knowledge_nodes (type)",
            "description": "Speed up node type filtering",
        },
        {
            "name": "idx_knowledge_nodes_creator_agent",
            "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_nodes_creator_agent ON db_knowledge_nodes (creator_agent_id)",
            "description": "Speed up agent's created nodes lookup",
        },
        {
            "name": "idx_knowledge_nodes_is_current",
            "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_nodes_is_current ON db_knowledge_nodes (is_current) WHERE is_current = true",
            "description": "Partial index for current nodes only",
        },
        {
            "name": "idx_knowledge_nodes_created_at",
            "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_nodes_created_at ON db_knowledge_nodes (created_at DESC)",
            "description": "Speed up recent knowledge queries",
        },
        {
            "name": "idx_knowledge_nodes_type_current",
            "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_nodes_type_current ON db_knowledge_nodes (type, is_current) WHERE is_current = true",
            "description": "Composite index for current nodes by type",
        },
        # Knowledge edge indexes
        {
            "name": "idx_knowledge_edges_source",
            "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_edges_source ON db_knowledge_edges (source_id)",
            "description": "Speed up outgoing edge lookups",
        },
        {
            "name": "idx_knowledge_edges_target",
            "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_edges_target ON db_knowledge_edges (target_id)",
            "description": "Speed up incoming edge lookups",
        },
        {
            "name": "idx_knowledge_edges_type",
            "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_edges_type ON db_knowledge_edges (type)",
            "description": "Speed up edge type filtering",
        },
        {
            "name": "idx_knowledge_edges_source_type",
            "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_edges_source_type ON db_knowledge_edges (source_id, type)",
            "description": "Composite index for typed edge traversal",
        },
        # GIN indexes for JSON columns (PostgreSQL-specific)
        {
            "name": "idx_agents_parameters_gin",
            "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_parameters_gin ON agents USING gin (parameters)",
            "description": "Speed up JSON parameter searches",
        },
        {
            "name": "idx_agents_metrics_gin",
            "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_metrics_gin ON agents USING gin (metrics)",
            "description": "Speed up JSON metrics searches",
        },
        {
            "name": "idx_coalitions_objectives_gin",
            "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_coalitions_objectives_gin ON coalitions USING gin (objectives)",
            "description": "Speed up JSON objectives searches",
        },
        {
            "name": "idx_knowledge_nodes_properties_gin",
            "sql": "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_nodes_properties_gin ON db_knowledge_nodes USING gin (properties)",
            "description": "Speed up JSON properties searches",
        },
    ]

    created_count = 0
    skipped_count = 0
    error_count = 0

    print("Creating database indexes...")
    print("=" * 50)

    with engine.connect() as conn:
        for index in indexes:
            try:
                # Check if index already exists
                check_query = text(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM pg_indexes
                        WHERE indexname = :index_name
                    )
                """
                )

                exists = conn.execute(
                    check_query, {"index_name": index["name"]}
                ).scalar()

                if exists:
                    print(f"⏭️  {index['name']} - Already exists")
                    skipped_count += 1
                else:
                    print(f"🔨 Creating {index['name']}...")
                    print(f"   Purpose: {index['description']}")
                    conn.execute(text(index["sql"]))
                    conn.commit()
                    print(f"✓  {index['name']} - Created successfully")
                    created_count += 1

            except ProgrammingError as e:
                if "already exists" in str(e):
                    print(f"⏭️  {index['name']} - Already exists")
                    skipped_count += 1
                else:
                    print(f"✗  {index['name']} - Error: {e}")
                    error_count += 1
            except Exception as e:
                print(f"✗  {index['name']} - Error: {e}")
                error_count += 1

    print("\n" + "=" * 50)
    print("Index Creation Summary:")
    print(f"  Created: {created_count}")
    print(f"  Skipped (already exist): {skipped_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total indexes: {len(indexes)}")

    return created_count, skipped_count, error_count


def analyze_tables(engine):
    """Run ANALYZE on all tables to update statistics."""
    print("\nUpdating table statistics...")

    tables = [
        "agents",
        "coalitions",
        "agent_coalition",
        "db_knowledge_nodes",
        "db_knowledge_edges",
    ]

    with engine.connect() as conn:
        for table in tables:
            try:
                print(f"  Analyzing {table}...")
                conn.execute(text(f"ANALYZE {table}"))
                conn.commit()
                print(f"  ✓ {table} analyzed")
            except Exception as e:
                print(f"  ✗ {table} - Error: {e}")

    print("\n✓ Table statistics updated")


def check_index_usage(engine):
    """Check current index usage statistics."""
    print("\nChecking index usage statistics...")

    query = text(
        """
        SELECT
            schemaname,
            tablename,
            indexname,
            idx_scan as scans,
            idx_tup_read as tuples_read,
            idx_tup_fetch as tuples_fetched,
            pg_size_pretty(pg_relation_size(indexrelid)) as size
        FROM pg_stat_user_indexes
        WHERE schemaname = 'public'
        ORDER BY idx_scan DESC
        LIMIT 20
    """
    )

    with engine.connect() as conn:
        result = conn.execute(query)
        indexes = result.fetchall()

        if indexes:
            print("\nTop 20 Most Used Indexes:")
            print("-" * 80)
            print(f"{'Index Name':<40} {'Scans':>10} {'Tuples Read':>15} {'Size':>10}")
            print("-" * 80)

            for idx in indexes:
                print(
                    f"{idx.indexname:<40} {idx.scans:>10,} {idx.tuples_read:>15,} {idx.size:>10}"
                )
        else:
            print("No index usage statistics available yet.")


def main():
    """Main function."""
    # Load environment variables
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env.production"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"Loaded environment from: {env_file}")
    else:
        load_dotenv()
        print("Using default .env file")

    # Get database URL
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("✗ DATABASE_URL not found in environment")
        sys.exit(1)

    print(
        f"Database URL: {database_url.replace(database_url.split('@')[0].split('//')[1].split(':')[1], '***')}"
    )

    # Create engine
    engine = create_engine(database_url)

    try:
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        # Create indexes
        created, skipped, errors = create_indexes(engine)

        # Analyze tables
        analyze_tables(engine)

        # Check usage (if any data exists)
        check_index_usage(engine)

        if errors > 0:
            print("\n⚠️  Some indexes failed to create. Check the errors above.")
            sys.exit(1)
        else:
            print("\n✓ All indexes are properly configured!")
            sys.exit(0)

    except Exception as e:
        print(f"\n✗ Database operation failed: {e}")
        sys.exit(1)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
