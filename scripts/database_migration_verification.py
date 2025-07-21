#!/usr/bin/env python3
"""
Comprehensive Database Migration Verification Script for FreeAgentics.

This script ensures all database migrations work flawlessly in production environments.
Specifically designed for PostgreSQL 15 with pgvector extension and Active Inference features.

Author: Database Migration Specialist Agent
Critical Mission: Bulletproof migration system for production deployment
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psycopg2
from alembic import command, script
from alembic.config import Config
from pgvector.psycopg2 import register_vector
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.models import Agent, Coalition, KnowledgeNode, User
from database.session import DATABASE_URL, engine
from database.vector_models import (
    VectorEmbedding,
    AgentMemory,
    KnowledgeVector,
    SemanticCluster,
    create_vector_indexes,
    create_vector_functions,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('migration_verification.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DatabaseMigrationVerifier:
    """Comprehensive database migration verification system."""
    
    def __init__(self, database_url: str = None):
        """Initialize the migration verifier.
        
        Args:
            database_url: PostgreSQL connection URL. If None, uses DATABASE_URL from env.
        """
        self.database_url = database_url or DATABASE_URL
        self.test_results = {}
        self.verification_report = {
            "timestamp": datetime.now().isoformat(),
            "database_url_hash": hash(self.database_url) % 10000,  # Partial hash for security
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "critical_failures": [],
            "warnings": [],
            "recommendations": []
        }
        
        if not self.database_url:
            raise ValueError("DATABASE_URL is required for migration verification")
            
        # Validate PostgreSQL connection
        if not self.database_url.startswith(('postgresql://', 'postgres://')):
            raise ValueError("Only PostgreSQL databases are supported in production")
    
    @contextmanager
    def get_connection(self):
        """Get a raw psycopg2 connection for low-level operations."""
        conn = None
        try:
            conn = psycopg2.connect(self.database_url)
            # Register pgvector types
            register_vector(conn)
            yield conn
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    @contextmanager
    def get_sqlalchemy_session(self):
        """Get SQLAlchemy session for ORM operations."""
        engine = create_engine(self.database_url)
        Session = sessionmaker(bind=engine)
        session = Session()
        try:
            yield session
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def log_test_result(self, test_name: str, success: bool, message: str = "", 
                       critical: bool = False):
        """Log a test result and update verification report."""
        self.test_results[test_name] = {
            "success": success,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "critical": critical
        }
        
        self.verification_report["tests_run"] += 1
        if success:
            self.verification_report["tests_passed"] += 1
            logger.info(f"✅ {test_name}: {message}")
        else:
            self.verification_report["tests_failed"] += 1
            if critical:
                self.verification_report["critical_failures"].append({
                    "test": test_name,
                    "message": message
                })
                logger.error(f"🚨 CRITICAL: {test_name}: {message}")
            else:
                self.verification_report["warnings"].append({
                    "test": test_name,
                    "message": message
                })
                logger.warning(f"⚠️ {test_name}: {message}")
    
    def verify_database_connection(self) -> bool:
        """Verify basic database connectivity and PostgreSQL version."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Check PostgreSQL version
                    cur.execute("SELECT version()")
                    version = cur.fetchone()[0]
                    logger.info(f"PostgreSQL version: {version}")
                    
                    # Verify minimum PostgreSQL 12 (recommended 15+)
                    cur.execute("SELECT current_setting('server_version_num')::int")
                    version_num = cur.fetchone()[0]
                    
                    if version_num >= 150000:  # PostgreSQL 15+
                        self.log_test_result(
                            "postgresql_version",
                            True,
                            f"PostgreSQL {version_num // 10000} detected (recommended for production)"
                        )
                    elif version_num >= 120000:  # PostgreSQL 12+
                        self.log_test_result(
                            "postgresql_version",
                            True,
                            f"PostgreSQL {version_num // 10000} detected (minimum supported)",
                            critical=False
                        )
                        self.verification_report["warnings"].append({
                            "test": "postgresql_version",
                            "message": "Consider upgrading to PostgreSQL 15+ for optimal performance"
                        })
                    else:
                        self.log_test_result(
                            "postgresql_version",
                            False,
                            f"PostgreSQL {version_num // 10000} is too old. Minimum version 12 required.",
                            critical=True
                        )
                        return False
                    
                    # Test basic SQL operations
                    cur.execute("SELECT 1 as test_connection")
                    result = cur.fetchone()[0]
                    if result == 1:
                        self.log_test_result(
                            "database_connectivity",
                            True,
                            "Database connection successful"
                        )
                        return True
                    
        except Exception as e:
            self.log_test_result(
                "database_connectivity",
                False,
                f"Connection failed: {str(e)}",
                critical=True
            )
            return False
        
        return False
    
    def verify_required_extensions(self) -> bool:
        """Verify that required PostgreSQL extensions are installed."""
        required_extensions = {
            'uuid-ossp': "UUID generation functions",
            'vector': "pgvector extension for embedding storage",
            'pg_stat_statements': "Query performance monitoring"
        }
        
        optional_extensions = {
            'h3': "H3 hierarchical indexing (recommended for spatial features)",
            'pg_trgm': "Trigram text search (recommended for full-text search)"
        }
        
        all_extensions_ok = True
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Check installed extensions
                    cur.execute("""
                        SELECT extname, extversion 
                        FROM pg_extension 
                        ORDER BY extname
                    """)
                    installed_extensions = {row[0]: row[1] for row in cur.fetchall()}
                    
                    logger.info(f"Installed extensions: {list(installed_extensions.keys())}")
                    
                    # Verify required extensions
                    for ext_name, description in required_extensions.items():
                        if ext_name in installed_extensions:
                            version = installed_extensions[ext_name]
                            self.log_test_result(
                                f"extension_{ext_name}",
                                True,
                                f"{description} v{version} is installed"
                            )
                        else:
                            self.log_test_result(
                                f"extension_{ext_name}",
                                False,
                                f"Required extension {ext_name} ({description}) is missing",
                                critical=True
                            )
                            all_extensions_ok = False
                    
                    # Check optional extensions
                    for ext_name, description in optional_extensions.items():
                        if ext_name in installed_extensions:
                            version = installed_extensions[ext_name]
                            self.log_test_result(
                                f"extension_{ext_name}_optional",
                                True,
                                f"Optional {description} v{version} is installed"
                            )
                        else:
                            self.verification_report["recommendations"].append(
                                f"Consider installing {ext_name} extension: {description}"
                            )
                    
                    # Special check for pgvector functionality
                    if 'vector' in installed_extensions:
                        try:
                            cur.execute("SELECT '[1,2,3]'::vector(3)")
                            self.log_test_result(
                                "pgvector_functionality",
                                True,
                                "pgvector extension is functional"
                            )
                        except Exception as e:
                            self.log_test_result(
                                "pgvector_functionality",
                                False,
                                f"pgvector extension installed but not functional: {e}",
                                critical=True
                            )
                            all_extensions_ok = False
                    
        except Exception as e:
            self.log_test_result(
                "extension_verification",
                False,
                f"Failed to verify extensions: {str(e)}",
                critical=True
            )
            return False
        
        return all_extensions_ok
    
    def verify_alembic_configuration(self) -> bool:
        """Verify Alembic migration configuration is correct."""
        try:
            alembic_cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
            
            # Verify script location
            script_location = alembic_cfg.get_main_option("script_location")
            if not Path(script_location).exists():
                self.log_test_result(
                    "alembic_script_location",
                    False,
                    f"Alembic script location {script_location} does not exist",
                    critical=True
                )
                return False
            
            # Verify migration scripts directory
            script_dir = script.ScriptDirectory.from_config(alembic_cfg)
            versions_dir = Path(script_dir.versions)
            
            if not versions_dir.exists():
                self.log_test_result(
                    "alembic_versions_directory",
                    False,
                    f"Alembic versions directory {versions_dir} does not exist",
                    critical=True
                )
                return False
            
            # Count migration files
            migration_files = list(versions_dir.glob("*.py"))
            migration_count = len([f for f in migration_files if not f.name.startswith("__")])
            
            self.log_test_result(
                "alembic_configuration",
                True,
                f"Alembic configuration valid, {migration_count} migration files found"
            )
            
            # Verify migration chain integrity
            try:
                revisions = script_dir.walk_revisions()
                revision_list = list(revisions)
                self.log_test_result(
                    "migration_chain_integrity",
                    True,
                    f"Migration chain is valid with {len(revision_list)} revisions"
                )
            except Exception as e:
                self.log_test_result(
                    "migration_chain_integrity",
                    False,
                    f"Migration chain has issues: {str(e)}",
                    critical=True
                )
                return False
            
            return True
            
        except Exception as e:
            self.log_test_result(
                "alembic_configuration",
                False,
                f"Alembic configuration error: {str(e)}",
                critical=True
            )
            return False
    
    def run_migrations_test(self) -> bool:
        """Run migrations in a test environment to verify they execute correctly."""
        try:
            # Create a temporary test database
            test_db_name = f"freeagentics_migration_test_{int(time.time())}"
            
            # Parse the original database URL to create test database
            from urllib.parse import urlparse
            parsed = urlparse(self.database_url)
            
            # Connect to postgres database to create test database
            admin_url = f"postgresql://{parsed.username}:{parsed.password}@{parsed.hostname}:{parsed.port}/postgres"
            
            with psycopg2.connect(admin_url) as admin_conn:
                admin_conn.autocommit = True
                with admin_conn.cursor() as cur:
                    cur.execute(f"CREATE DATABASE {test_db_name}")
            
            # Construct test database URL
            test_db_url = f"postgresql://{parsed.username}:{parsed.password}@{parsed.hostname}:{parsed.port}/{test_db_name}"
            
            try:
                # Run migrations on test database
                alembic_cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
                alembic_cfg.set_main_option("sqlalchemy.url", test_db_url)
                
                # Run upgrade to head
                command.upgrade(alembic_cfg, "head")
                
                self.log_test_result(
                    "migration_execution",
                    True,
                    f"All migrations executed successfully on test database {test_db_name}"
                )
                
                # Verify tables were created correctly
                with psycopg2.connect(test_db_url) as test_conn:
                    with test_conn.cursor() as cur:
                        # Check for core tables
                        expected_tables = [
                            'agents', 'coalitions', 'agent_coalition',
                            'db_knowledge_nodes', 'db_knowledge_edges',
                            'conversations', 'prompts', 'knowledge_graph_updates',
                            'prompt_templates', 'users', 'mfa_settings', 'mfa_audit_log'
                        ]
                        
                        cur.execute("""
                            SELECT table_name 
                            FROM information_schema.tables 
                            WHERE table_schema = 'public'
                            AND table_type = 'BASE TABLE'
                        """)
                        existing_tables = [row[0] for row in cur.fetchall()]
                        
                        missing_tables = set(expected_tables) - set(existing_tables)
                        if missing_tables:
                            self.log_test_result(
                                "migration_table_creation",
                                False,
                                f"Missing tables after migration: {missing_tables}",
                                critical=True
                            )
                            return False
                        else:
                            self.log_test_result(
                                "migration_table_creation",
                                True,
                                f"All expected tables created: {len(expected_tables)} tables verified"
                            )
                
                # Test rollback functionality
                try:
                    command.downgrade(alembic_cfg, "base")
                    self.log_test_result(
                        "migration_rollback",
                        True,
                        "Migration rollback to base executed successfully"
                    )
                    
                    # Upgrade back to head for further tests
                    command.upgrade(alembic_cfg, "head")
                    
                except Exception as e:
                    self.log_test_result(
                        "migration_rollback",
                        False,
                        f"Migration rollback failed: {str(e)}",
                        critical=False  # Rollback issues are concerning but not critical
                    )
                
                return True
                
            finally:
                # Clean up test database
                try:
                    with psycopg2.connect(admin_url) as admin_conn:
                        admin_conn.autocommit = True
                        with admin_conn.cursor() as cur:
                            cur.execute(f"DROP DATABASE IF EXISTS {test_db_name}")
                    logger.info(f"Test database {test_db_name} cleaned up")
                except Exception as e:
                    logger.warning(f"Failed to clean up test database {test_db_name}: {e}")
                    
        except Exception as e:
            self.log_test_result(
                "migration_execution",
                False,
                f"Migration test failed: {str(e)}",
                critical=True
            )
            return False
    
    def verify_vector_operations(self) -> bool:
        """Verify pgvector operations work correctly with the schema."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Test vector operations on a temporary table
                    cur.execute("""
                        CREATE TEMPORARY TABLE test_vectors (
                            id SERIAL PRIMARY KEY,
                            embedding vector(512),
                            content text
                        )
                    """)
                    
                    # Insert test vectors
                    import numpy as np
                    test_embedding = np.random.rand(512).tolist()
                    
                    cur.execute(
                        "INSERT INTO test_vectors (embedding, content) VALUES (%s, %s)",
                        (test_embedding, "test content")
                    )
                    
                    # Test vector similarity search
                    query_embedding = np.random.rand(512).tolist()
                    cur.execute("""
                        SELECT id, content, embedding <=> %s as distance
                        FROM test_vectors
                        ORDER BY embedding <=> %s
                        LIMIT 5
                    """, (query_embedding, query_embedding))
                    
                    results = cur.fetchall()
                    if results:
                        self.log_test_result(
                            "vector_similarity_search",
                            True,
                            f"Vector similarity search functional, returned {len(results)} results"
                        )
                    else:
                        self.log_test_result(
                            "vector_similarity_search",
                            False,
                            "Vector similarity search returned no results",
                            critical=True
                        )
                        return False
                    
                    # Test vector indexing
                    try:
                        cur.execute("""
                            CREATE INDEX test_vector_idx ON test_vectors 
                            USING ivfflat (embedding vector_cosine_ops) 
                            WITH (lists = 1)
                        """)
                        self.log_test_result(
                            "vector_indexing",
                            True,
                            "Vector indexing (ivfflat) functional"
                        )
                    except Exception as e:
                        self.log_test_result(
                            "vector_indexing",
                            False,
                            f"Vector indexing failed: {str(e)}",
                            critical=False
                        )
                    
                    return True
                    
        except Exception as e:
            self.log_test_result(
                "vector_operations",
                False,
                f"Vector operations verification failed: {str(e)}",
                critical=True
            )
            return False
    
    def verify_active_inference_storage(self) -> bool:
        """Verify Active Inference specific storage capabilities."""
        try:
            with self.get_sqlalchemy_session() as session:
                # Test agent creation with Active Inference parameters
                test_agent = Agent(
                    name="test_agent",
                    template="active_inference",
                    gmn_spec="test GMN specification",
                    beliefs={"belief_1": 0.8, "belief_2": 0.3},
                    preferences={"preference_1": 1.0},
                    parameters={"precision": 16.0, "learning_rate": 0.01}
                )
                
                session.add(test_agent)
                session.commit()
                
                # Verify the agent was stored correctly
                retrieved_agent = session.query(Agent).filter_by(name="test_agent").first()
                if not retrieved_agent:
                    self.log_test_result(
                        "active_inference_agent_storage",
                        False,
                        "Failed to store and retrieve Active Inference agent",
                        critical=True
                    )
                    return False
                
                # Verify JSON fields are properly stored
                if (retrieved_agent.beliefs.get("belief_1") != 0.8 or
                    retrieved_agent.parameters.get("precision") != 16.0):
                    self.log_test_result(
                        "active_inference_json_storage",
                        False,
                        "Active Inference parameters not stored correctly in JSON fields",
                        critical=True
                    )
                    return False
                
                self.log_test_result(
                    "active_inference_storage",
                    True,
                    "Active Inference agent storage and retrieval successful"
                )
                
                # Test coalition functionality
                test_coalition = Coalition(
                    name="test_coalition",
                    description="Test coalition for verification",
                    objectives={"primary": "test objective"},
                    required_capabilities=["capability_1", "capability_2"]
                )
                
                session.add(test_coalition)
                session.commit()
                
                # Add agent to coalition
                test_coalition.agents.append(retrieved_agent)
                session.commit()
                
                # Verify many-to-many relationship
                coalition_check = session.query(Coalition).filter_by(name="test_coalition").first()
                if not coalition_check or len(coalition_check.agents) != 1:
                    self.log_test_result(
                        "coalition_agent_relationship",
                        False,
                        "Coalition-Agent relationship not working correctly",
                        critical=True
                    )
                    return False
                
                self.log_test_result(
                    "coalition_functionality",
                    True,
                    "Coalition creation and agent assignment successful"
                )
                
                # Clean up test data
                session.delete(test_coalition)
                session.delete(retrieved_agent)
                session.commit()
                
                return True
                
        except Exception as e:
            self.log_test_result(
                "active_inference_storage",
                False,
                f"Active Inference storage verification failed: {str(e)}",
                critical=True
            )
            return False
    
    def verify_performance_features(self) -> bool:
        """Verify database performance features and indexes."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Check for performance indexes
                    cur.execute("""
                        SELECT indexname, tablename 
                        FROM pg_indexes 
                        WHERE schemaname = 'public'
                        AND indexname LIKE 'idx_%'
                        ORDER BY tablename, indexname
                    """)
                    
                    indexes = cur.fetchall()
                    index_count = len(indexes)
                    
                    if index_count > 0:
                        self.log_test_result(
                            "performance_indexes",
                            True,
                            f"Found {index_count} performance indexes"
                        )
                        
                        # Log indexes for review
                        logger.info("Performance indexes found:")
                        for idx_name, table_name in indexes:
                            logger.info(f"  - {table_name}.{idx_name}")
                    else:
                        self.log_test_result(
                            "performance_indexes",
                            False,
                            "No performance indexes found - this may impact production performance",
                            critical=False
                        )
                        self.verification_report["recommendations"].append(
                            "Consider running performance optimization migrations to add indexes"
                        )
                    
                    # Check pg_stat_statements is working
                    try:
                        cur.execute("SELECT count(*) FROM pg_stat_statements LIMIT 1")
                        self.log_test_result(
                            "query_monitoring",
                            True,
                            "pg_stat_statements extension is functional for query monitoring"
                        )
                    except Exception as e:
                        self.log_test_result(
                            "query_monitoring",
                            False,
                            f"pg_stat_statements not functional: {str(e)}",
                            critical=False
                        )
                    
                    return True
                    
        except Exception as e:
            self.log_test_result(
                "performance_verification",
                False,
                f"Performance features verification failed: {str(e)}",
                critical=False
            )
            return False
    
    def verify_security_features(self) -> bool:
        """Verify database security features are in place."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Check SSL/TLS connection
                    cur.execute("SHOW ssl")
                    ssl_status = cur.fetchone()[0]
                    
                    if ssl_status == 'on':
                        self.log_test_result(
                            "ssl_encryption",
                            True,
                            "SSL/TLS encryption is enabled"
                        )
                    else:
                        self.log_test_result(
                            "ssl_encryption",
                            False,
                            "SSL/TLS encryption is not enabled - security risk in production",
                            critical=False
                        )
                        self.verification_report["recommendations"].append(
                            "Enable SSL/TLS encryption for production database connections"
                        )
                    
                    # Check for MFA tables (security feature)
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_schema = 'public' 
                            AND table_name = 'mfa_settings'
                        )
                    """)
                    
                    mfa_tables_exist = cur.fetchone()[0]
                    if mfa_tables_exist:
                        self.log_test_result(
                            "mfa_security_tables",
                            True,
                            "Multi-factor authentication tables are present"
                        )
                    else:
                        self.log_test_result(
                            "mfa_security_tables",
                            False,
                            "MFA tables missing - authentication security features unavailable",
                            critical=False
                        )
                    
                    # Check connection limits
                    cur.execute("SHOW max_connections")
                    max_connections = int(cur.fetchone()[0])
                    
                    if max_connections >= 100:
                        self.log_test_result(
                            "connection_limits",
                            True,
                            f"Connection limit configured: {max_connections}"
                        )
                    else:
                        self.verification_report["recommendations"].append(
                            f"Consider increasing max_connections from {max_connections} for production load"
                        )
                    
                    return True
                    
        except Exception as e:
            self.log_test_result(
                "security_verification",
                False,
                f"Security features verification failed: {str(e)}",
                critical=False
            )
            return False
    
    def generate_migration_deployment_script(self) -> str:
        """Generate a production-ready migration deployment script."""
        script_content = f'''#!/bin/bash
# FreeAgentics Production Migration Deployment Script
# Generated: {datetime.now().isoformat()}
# This script ensures safe migration deployment in production

set -euo pipefail

echo "🚀 FreeAgentics Production Migration Deployment"
echo "=============================================="
echo ""

# Configuration
BACKUP_DIR="/var/backups/freeagentics"
MIGRATION_LOG="/var/log/freeagentics/migration_$(date +%Y%m%d_%H%M%S).log"
ROLLBACK_SCRIPT="/tmp/freeagentics_rollback_$(date +%Y%m%d_%H%M%S).sql"

# Create directories
mkdir -p "$BACKUP_DIR"
mkdir -p "$(dirname "$MIGRATION_LOG")"

# Function to log with timestamp
log() {{
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MIGRATION_LOG"
}}

# Function to handle errors
handle_error() {{
    log "❌ ERROR: Migration failed on line $1"
    log "🔄 Check the rollback script at: $ROLLBACK_SCRIPT"
    log "📋 Full log available at: $MIGRATION_LOG"
    exit 1
}}

# Set up error handling
trap 'handle_error $LINENO' ERR

log "🔍 Pre-migration checks..."

# Check database connectivity
log "Checking database connectivity..."
python3 -c "
import sys
sys.path.insert(0, '{PROJECT_ROOT}')
from scripts.database_migration_verification import DatabaseMigrationVerifier
verifier = DatabaseMigrationVerifier()
if not verifier.verify_database_connection():
    print('Database connection failed')
    sys.exit(1)
print('Database connection successful')
"

# Check required extensions
log "Verifying required PostgreSQL extensions..."
psql "$DATABASE_URL" -c "SELECT extname FROM pg_extension WHERE extname IN ('uuid-ossp', 'vector', 'pg_stat_statements');" -t | grep -E "(uuid-ossp|vector|pg_stat_statements)" | wc -l | grep -q "3" || {{
    log "❌ Required PostgreSQL extensions missing"
    exit 1
}}

# Create database backup
log "📦 Creating pre-migration backup..."
BACKUP_FILE="$BACKUP_DIR/pre_migration_$(date +%Y%m%d_%H%M%S).sql"
pg_dump "$DATABASE_URL" > "$BACKUP_FILE"
log "✅ Backup created: $BACKUP_FILE"

# Generate rollback script
log "📝 Generating rollback script..."
alembic -c {PROJECT_ROOT}/alembic.ini current > /tmp/current_revision.txt
CURRENT_REVISION=$(cat /tmp/current_revision.txt | grep -o '[a-f0-9]{{12}}' | head -1)
echo "-- Rollback script generated $(date)" > "$ROLLBACK_SCRIPT"
echo "-- To rollback: alembic downgrade $CURRENT_REVISION" >> "$ROLLBACK_SCRIPT"
echo "-- Or restore from backup: psql \$DATABASE_URL < $BACKUP_FILE" >> "$ROLLBACK_SCRIPT"
log "✅ Rollback script ready: $ROLLBACK_SCRIPT"

# Run migrations
log "🔄 Running Alembic migrations..."
cd {PROJECT_ROOT}
alembic upgrade head 2>&1 | tee -a "$MIGRATION_LOG"

# Verify migration success
log "✅ Verifying migration success..."
python3 -c "
import sys
sys.path.insert(0, '{PROJECT_ROOT}')
from scripts.database_migration_verification import DatabaseMigrationVerifier
verifier = DatabaseMigrationVerifier()
success = (
    verifier.verify_database_connection() and
    verifier.verify_required_extensions() and
    verifier.verify_active_inference_storage()
)
if not success:
    print('Post-migration verification failed')
    sys.exit(1)
print('Post-migration verification successful')
"

# Test vector operations if pgvector is available
log "🧪 Testing vector operations..."
python3 -c "
import sys
sys.path.insert(0, '{PROJECT_ROOT}')
from scripts.database_migration_verification import DatabaseMigrationVerifier
verifier = DatabaseMigrationVerifier()
if verifier.verify_vector_operations():
    print('Vector operations verified successfully')
else:
    print('Vector operations verification failed - check pgvector setup')
"

# Final health check
log "🏥 Final health check..."
python3 -c "
import sys
sys.path.insert(0, '{PROJECT_ROOT}')
from database.session import check_database_health
if check_database_health():
    print('Database health check passed')
else:
    print('Database health check failed')
    sys.exit(1)
"

log ""
log "🎉 Migration deployment completed successfully!"
log "📊 Migration log: $MIGRATION_LOG"
log "🔄 Rollback script: $ROLLBACK_SCRIPT"
log "📦 Backup available: $BACKUP_FILE"
log ""
log "Next steps:"
log "1. Monitor application logs for any issues"
log "2. Run performance tests to verify system stability"
log "3. Archive backup after successful verification period"
log ""
'''
        
        return script_content
    
    def run_comprehensive_verification(self) -> Dict:
        """Run all verification tests and generate comprehensive report."""
        logger.info("🔍 Starting comprehensive database migration verification...")
        logger.info(f"Database URL (hash): {hash(self.database_url) % 10000}")
        
        # Run all verification tests
        tests = [
            ("Database Connection", self.verify_database_connection),
            ("Required Extensions", self.verify_required_extensions),
            ("Alembic Configuration", self.verify_alembic_configuration),
            ("Migration Execution", self.run_migrations_test),
            ("Vector Operations", self.verify_vector_operations),
            ("Active Inference Storage", self.verify_active_inference_storage),
            ("Performance Features", self.verify_performance_features),
            ("Security Features", self.verify_security_features),
        ]
        
        overall_success = True
        
        for test_name, test_function in tests:
            logger.info(f"🧪 Running: {test_name}")
            try:
                success = test_function()
                if not success:
                    overall_success = False
            except Exception as e:
                logger.error(f"Test {test_name} threw exception: {e}")
                self.log_test_result(
                    test_name.lower().replace(" ", "_"),
                    False,
                    f"Test threw exception: {str(e)}",
                    critical=True
                )
                overall_success = False
        
        # Generate final assessment
        self.verification_report["overall_success"] = overall_success
        self.verification_report["production_ready"] = (
            overall_success and 
            len(self.verification_report["critical_failures"]) == 0
        )
        
        # Generate recommendations
        if not self.verification_report["production_ready"]:
            self.verification_report["recommendations"].insert(0, 
                "CRITICAL: System is NOT ready for production deployment"
            )
        else:
            self.verification_report["recommendations"].insert(0,
                "✅ System appears ready for production deployment"
            )
        
        # Save verification report
        report_file = PROJECT_ROOT / "migration_verification_report.json"
        with open(report_file, 'w') as f:
            json.dump(self.verification_report, f, indent=2)
        
        logger.info(f"📄 Verification report saved: {report_file}")
        
        # Generate deployment script
        deployment_script = self.generate_migration_deployment_script()
        script_file = PROJECT_ROOT / "scripts" / "production_migration_deploy.sh"
        with open(script_file, 'w') as f:
            f.write(deployment_script)
        os.chmod(script_file, 0o755)
        
        logger.info(f"🚀 Production deployment script generated: {script_file}")
        
        return self.verification_report
    
    def print_summary(self):
        """Print a human-readable summary of the verification results."""
        print("\n" + "="*80)
        print("🔍 DATABASE MIGRATION VERIFICATION SUMMARY")
        print("="*80)
        
        print(f"📊 Tests Run: {self.verification_report['tests_run']}")
        print(f"✅ Tests Passed: {self.verification_report['tests_passed']}")
        print(f"❌ Tests Failed: {self.verification_report['tests_failed']}")
        
        if self.verification_report['critical_failures']:
            print(f"\n🚨 CRITICAL FAILURES ({len(self.verification_report['critical_failures'])}):")
            for failure in self.verification_report['critical_failures']:
                print(f"   - {failure['test']}: {failure['message']}")
        
        if self.verification_report['warnings']:
            print(f"\n⚠️  WARNINGS ({len(self.verification_report['warnings'])}):")
            for warning in self.verification_report['warnings'][:5]:  # Show first 5
                print(f"   - {warning['test']}: {warning['message']}")
        
        if self.verification_report['recommendations']:
            print(f"\n💡 RECOMMENDATIONS ({len(self.verification_report['recommendations'])}):")
            for rec in self.verification_report['recommendations'][:10]:  # Show first 10
                print(f"   - {rec}")
        
        print(f"\n🏁 FINAL ASSESSMENT:")
        if self.verification_report['production_ready']:
            print("   ✅ SYSTEM IS READY FOR PRODUCTION DEPLOYMENT")
        else:
            print("   ❌ SYSTEM IS NOT READY FOR PRODUCTION")
            print("   🔧 Address critical failures before deploying")
        
        print("="*80)


def main():
    """Main function to run database migration verification."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FreeAgentics Database Migration Verification")
    parser.add_argument(
        "--database-url", 
        help="PostgreSQL database URL (overrides DATABASE_URL env var)"
    )
    parser.add_argument(
        "--output", 
        default="migration_verification_report.json",
        help="Output file for verification report"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    try:
        verifier = DatabaseMigrationVerifier(database_url=args.database_url)
        report = verifier.run_comprehensive_verification()
        
        if not args.quiet:
            verifier.print_summary()
        
        # Exit with appropriate code
        if report['production_ready']:
            logger.info("✅ Verification completed successfully")
            sys.exit(0)
        else:
            logger.error("❌ Verification failed - system not ready for production")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"💥 Verification script failed: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()