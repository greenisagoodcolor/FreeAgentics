-- PostgreSQL Production Database Initialization - Secure Configuration
-- This script sets up the database with security hardening for production

-- Create extensions with security considerations
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Configure secure production settings
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET pg_stat_statements.track = 'all';
ALTER SYSTEM SET log_statement = 'mod';
ALTER SYSTEM SET log_min_duration_statement = 500;
ALTER SYSTEM SET log_connections = on;
ALTER SYSTEM SET log_disconnections = on;
ALTER SYSTEM SET log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h ';

-- Security hardening
ALTER SYSTEM SET ssl = on;
ALTER SYSTEM SET password_encryption = 'scram-sha-256';
ALTER SYSTEM SET row_security = on;

-- Connection security
ALTER SYSTEM SET max_connections = 100;
ALTER SYSTEM SET superuser_reserved_connections = 3;

-- Performance and reliability settings
ALTER SYSTEM SET effective_cache_size = '256MB';
ALTER SYSTEM SET shared_buffers = '64MB';
ALTER SYSTEM SET work_mem = '4MB';
ALTER SYSTEM SET maintenance_work_mem = '16MB';

-- WAL and checkpoint settings for production
ALTER SYSTEM SET wal_level = 'replica';
ALTER SYSTEM SET max_wal_size = '1GB';
ALTER SYSTEM SET min_wal_size = '80MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;

-- Create application-specific schema with proper permissions
CREATE SCHEMA IF NOT EXISTS freeagentics_app AUTHORIZATION freeagentics;

-- Revoke public schema permissions for security
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON DATABASE freeagentics FROM PUBLIC;

-- Grant minimal necessary permissions to app user
GRANT CONNECT ON DATABASE freeagentics TO freeagentics;
GRANT USAGE ON SCHEMA freeagentics_app TO freeagentics;
GRANT CREATE ON SCHEMA freeagentics_app TO freeagentics;
GRANT USAGE ON SCHEMA public TO freeagentics;

-- Create monitoring user with minimal privileges
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'freeagentics_monitor') THEN
        CREATE USER freeagentics_monitor WITH PASSWORD 'monitor_change_me_in_production';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE freeagentics TO freeagentics_monitor;
GRANT SELECT ON ALL TABLES IN SCHEMA information_schema TO freeagentics_monitor;
GRANT SELECT ON ALL TABLES IN SCHEMA pg_catalog TO freeagentics_monitor;

-- Create health check function for monitoring
CREATE OR REPLACE FUNCTION health_check()
RETURNS TABLE(
    status text,
    connections integer,
    database_size text,
    uptime interval
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        'healthy'::text as status,
        numbackends as connections,
        pg_size_pretty(pg_database_size(current_database())) as database_size,
        now() - pg_postmaster_start_time() as uptime
    FROM pg_stat_database
    WHERE datname = current_database();
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant execute permission on health check to monitor user
GRANT EXECUTE ON FUNCTION health_check() TO freeagentics_monitor;

-- Create function to safely reset statistics
CREATE OR REPLACE FUNCTION reset_stats()
RETURNS void AS $$
BEGIN
    SELECT pg_stat_reset();
    SELECT pg_stat_reset_shared('bgwriter');
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Only superuser can reset stats
REVOKE EXECUTE ON FUNCTION reset_stats() FROM PUBLIC;

-- Create monitoring schema for performance tracking
CREATE SCHEMA IF NOT EXISTS monitoring;
GRANT USAGE ON SCHEMA monitoring TO freeagentics;
GRANT USAGE ON SCHEMA monitoring TO freeagentics_monitor;

-- Create table for storing query performance metrics
CREATE TABLE IF NOT EXISTS monitoring.slow_queries (
    id SERIAL PRIMARY KEY,
    query_text TEXT,
    execution_time FLOAT,
    calls BIGINT,
    mean_time FLOAT,
    max_time FLOAT,
    total_time FLOAT,
    captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index usage monitoring table
CREATE TABLE IF NOT EXISTS monitoring.index_usage (
    id SERIAL PRIMARY KEY,
    schemaname TEXT,
    tablename TEXT,
    indexname TEXT,
    idx_scan BIGINT,
    idx_tup_read BIGINT,
    idx_tup_fetch BIGINT,
    idx_size TEXT,
    captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create table size monitoring
CREATE TABLE IF NOT EXISTS monitoring.table_sizes (
    id SERIAL PRIMARY KEY,
    schemaname TEXT,
    tablename TEXT,
    row_count BIGINT,
    total_size TEXT,
    table_size TEXT,
    indexes_size TEXT,
    captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Grant permissions on monitoring tables
GRANT SELECT ON ALL TABLES IN SCHEMA monitoring TO freeagentics_monitor;
GRANT ALL ON ALL TABLES IN SCHEMA monitoring TO freeagentics;
GRANT ALL ON ALL SEQUENCES IN SCHEMA monitoring TO freeagentics;

-- Create helper function for coalition performance calculations
CREATE OR REPLACE FUNCTION calculate_coalition_performance(
    agent_count INTEGER,
    objectives_completed INTEGER,
    total_objectives INTEGER,
    cohesion_score FLOAT
) RETURNS FLOAT AS $$
BEGIN
    IF total_objectives = 0 THEN
        RETURN cohesion_score;
    END IF;

    RETURN (
        (objectives_completed::FLOAT / total_objectives) * 0.6 +  -- 60% weight on objectives
        (cohesion_score * 0.3) +                                  -- 30% weight on cohesion
        (LEAST(agent_count / 5.0, 1.0) * 0.1)                   -- 10% weight on size (max at 5 agents)
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Create notification triggers for real-time updates
CREATE OR REPLACE FUNCTION notify_agent_status_change()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM pg_notify(
        'agent_status_changed',
        json_build_object(
            'agent_id', NEW.id,
            'old_status', OLD.status,
            'new_status', NEW.status,
            'timestamp', CURRENT_TIMESTAMP
        )::text
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create notification function for coalition changes
CREATE OR REPLACE FUNCTION notify_coalition_change()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM pg_notify(
        'coalition_changed',
        json_build_object(
            'coalition_id', NEW.id,
            'status', NEW.status,
            'performance_score', NEW.performance_score,
            'timestamp', CURRENT_TIMESTAMP
        )::text
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
