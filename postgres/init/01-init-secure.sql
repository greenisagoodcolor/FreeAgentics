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
