-- PostgreSQL Production Database Initialization
-- This script sets up the database for production deployment

-- Create additional databases if needed
-- CREATE DATABASE freeagentics_test;

-- Set up extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Configure production settings
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET pg_stat_statements.track = 'all';
ALTER SYSTEM SET log_statement = 'mod';
ALTER SYSTEM SET log_min_duration_statement = 1000;

-- Create performance monitoring view
CREATE OR REPLACE VIEW database_performance AS
SELECT
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats
WHERE schemaname NOT IN ('information_schema', 'pg_catalog');

-- Grant necessary permissions
GRANT USAGE ON SCHEMA public TO freeagentics;
GRANT CREATE ON SCHEMA public TO freeagentics;

-- Set up connection limits and timeouts
ALTER DATABASE freeagentics SET statement_timeout = '30min';
ALTER DATABASE freeagentics SET idle_in_transaction_session_timeout = '10min';

-- Create monitoring user for health checks
CREATE USER monitoring WITH PASSWORD 'monitoring_password_change_me';
GRANT CONNECT ON DATABASE freeagentics TO monitoring;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO monitoring;

-- Log initialization completion
INSERT INTO pg_stat_statements_info
SELECT 'Database initialized for production' AS note
WHERE EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements');
