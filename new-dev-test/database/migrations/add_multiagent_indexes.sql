-- Multi-Agent System Performance Optimization Indexes
-- Generated for Task 20.4: Database Query Optimization
--
-- This migration adds comprehensive indexes optimized for multi-agent coordination scenarios
-- including high-concurrency operations, time-series queries, and coalition management

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For trigram text search
CREATE EXTENSION IF NOT EXISTS btree_gin;  -- For GIN indexes on btree-indexable types
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;  -- For query performance monitoring

-- ========================================
-- AGENT TABLE INDEXES
-- ========================================

-- Composite index for active agent queries with status filtering
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_active_status_perf
ON agents (status, last_active DESC NULLS LAST, inference_count DESC)
WHERE status = 'active';

-- Index for agent template queries with status
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_template_status_active
ON agents (template, status, created_at DESC)
WHERE status IN ('active', 'pending');

-- Covering index for agent lookups (reduces table access)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_lookup_covering
ON agents (id)
INCLUDE (name, status, template, last_active, inference_count);

-- Trigram index for agent name search
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_name_trgm
ON agents USING gin (name gin_trgm_ops);

-- Time-series index for agent activity monitoring
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_time_series
ON agents (last_active DESC NULLS LAST)
WHERE last_active IS NOT NULL;

-- Index for recent active agents (partial index)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_recent_active
ON agents (last_active DESC, inference_count DESC)
WHERE last_active > NOW() - INTERVAL '24 hours' AND status = 'active';

-- JSON field indexes for belief and preference queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_beliefs_state
ON agents USING gin ((beliefs -> 'state'));

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_preferences_goals
ON agents USING gin ((preferences -> 'goals'));

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_metrics_performance
ON agents USING gin ((metrics -> 'performance'));

-- ========================================
-- COALITION TABLE INDEXES
-- ========================================

-- Composite index for active coalition queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_coalitions_active_performance
ON coalitions (status, performance_score DESC, cohesion_score DESC)
WHERE status = 'active';

-- Index for coalition formation queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_coalitions_forming_created
ON coalitions (status, created_at DESC)
WHERE status = 'forming';

-- Covering index for coalition lookups
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_coalitions_lookup_covering
ON coalitions (id)
INCLUDE (name, status, performance_score, created_at);

-- Partial index for high-performing coalitions
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_coalitions_high_performing
ON coalitions (performance_score DESC, cohesion_score DESC)
WHERE performance_score > 0.7 AND status = 'active';

-- Date range index for coalition lifecycle queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_coalitions_lifecycle
ON coalitions (created_at, dissolved_at)
WHERE dissolved_at IS NULL;

-- JSON indexes for coalition objectives and capabilities
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_coalitions_objectives
ON coalitions USING gin (objectives);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_coalitions_capabilities
ON coalitions USING gin (required_capabilities);

-- ========================================
-- AGENT_COALITION ASSOCIATION INDEXES
-- ========================================

-- Composite index for agent's coalitions lookup
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_coalition_agent_lookup
ON agent_coalition (agent_id, coalition_id, joined_at DESC);

-- Composite index for coalition's members lookup
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_coalition_coalition_lookup
ON agent_coalition (coalition_id, agent_id, role, contribution_score DESC);

-- Index for role-based queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_coalition_role_lookup
ON agent_coalition (role, coalition_id)
WHERE role IN ('leader', 'coordinator');

-- Performance-based member ranking index
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_coalition_performance
ON agent_coalition (coalition_id, contribution_score DESC, trust_score DESC)
WHERE contribution_score > 0;

-- Time-based coalition membership queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_coalition_temporal
ON agent_coalition (joined_at DESC, coalition_id, agent_id);

-- ========================================
-- KNOWLEDGE GRAPH INDEXES
-- ========================================

-- Composite index for knowledge node queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_nodes_type_creator
ON knowledge_nodes (type, creator_agent_id, created_at DESC);

-- Current version index for knowledge nodes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_nodes_current_version
ON knowledge_nodes (type, is_current, version DESC)
WHERE is_current = true;

-- Confidence-based knowledge queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_nodes_confidence
ON knowledge_nodes (confidence DESC, type)
WHERE confidence > 0.5;

-- Knowledge edge traversal indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_edges_source_type
ON knowledge_edges (source_id, type, confidence DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_edges_target_type
ON knowledge_edges (target_id, type, confidence DESC);

-- ========================================
-- PERFORMANCE METRICS INDEXES
-- ========================================

-- Time-series index for metrics queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_performance_metrics_time_type
ON performance_metrics (timestamp DESC, metric_type, test_run_id);

-- Metric type and value queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_performance_metrics_type_value
ON performance_metrics (metric_type, value DESC, timestamp DESC);

-- Test run performance analysis
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_performance_metrics_test_analysis
ON performance_metrics (test_run_id, metric_type, timestamp DESC);

-- ========================================
-- SPECIALIZED MULTI-AGENT INDEXES
-- ========================================

-- Index for finding agents without coalitions
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_no_coalition
ON agents (id, status, template)
WHERE id NOT IN (SELECT DISTINCT agent_id FROM agent_coalition)
  AND status = 'active';

-- Index for coalition recommendation queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_coalition_compatibility
ON agents USING gin (parameters)
WHERE status = 'active';

-- Index for concurrent agent updates
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_concurrent_updates
ON agents (id, updated_at DESC)
WHERE updated_at > NOW() - INTERVAL '5 minutes';

-- Index for coalition formation patterns
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_coalitions_formation_pattern
ON coalitions USING gin (objectives)
WHERE status IN ('forming', 'active');

-- ========================================
-- MONITORING AND MAINTENANCE INDEXES
-- ========================================

-- Index for slow query identification
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_slow_operations
ON agents (id, inference_count, total_steps)
WHERE inference_count > 1000 OR total_steps > 10000;

-- Index for maintenance operations
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_all_tables_updated_at
ON agents (updated_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_coalitions_updated_at
ON coalitions (updated_at DESC);

-- ========================================
-- STATISTICS UPDATE
-- ========================================

-- Update table statistics for query planner
ANALYZE agents;
ANALYZE coalitions;
ANALYZE agent_coalition;
ANALYZE knowledge_nodes;
ANALYZE knowledge_edges;
ANALYZE performance_metrics;

-- ========================================
-- QUERY PLANNER CONFIGURATION
-- ========================================

-- Set planner parameters for multi-agent workloads
-- These are session-level settings that can be made permanent in postgresql.conf

-- Optimize for SSD storage
SET random_page_cost = 1.1;

-- Increase work memory for complex queries
SET work_mem = '32MB';

-- Enable parallel query execution
SET max_parallel_workers_per_gather = 4;

-- Optimize for many concurrent connections
SET effective_cache_size = '4GB';

-- ========================================
-- VALIDATION QUERIES
-- ========================================

-- Validate all indexes were created successfully
SELECT
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'public'
    AND indexname LIKE 'idx_%'
    AND tablename IN ('agents', 'coalitions', 'agent_coalition', 'knowledge_nodes', 'knowledge_edges', 'performance_metrics')
ORDER BY tablename, indexname;

-- Check index usage statistics (run after indexes have been used)
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
ORDER BY idx_scan DESC;
