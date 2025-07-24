-- FreeAgentics Test Database Schema
-- Matches production schema with additional test-specific tables

-- Create custom types
DO $$ BEGIN
    CREATE TYPE agentstatus AS ENUM ('PENDING', 'ACTIVE', 'PAUSED', 'STOPPED', 'ERROR');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE coalitionstatus AS ENUM ('FORMING', 'ACTIVE', 'DISBANDING', 'DISSOLVED');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE agentrole AS ENUM ('LEADER', 'COORDINATOR', 'MEMBER', 'OBSERVER');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Create agents table
CREATE TABLE IF NOT EXISTS agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    template VARCHAR(50) NOT NULL,
    status agentstatus DEFAULT 'PENDING',
    gmn_spec TEXT,
    pymdp_config JSON,
    beliefs JSON,
    preferences JSON,
    position JSON,
    metrics JSON,
    parameters JSON,
    created_at TIMESTAMP DEFAULT NOW(),
    last_active TIMESTAMP,
    updated_at TIMESTAMP DEFAULT NOW(),
    inference_count INTEGER DEFAULT 0,
    total_steps INTEGER DEFAULT 0
);

-- Create coalitions table
CREATE TABLE IF NOT EXISTS coalitions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    status coalitionstatus DEFAULT 'FORMING',
    objectives JSON,
    required_capabilities JSON,
    achieved_objectives JSON,
    performance_score DOUBLE PRECISION,
    cohesion_score DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT NOW(),
    dissolved_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create agent_coalition junction table
CREATE TABLE IF NOT EXISTS agent_coalition (
    agent_id UUID NOT NULL,
    coalition_id UUID NOT NULL,
    role agentrole DEFAULT 'MEMBER',
    joined_at TIMESTAMP DEFAULT NOW(),
    contribution_score DOUBLE PRECISION,
    PRIMARY KEY (agent_id, coalition_id),
    FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE,
    FOREIGN KEY (coalition_id) REFERENCES coalitions(id) ON DELETE CASCADE
);

-- Create knowledge nodes table
CREATE TABLE IF NOT EXISTS knowledge_nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type VARCHAR(50) NOT NULL,
    label VARCHAR(200) NOT NULL,
    properties JSON,
    version INTEGER DEFAULT 1,
    is_current BOOLEAN DEFAULT TRUE,
    confidence DOUBLE PRECISION,
    source VARCHAR(100),
    creator_agent_id UUID,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (creator_agent_id) REFERENCES agents(id) ON DELETE SET NULL
);

-- Create knowledge edges table
CREATE TABLE IF NOT EXISTS knowledge_edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID NOT NULL,
    target_id UUID NOT NULL,
    type VARCHAR(50) NOT NULL,
    properties JSON,
    confidence DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (source_id) REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES knowledge_nodes(id) ON DELETE CASCADE
);

-- Create database knowledge nodes table (for testing database-backed knowledge)
CREATE TABLE IF NOT EXISTS db_knowledge_nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type VARCHAR(50) NOT NULL,
    label VARCHAR(200) NOT NULL,
    properties JSON,
    version INTEGER DEFAULT 1,
    is_current BOOLEAN DEFAULT TRUE,
    confidence DOUBLE PRECISION,
    source VARCHAR(100),
    creator_agent_id UUID,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (creator_agent_id) REFERENCES agents(id) ON DELETE SET NULL
);

-- Create database knowledge edges table
CREATE TABLE IF NOT EXISTS db_knowledge_edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID NOT NULL,
    target_id UUID NOT NULL,
    type VARCHAR(50) NOT NULL,
    properties JSON,
    confidence DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (source_id) REFERENCES db_knowledge_nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES db_knowledge_nodes(id) ON DELETE CASCADE
);

-- Test-specific tables for performance monitoring
CREATE TABLE IF NOT EXISTS test_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    test_name VARCHAR(200) NOT NULL,
    start_time TIMESTAMP NOT NULL DEFAULT NOW(),
    end_time TIMESTAMP,
    status VARCHAR(50) DEFAULT 'RUNNING',
    total_agents INTEGER,
    total_coalitions INTEGER,
    total_operations INTEGER,
    metadata JSON,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    test_run_id UUID NOT NULL,
    metric_type VARCHAR(100) NOT NULL,
    metric_name VARCHAR(200) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    unit VARCHAR(50),
    timestamp TIMESTAMP DEFAULT NOW(),
    metadata JSON,
    FOREIGN KEY (test_run_id) REFERENCES test_runs(id) ON DELETE CASCADE
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);
CREATE INDEX IF NOT EXISTS idx_agents_template ON agents(template);
CREATE INDEX IF NOT EXISTS idx_agents_created_at ON agents(created_at);
CREATE INDEX IF NOT EXISTS idx_agents_last_active ON agents(last_active);

CREATE INDEX IF NOT EXISTS idx_coalitions_status ON coalitions(status);
CREATE INDEX IF NOT EXISTS idx_coalitions_created_at ON coalitions(created_at);

CREATE INDEX IF NOT EXISTS idx_agent_coalition_agent_id ON agent_coalition(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_coalition_coalition_id ON agent_coalition(coalition_id);
CREATE INDEX IF NOT EXISTS idx_agent_coalition_joined_at ON agent_coalition(joined_at);

CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_type ON knowledge_nodes(type);
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_creator ON knowledge_nodes(creator_agent_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_created_at ON knowledge_nodes(created_at);
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_current ON knowledge_nodes(is_current);

CREATE INDEX IF NOT EXISTS idx_knowledge_edges_source ON knowledge_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_edges_target ON knowledge_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_edges_type ON knowledge_edges(type);

CREATE INDEX IF NOT EXISTS idx_db_knowledge_nodes_type ON db_knowledge_nodes(type);
CREATE INDEX IF NOT EXISTS idx_db_knowledge_nodes_creator ON db_knowledge_nodes(creator_agent_id);

CREATE INDEX IF NOT EXISTS idx_performance_metrics_test_run ON performance_metrics(test_run_id);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_type ON performance_metrics(metric_type);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp ON performance_metrics(timestamp);

-- Create update timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at columns
DROP TRIGGER IF EXISTS update_agents_updated_at ON agents;
CREATE TRIGGER update_agents_updated_at BEFORE UPDATE ON agents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_coalitions_updated_at ON coalitions;
CREATE TRIGGER update_coalitions_updated_at BEFORE UPDATE ON coalitions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_knowledge_nodes_updated_at ON knowledge_nodes;
CREATE TRIGGER update_knowledge_nodes_updated_at BEFORE UPDATE ON knowledge_nodes
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_db_knowledge_nodes_updated_at ON db_knowledge_nodes;
CREATE TRIGGER update_db_knowledge_nodes_updated_at BEFORE UPDATE ON db_knowledge_nodes
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
