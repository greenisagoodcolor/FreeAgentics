-- CogniticNet Demo Database Initialization
-- Creates schema and tables for demo environment

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS agents;
CREATE SCHEMA IF NOT EXISTS world;
CREATE SCHEMA IF NOT EXISTS knowledge;
CREATE SCHEMA IF NOT EXISTS demo;

-- Agent Classes Enum
CREATE TYPE agents.agent_class AS ENUM ('explorer', 'merchant', 'scholar', 'guardian');
CREATE TYPE agents.agent_status AS ENUM ('active', 'training', 'ready', 'deployed', 'inactive');

-- Agents Table
CREATE TABLE IF NOT EXISTS agents.agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    class agents.agent_class NOT NULL,
    status agents.agent_status DEFAULT 'training',
    personality JSONB NOT NULL,
    gnn_model JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_active TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Agent Stats Table
CREATE TABLE IF NOT EXISTS agents.agent_stats (
    agent_id UUID PRIMARY KEY REFERENCES agents.agents(id) ON DELETE CASCADE,
    total_goals_attempted INTEGER DEFAULT 0,
    successful_goals INTEGER DEFAULT 0,
    complex_goals_completed INTEGER DEFAULT 0,
    total_interactions INTEGER DEFAULT 0,
    successful_interactions INTEGER DEFAULT 0,
    knowledge_items_shared INTEGER DEFAULT 0,
    energy_efficiency DECIMAL(3,2) DEFAULT 0.50,
    resource_efficiency DECIMAL(3,2) DEFAULT 0.50,
    sustainability_score DECIMAL(3,2) DEFAULT 0.50,
    experience_count INTEGER DEFAULT 0,
    pattern_count INTEGER DEFAULT 0,
    avg_pattern_confidence DECIMAL(3,2) DEFAULT 0.50,
    model_update_count INTEGER DEFAULT 0,
    is_model_converged BOOLEAN DEFAULT FALSE,
    stable_iterations INTEGER DEFAULT 0,
    unique_collaborators INTEGER DEFAULT 0,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Readiness Evaluations Table
CREATE TABLE IF NOT EXISTS agents.readiness_evaluations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID REFERENCES agents.agents(id) ON DELETE CASCADE,
    evaluated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    knowledge_maturity DECIMAL(3,2) NOT NULL,
    goal_achievement DECIMAL(3,2) NOT NULL,
    model_stability DECIMAL(3,2) NOT NULL,
    collaboration DECIMAL(3,2) NOT NULL,
    resource_management DECIMAL(3,2) NOT NULL,
    overall_score DECIMAL(3,2) NOT NULL,
    is_ready BOOLEAN NOT NULL,
    metrics JSONB NOT NULL,
    recommendations TEXT[]
);

-- World Grid Table (H3 hexagons)
CREATE TABLE IF NOT EXISTS world.grid_cells (
    h3_index VARCHAR(15) PRIMARY KEY,
    center_lat DECIMAL(10,7) NOT NULL,
    center_lng DECIMAL(10,7) NOT NULL,
    terrain_type VARCHAR(50) NOT NULL,
    elevation INTEGER DEFAULT 0,
    resources JSONB DEFAULT '[]'::jsonb,
    occupant_id UUID REFERENCES agents.agents(id),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Knowledge Graph Nodes
CREATE TABLE IF NOT EXISTS knowledge.nodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id UUID REFERENCES agents.agents(id) ON DELETE CASCADE,
    node_type VARCHAR(50) NOT NULL,
    content JSONB NOT NULL,
    confidence DECIMAL(3,2) DEFAULT 0.50,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Knowledge Graph Edges
CREATE TABLE IF NOT EXISTS knowledge.edges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID REFERENCES knowledge.nodes(id) ON DELETE CASCADE,
    target_id UUID REFERENCES knowledge.nodes(id) ON DELETE CASCADE,
    edge_type VARCHAR(50) NOT NULL,
    weight DECIMAL(3,2) DEFAULT 0.50,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Agent Conversations
CREATE TABLE IF NOT EXISTS agents.conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    participant_ids UUID[] NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP WITH TIME ZONE,
    message_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Conversation Messages
CREATE TABLE IF NOT EXISTS agents.messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID REFERENCES agents.conversations(id) ON DELETE CASCADE,
    sender_id UUID REFERENCES agents.agents(id),
    content TEXT NOT NULL,
    message_type VARCHAR(50) DEFAULT 'text',
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Demo Scenarios
CREATE TABLE IF NOT EXISTS demo.scenarios (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    script JSONB NOT NULL,
    duration_seconds INTEGER DEFAULT 300,
    is_active BOOLEAN DEFAULT TRUE,
    last_run TIMESTAMP WITH TIME ZONE,
    run_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Demo Events Log
CREATE TABLE IF NOT EXISTS demo.events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scenario_id UUID REFERENCES demo.scenarios(id),
    event_type VARCHAR(100) NOT NULL,
    agent_id UUID REFERENCES agents.agents(id),
    description TEXT,
    data JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_agents_class ON agents.agents(class);
CREATE INDEX idx_agents_status ON agents.agents(status);
CREATE INDEX idx_agents_last_active ON agents.agents(last_active);
CREATE INDEX idx_readiness_agent_id ON agents.readiness_evaluations(agent_id);
CREATE INDEX idx_readiness_evaluated_at ON agents.readiness_evaluations(evaluated_at);
CREATE INDEX idx_grid_occupant ON world.grid_cells(occupant_id);
CREATE INDEX idx_knowledge_agent ON knowledge.nodes(agent_id);
CREATE INDEX idx_conversations_participants ON agents.conversations USING GIN(participant_ids);
CREATE INDEX idx_messages_conversation ON agents.messages(conversation_id);
CREATE INDEX idx_messages_created ON agents.messages(created_at);
CREATE INDEX idx_demo_events_created ON demo.events(created_at);
CREATE INDEX idx_demo_events_scenario ON demo.events(scenario_id);

-- Create update timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update trigger to relevant tables
CREATE TRIGGER update_agents_updated_at BEFORE UPDATE ON agents.agents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agent_stats_updated_at BEFORE UPDATE ON agents.agent_stats
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_knowledge_nodes_updated_at BEFORE UPDATE ON knowledge.nodes
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA agents TO demo;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA world TO demo;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA knowledge TO demo;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA demo TO demo;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA agents TO demo;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA world TO demo;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA knowledge TO demo;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA demo TO demo;
