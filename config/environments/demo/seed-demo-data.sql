-- CogniticNet Demo Data Seeding
-- Pre-populates database with interesting demo agents and scenarios

-- Insert demo agents with varied personalities and readiness levels
INSERT INTO agents.agents (id, name, class, status, personality, gnn_model) VALUES
-- Ready Explorer
('a1111111-1111-1111-1111-111111111111', 'Nova Explorer', 'explorer', 'ready',
 '{"openness": 0.9, "conscientiousness": 0.7, "extraversion": 0.6, "agreeableness": 0.8, "neuroticism": 0.3}',
 '{"nodes": [{"id": "n1", "type": "input", "parameters": {"curiosity": 0.9}}, {"id": "n2", "type": "memory", "parameters": {"capacity": 1000}}], "edges": [{"source": "n1", "target": "n2", "weight": 0.8}]}'),

-- Training Merchant
('a2222222-2222-2222-2222-222222222222', 'Zara Trader', 'merchant', 'training',
 '{"openness": 0.6, "conscientiousness": 0.9, "extraversion": 0.8, "agreeableness": 0.7, "neuroticism": 0.4}',
 '{"nodes": [{"id": "n1", "type": "input", "parameters": {"negotiation": 0.8}}, {"id": "n2", "type": "memory", "parameters": {"capacity": 800}}], "edges": [{"source": "n1", "target": "n2", "weight": 0.7}]}'),

-- Active Scholar
('a3333333-3333-3333-3333-333333333333', 'Sage Researcher', 'scholar', 'active',
 '{"openness": 0.95, "conscientiousness": 0.85, "extraversion": 0.4, "agreeableness": 0.9, "neuroticism": 0.2}',
 '{"nodes": [{"id": "n1", "type": "input", "parameters": {"analysis": 0.95}}, {"id": "n2", "type": "memory", "parameters": {"capacity": 1500}}], "edges": [{"source": "n1", "target": "n2", "weight": 0.9}]}'),

-- Ready Guardian
('a4444444-4444-4444-4444-444444444444', 'Atlas Defender', 'guardian', 'ready',
 '{"openness": 0.5, "conscientiousness": 0.95, "extraversion": 0.3, "agreeableness": 0.6, "neuroticism": 0.2}',
 '{"nodes": [{"id": "n1", "type": "input", "parameters": {"vigilance": 0.9}}, {"id": "n2", "type": "memory", "parameters": {"capacity": 1200}}], "edges": [{"source": "n1", "target": "n2", "weight": 0.85}]}'),

-- New Explorer (for character creation demo)
('a5555555-5555-5555-5555-555555555555', 'Echo Wanderer', 'explorer', 'training',
 '{"openness": 0.8, "conscientiousness": 0.6, "extraversion": 0.7, "agreeableness": 0.75, "neuroticism": 0.35}',
 '{"nodes": [{"id": "n1", "type": "input", "parameters": {"curiosity": 0.8}}, {"id": "n2", "type": "memory", "parameters": {"capacity": 600}}], "edges": [{"source": "n1", "target": "n2", "weight": 0.6}]}');

-- Insert agent stats
INSERT INTO agents.agent_stats (agent_id, total_goals_attempted, successful_goals, complex_goals_completed,
    total_interactions, successful_interactions, knowledge_items_shared, energy_efficiency,
    resource_efficiency, sustainability_score, experience_count, pattern_count,
    avg_pattern_confidence, model_update_count, is_model_converged, stable_iterations, unique_collaborators)
VALUES
-- Nova Explorer (ready)
('a1111111-1111-1111-1111-111111111111', 120, 108, 8, 65, 58, 22, 0.88, 0.85, 0.90,
 1450, 72, 0.89, 180, true, 140, 5),

-- Zara Trader (training)
('a2222222-2222-2222-2222-222222222222', 45, 32, 2, 28, 20, 8, 0.72, 0.68, 0.70,
 580, 28, 0.74, 65, false, 35, 3),

-- Sage Researcher (active)
('a3333333-3333-3333-3333-333333333333', 95, 88, 12, 42, 40, 35, 0.92, 0.90, 0.94,
 2100, 95, 0.93, 220, true, 180, 4),

-- Atlas Defender (ready)
('a4444444-4444-4444-4444-444444444444', 110, 102, 7, 55, 52, 18, 0.86, 0.88, 0.87,
 1320, 65, 0.87, 160, true, 125, 4),

-- Echo Wanderer (new)
('a5555555-5555-5555-5555-555555555555', 15, 8, 0, 10, 6, 2, 0.60, 0.55, 0.58,
 180, 8, 0.65, 25, false, 10, 1);

-- Insert readiness evaluations
INSERT INTO agents.readiness_evaluations (agent_id, knowledge_maturity, goal_achievement,
    model_stability, collaboration, resource_management, overall_score, is_ready, metrics, recommendations)
VALUES
-- Nova Explorer (ready)
('a1111111-1111-1111-1111-111111111111', 0.92, 0.90, 0.88, 0.85, 0.89, 0.89, true,
 '{"knowledge": {"experience_count": 1450, "pattern_count": 72}, "goals": {"success_rate": 0.90}}',
 '{}'),

-- Atlas Defender (ready)
('a4444444-4444-4444-4444-444444444444', 0.88, 0.93, 0.86, 0.82, 0.87, 0.87, true,
 '{"knowledge": {"experience_count": 1320, "pattern_count": 65}, "goals": {"success_rate": 0.93}}',
 '{}');

-- Insert demo scenarios
INSERT INTO demo.scenarios (id, name, description, script, duration_seconds) VALUES
('s1111111-1111-1111-1111-111111111111', 'explorer_discovery',
 'Explorer agent discovers new resources and learns optimal paths',
 '{"steps": ["spawn_resources", "explore_area", "learn_paths", "optimize_route"]}', 180),

('s2222222-2222-2222-2222-222222222222', 'merchant_trade',
 'Merchant agents negotiate and complete trades',
 '{"steps": ["setup_market", "negotiate_prices", "execute_trades", "profit_analysis"]}', 240),

('s3333333-3333-3333-3333-333333333333', 'scholar_research',
 'Scholar agents share knowledge and make discoveries',
 '{"steps": ["gather_data", "analyze_patterns", "share_findings", "collaborative_research"]}', 300),

('s4444444-4444-4444-4444-444444444444', 'guardian_patrol',
 'Guardian agents protect territories and respond to threats',
 '{"steps": ["establish_perimeter", "patrol_route", "detect_intrusion", "coordinate_response"]}', 240),

('s5555555-5555-5555-5555-555555555555', 'multi_agent_collaboration',
 'Multiple agents work together to achieve complex goals',
 '{"steps": ["form_team", "assign_roles", "coordinate_actions", "achieve_goal", "share_rewards"]}', 360);

-- Insert sample world grid cells (simplified for demo)
INSERT INTO world.grid_cells (h3_index, center_lat, center_lng, terrain_type, elevation, resources) VALUES
('8928308280fffff', 37.7749, -122.4194, 'plains', 10, '[{"type": "food", "amount": 50}, {"type": "water", "amount": 100}]'),
('8928308280affff', 37.7751, -122.4190, 'forest', 25, '[{"type": "wood", "amount": 200}, {"type": "food", "amount": 30}]'),
('8928308281fffff', 37.7747, -122.4198, 'hills', 45, '[{"type": "stone", "amount": 150}, {"type": "metal", "amount": 20}]'),
('8928308282fffff', 37.7745, -122.4192, 'water', 0, '[{"type": "water", "amount": 500}, {"type": "fish", "amount": 80}]'),
('8928308283fffff', 37.7753, -122.4196, 'mountains', 120, '[{"type": "metal", "amount": 100}, {"type": "gems", "amount": 5}]');

-- Insert sample knowledge nodes
INSERT INTO knowledge.nodes (agent_id, node_type, content, confidence) VALUES
('a1111111-1111-1111-1111-111111111111', 'location', '{"name": "Resource Valley", "coordinates": [37.7749, -122.4194], "resources": ["food", "water"]}', 0.95),
('a1111111-1111-1111-1111-111111111111', 'pattern', '{"type": "resource_regeneration", "interval": 300, "location": "8928308280fffff"}', 0.87),
('a3333333-3333-3333-3333-333333333333', 'discovery', '{"title": "Optimal Foraging Theory", "description": "Resources regenerate faster near water sources"}', 0.92),
('a3333333-3333-3333-3333-333333333333', 'theory', '{"name": "Collaborative Efficiency", "formula": "efficiency = 1.5 * sqrt(agent_count)"}', 0.88);

-- Insert sample conversations
INSERT INTO agents.conversations (id, participant_ids, message_count) VALUES
('c1111111-1111-1111-1111-111111111111', ARRAY['a1111111-1111-1111-1111-111111111111'::uuid, 'a2222222-2222-2222-2222-222222222222'::uuid], 5),
('c2222222-2222-2222-2222-222222222222', ARRAY['a3333333-3333-3333-3333-333333333333'::uuid, 'a4444444-4444-4444-4444-444444444444'::uuid], 8);

-- Insert sample messages
INSERT INTO agents.messages (conversation_id, sender_id, content, message_type) VALUES
('c1111111-1111-1111-1111-111111111111', 'a1111111-1111-1111-1111-111111111111', 'I found abundant resources at coordinates [37.7749, -122.4194]!', 'discovery'),
('c1111111-1111-1111-1111-111111111111', 'a2222222-2222-2222-2222-222222222222', 'Excellent! I can offer 20 units of metal in exchange for 50 food.', 'trade_offer'),
('c1111111-1111-1111-1111-111111111111', 'a1111111-1111-1111-1111-111111111111', 'That seems fair. Let us meet at the trading post.', 'agreement');

-- Insert initial demo events
INSERT INTO demo.events (scenario_id, event_type, agent_id, description, data) VALUES
('s1111111-1111-1111-1111-111111111111', 'scenario_start', 'a1111111-1111-1111-1111-111111111111',
 'Explorer Nova begins resource discovery mission', '{"start_location": "8928308280fffff"}'),
('s3333333-3333-3333-3333-333333333333', 'knowledge_shared', 'a3333333-3333-3333-3333-333333333333',
 'Scholar Sage shares research findings with the community', '{"knowledge_type": "theory", "recipients": 3}');

-- Create a view for agent readiness dashboard
CREATE OR REPLACE VIEW demo.agent_readiness_summary AS
SELECT
    a.id,
    a.name,
    a.class,
    a.status,
    COALESCE(r.overall_score, 0) as readiness_score,
    COALESCE(r.is_ready, false) as is_ready,
    s.experience_count,
    s.successful_goals,
    s.total_goals_attempted,
    CASE
        WHEN s.total_goals_attempted > 0
        THEN ROUND((s.successful_goals::numeric / s.total_goals_attempted) * 100, 1)
        ELSE 0
    END as success_rate
FROM agents.agents a
LEFT JOIN agents.agent_stats s ON a.id = s.agent_id
LEFT JOIN LATERAL (
    SELECT * FROM agents.readiness_evaluations
    WHERE agent_id = a.id
    ORDER BY evaluated_at DESC
    LIMIT 1
) r ON true
ORDER BY r.overall_score DESC NULLS LAST;

-- Create a view for active scenarios
CREATE OR REPLACE VIEW demo.active_scenarios AS
SELECT
    s.name,
    s.description,
    s.duration_seconds,
    s.last_run,
    s.run_count,
    COUNT(DISTINCT e.id) as event_count,
    MAX(e.created_at) as last_event
FROM demo.scenarios s
LEFT JOIN demo.events e ON s.id = e.scenario_id
WHERE s.is_active = true
GROUP BY s.id, s.name, s.description, s.duration_seconds, s.last_run, s.run_count
ORDER BY s.name;
