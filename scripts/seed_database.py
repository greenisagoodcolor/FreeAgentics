#!/usr/bin/env python3
"""
Seed Database Script
Populates the database with initial data for production use
"""

import os
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.connection_manager import DatabaseConnectionManager
from database.models import (
    Agent,
    AgentRole,
    AgentStatus,
    Coalition,
    CoalitionStatus,
    KnowledgeEdge,
    KnowledgeNode,
)


class DatabaseSeeder:
    """Handles database seeding operations."""

    def __init__(self, database_url: str):
        """Initialize seeder with database connection."""
        self.manager = DatabaseConnectionManager(database_url)
        self.engine = self.manager.create_engine_with_pool_config()
        self.Session = sessionmaker(bind=self.engine)

    def seed_agents(self, session) -> List[Agent]:
        """Create initial agents."""
        print("Seeding agents...")

        agents_data = [
            {
                "name": "Explorer Alpha",
                "template": "grid_world",
                "status": AgentStatus.ACTIVE,
                "gmn_spec": "AGENT Explorer\nSTATE position[0,0]\nGOAL explore_map\nACTION move_north,move_south,move_east,move_west",
                "parameters": {
                    "exploration_rate": 0.8,
                    "risk_tolerance": 0.6,
                    "energy_efficiency": 0.7,
                },
                "metrics": {
                    "areas_explored": 0,
                    "distance_traveled": 0,
                    "energy_consumed": 0,
                },
                "position": [0, 0],
            },
            {
                "name": "Guardian Beta",
                "template": "security",
                "status": AgentStatus.ACTIVE,
                "gmn_spec": "AGENT Guardian\nSTATE alert_level[low]\nGOAL maintain_security\nACTION patrol,investigate,report",
                "parameters": {
                    "vigilance": 0.9,
                    "response_time": 0.95,
                    "threat_detection": 0.85,
                },
                "metrics": {
                    "threats_detected": 0,
                    "false_positives": 0,
                    "response_accuracy": 0.98,
                },
            },
            {
                "name": "Analyzer Gamma",
                "template": "data_analysis",
                "status": AgentStatus.ACTIVE,
                "gmn_spec": "AGENT Analyzer\nSTATE processing[idle]\nGOAL analyze_patterns\nACTION collect_data,process,report_findings",
                "parameters": {
                    "accuracy": 0.95,
                    "processing_speed": 0.8,
                    "pattern_recognition": 0.9,
                },
                "metrics": {
                    "patterns_found": 0,
                    "data_processed_gb": 0,
                    "insights_generated": 0,
                },
            },
            {
                "name": "Coordinator Delta",
                "template": "coalition_manager",
                "status": AgentStatus.ACTIVE,
                "gmn_spec": "AGENT Coordinator\nSTATE coordinating[false]\nGOAL optimize_collaboration\nACTION form_coalition,assign_tasks,monitor_progress",
                "parameters": {
                    "leadership": 0.85,
                    "communication": 0.9,
                    "strategic_planning": 0.88,
                },
                "metrics": {
                    "coalitions_formed": 0,
                    "tasks_coordinated": 0,
                    "efficiency_improvement": 0,
                },
            },
            {
                "name": "Scout Echo",
                "template": "grid_world",
                "status": AgentStatus.PAUSED,
                "gmn_spec": "AGENT Scout\nSTATE position[5,5]\nGOAL reconnaissance\nACTION scan,move,report",
                "parameters": {
                    "speed": 0.9,
                    "stealth": 0.7,
                    "observation": 0.85,
                },
                "metrics": {"areas_scouted": 0, "intel_gathered": 0},
                "position": [5, 5],
            },
        ]

        agents = []
        for agent_data in agents_data:
            # Check if agent already exists
            existing = (
                session.query(Agent).filter_by(name=agent_data["name"]).first()
            )
            if existing:
                print(
                    f"  Agent '{agent_data['name']}' already exists, skipping..."
                )
                agents.append(existing)
                continue

            agent = Agent(**agent_data)
            agent.created_at = datetime.utcnow() - timedelta(days=30)
            agent.last_active = datetime.utcnow() - timedelta(hours=1)
            agent.inference_count = 100 + (len(agents) * 50)
            agent.total_steps = 1000 + (len(agents) * 200)

            session.add(agent)
            agents.append(agent)
            print(f"  Created agent: {agent.name}")

        session.commit()
        return agents

    def seed_coalitions(self, session, agents: List[Agent]) -> List[Coalition]:
        """Create initial coalitions."""
        print("\nSeeding coalitions...")

        coalitions_data = [
            {
                "name": "Exploration Team Alpha",
                "description": "Dedicated to mapping unknown territories and discovering resources",
                "status": CoalitionStatus.ACTIVE,
                "objectives": {
                    "primary": "Map entire grid world",
                    "secondary": [
                        "Identify resource locations",
                        "Establish safe routes",
                    ],
                    "progress": 0.45,
                },
                "required_capabilities": [
                    "navigation",
                    "mapping",
                    "resource_detection",
                ],
                "performance_score": 0.78,
                "cohesion_score": 0.85,
                "agent_assignments": [
                    (agents[0], AgentRole.LEADER),  # Explorer Alpha
                    (agents[4], AgentRole.MEMBER),  # Scout Echo
                ],
            },
            {
                "name": "Security Coalition",
                "description": "Maintains system security and responds to threats",
                "status": CoalitionStatus.ACTIVE,
                "objectives": {
                    "primary": "Maintain zero security breaches",
                    "secondary": [
                        "Monitor all sectors",
                        "Rapid threat response",
                    ],
                    "progress": 0.92,
                },
                "required_capabilities": [
                    "threat_detection",
                    "rapid_response",
                    "monitoring",
                ],
                "performance_score": 0.94,
                "cohesion_score": 0.88,
                "agent_assignments": [
                    (agents[1], AgentRole.LEADER),  # Guardian Beta
                    (agents[3], AgentRole.COORDINATOR),  # Coordinator Delta
                ],
            },
            {
                "name": "Data Analysis Consortium",
                "description": "Processes and analyzes system-wide data for insights",
                "status": CoalitionStatus.FORMING,
                "objectives": {
                    "primary": "Process all incoming data streams",
                    "secondary": [
                        "Generate actionable insights",
                        "Predict trends",
                    ],
                    "progress": 0.15,
                },
                "required_capabilities": [
                    "data_processing",
                    "pattern_recognition",
                    "reporting",
                ],
                "performance_score": 0.72,
                "cohesion_score": 0.70,
                "agent_assignments": [
                    (agents[2], AgentRole.LEADER)  # Analyzer Gamma
                ],
            },
        ]

        coalitions = []
        for coalition_data in coalitions_data:
            # Extract agent assignments
            agent_assignments = coalition_data.pop("agent_assignments", [])

            # Check if coalition already exists
            existing = (
                session.query(Coalition)
                .filter_by(name=coalition_data["name"])
                .first()
            )
            if existing:
                print(
                    f"  Coalition '{coalition_data['name']}' already exists, skipping..."
                )
                coalitions.append(existing)
                continue

            coalition = Coalition(**coalition_data)
            coalition.created_at = datetime.utcnow() - timedelta(days=20)

            # Add agents to coalition
            for agent, role in agent_assignments:
                # This will use the association table with role
                coalition.agents.append(agent)
                # Note: The role assignment is handled by the association table

            session.add(coalition)
            coalitions.append(coalition)
            print(
                f"  Created coalition: {coalition.name} with {len(agent_assignments)} agents"
            )

        session.commit()
        return coalitions

    def seed_knowledge_graph(self, session, agents: List[Agent]) -> None:
        """Create initial knowledge graph nodes and edges."""
        print("\nSeeding knowledge graph...")

        # Create knowledge nodes
        nodes_data = [
            {
                "type": "concept",
                "label": "Active Inference",
                "properties": {
                    "description": "Framework for understanding perception and action",
                    "importance": "high",
                    "domain": "cognitive_science",
                },
                "confidence": 1.0,
                "source": "system",
                "creator_agent": agents[2],  # Analyzer Gamma
            },
            {
                "type": "location",
                "label": "Grid World Center",
                "properties": {
                    "coordinates": [50, 50],
                    "terrain": "neutral",
                    "resources": ["energy", "data"],
                },
                "confidence": 0.95,
                "source": "exploration",
                "creator_agent": agents[0],  # Explorer Alpha
            },
            {
                "type": "threat",
                "label": "Anomaly Detection Pattern",
                "properties": {
                    "severity": "medium",
                    "frequency": "periodic",
                    "mitigation": "monitoring",
                },
                "confidence": 0.88,
                "source": "security_scan",
                "creator_agent": agents[1],  # Guardian Beta
            },
            {
                "type": "strategy",
                "label": "Optimal Exploration Path",
                "properties": {
                    "efficiency": 0.82,
                    "coverage": 0.90,
                    "risk_level": "low",
                },
                "confidence": 0.91,
                "source": "analysis",
                "creator_agent": agents[3],  # Coordinator Delta
            },
            {
                "type": "resource",
                "label": "Energy Distribution Network",
                "properties": {
                    "capacity": 1000,
                    "utilization": 0.65,
                    "redundancy": "high",
                },
                "confidence": 0.97,
                "source": "system_analysis",
                "creator_agent": agents[2],  # Analyzer Gamma
            },
        ]

        nodes = []
        for node_data in nodes_data:
            # Extract creator agent
            creator_agent = node_data.pop("creator_agent", None)

            node = KnowledgeNode(**node_data)
            if creator_agent:
                node.creator_agent_id = creator_agent.id

            session.add(node)
            nodes.append(node)
            print(f"  Created knowledge node: {node.label}")

        session.commit()

        # Create edges between nodes
        edges_data = [
            {
                "source": nodes[0],  # Active Inference
                "target": nodes[3],  # Optimal Exploration Path
                "type": "influences",
                "properties": {"strength": 0.8},
                "confidence": 0.9,
            },
            {
                "source": nodes[1],  # Grid World Center
                "target": nodes[4],  # Energy Distribution Network
                "type": "contains",
                "properties": {"distance": 0},
                "confidence": 0.95,
            },
            {
                "source": nodes[2],  # Anomaly Detection Pattern
                "target": nodes[3],  # Optimal Exploration Path
                "type": "constrains",
                "properties": {"impact": "medium"},
                "confidence": 0.85,
            },
            {
                "source": nodes[3],  # Optimal Exploration Path
                "target": nodes[1],  # Grid World Center
                "type": "targets",
                "properties": {"priority": "high"},
                "confidence": 0.92,
            },
        ]

        for edge_data in edges_data:
            source = edge_data.pop("source")
            target = edge_data.pop("target")

            edge = KnowledgeEdge(
                source_id=source.id, target_id=target.id, **edge_data
            )
            session.add(edge)
            print(
                f"  Created edge: {source.label} -> {target.label} ({edge.type})"
            )

        session.commit()

    def seed_all(self):
        """Run all seeding operations."""
        session = self.Session()

        try:
            print("Starting database seeding...")
            print("=" * 50)

            # Seed data in order
            agents = self.seed_agents(session)
            coalitions = self.seed_coalitions(session, agents)
            self.seed_knowledge_graph(session, agents)

            # Print summary
            print("\n" + "=" * 50)
            print("Seeding Summary:")
            print(f"  Agents created: {len(agents)}")
            print(f"  Coalitions created: {len(coalitions)}")
            print(f"  Total agents in system: {session.query(Agent).count()}")
            print(
                f"  Total coalitions in system: {session.query(Coalition).count()}"
            )
            print(
                f"  Total knowledge nodes: {session.query(KnowledgeNode).count()}"
            )
            print(
                f"  Total knowledge edges: {session.query(KnowledgeEdge).count()}"
            )

            print("\n✓ Database seeding completed successfully!")

        except Exception as e:
            print(f"\n✗ Error during seeding: {e}")
            session.rollback()
            raise
        finally:
            session.close()


def main():
    """Main function."""
    # Load environment variables
    env_file = os.path.join(project_root, '.env.production')
    if os.path.exists(env_file):
        load_dotenv(env_file)
        print(f"Loaded environment from: {env_file}")
    else:
        load_dotenv()
        print("Using default .env file")

    # Get database URL
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("✗ DATABASE_URL not found in environment")
        sys.exit(1)

    # Confirm before seeding
    print(
        f"Database URL: {database_url.replace(database_url.split('@')[0].split('//')[1].split(':')[1], '***')}"
    )

    response = input(
        "\nThis will add seed data to the database. Continue? (y/N): "
    )
    if response.lower() != 'y':
        print("Seeding cancelled.")
        sys.exit(0)

    # Run seeding
    seeder = DatabaseSeeder(database_url)
    seeder.seed_all()


if __name__ == "__main__":
    main()
