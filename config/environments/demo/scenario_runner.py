#!/usr/bin/env python3
"""
FreeAgentics Demo Scenario Runner.

Executes predefined scenarios for compelling demonstrations.
"""

import asyncio
import json
import logging
import os
import random
from datetime import datetime
from typing import Any, Dict, List

import psycopg2
import redis
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ScenarioRunner:
    """Runs predefined demo scenarios."""

    def __init__(self) -> None:
        """Initialize scenario runner."""
        self.db_url = os.environ.get("DATABASE_URL")
        self.redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        interval_env = os.environ.get("SCENARIO_INTERVAL", "300")
        self.scenario_interval = int(interval_env)
        self.active_scenarios = os.environ.get("RUN_SCENARIOS", "").split(",")

        # Connect to database
        self.db_conn = psycopg2.connect(self.db_url)

        # Connect to Redis
        self.redis_client = redis.from_url(self.redis_url)

        self.running = True
        self.current_scenario = None

    def load_scenarios(self) -> List[Dict[str, Any]]:
        """Load active scenarios from database."""
        with self.db_conn.cursor(cursor_factory=RealDictCursor) as cursor:
            if self.active_scenarios and self.active_scenarios[0]:
                # Load specific scenarios
                placeholders = ",".join(["%s"] * len(self.active_scenarios))
                cursor.execute(
                    f"""
                    SELECT * FROM demo.scenarios
                    WHERE name IN ({placeholders}) AND is_active = true
                    ORDER BY name
                """,
                    self.active_scenarios,
                )
            else:
                # Load all active scenarios
                cursor.execute(
                    """
                    SELECT * FROM demo.scenarios
                    WHERE is_active = true
                    ORDER BY name
                """
                )

            return cursor.fetchall()

    def get_agents_by_class(self, agent_class: str) -> List[Dict[str, Any]]:
        """Get agents of a specific class."""
        with self.db_conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT * FROM agents.agents
                WHERE class = %s AND status IN ('active', 'ready')
                ORDER BY created_at
            """,
                (agent_class,),
            )
            return cursor.fetchall()

    async def run_explorer_discovery(self, scenario: Dict[str, Any]):
        """Run explorer resource discovery scenario."""
        logger.info("Starting Explorer Discovery scenario")

        # Get explorer agents
        explorers = self.get_agents_by_class("explorer")
        if not explorers:
            logger.warning("No explorer agents available")
            return

        explorer = explorers[0]

        # Announce scenario
        self.broadcast_scenario_event(
            {
                "scenario": "explorer_discovery",
                "phase": "start",
                "agent": explorer["name"],
                "message": f"{explorer['name']} begins exploration mission",
            }
        )

        # Phase 1: Initial exploration
        await asyncio.sleep(3)
        locations = [
            "Northern Valley",
            "Eastern Ridge",
            "Southern Plains",
            "Western Forest",
        ]
        for location in locations:
            self.broadcast_scenario_event(
                {
                    "scenario": "explorer_discovery",
                    "phase": "explore",
                    "agent": explorer["name"],
                    "location": location,
                    "message": f"{explorer['name']} explores {location}",
                }
            )

            # Simulate resource discovery
            if random.random() > 0.3:
                resources = random.choice(
                    ["abundant food", "fresh water", "rare metals", "ancient artifacts"]
                )
                self.broadcast_scenario_event(
                    {
                        "scenario": "explorer_discovery",
                        "phase": "discovery",
                        "agent": explorer["name"],
                        "location": location,
                        "resources": resources,
                        "message": f"{explorer['name']} discovers {resources} in {location}!",
                    }
                )

                # Update agent stats
                self.update_agent_experience(explorer["id"], "discovery")

            await asyncio.sleep(5)

        # Phase 2: Path optimization
        self.broadcast_scenario_event(
            {
                "scenario": "explorer_discovery",
                "phase": "analysis",
                "agent": explorer["name"],
                "message": f"{explorer['name']} analyzes exploration data to find optimal paths",
            }
        )

        await asyncio.sleep(3)

        # Phase 3: Knowledge sharing
        self.broadcast_scenario_event(
            {
                "scenario": "explorer_discovery",
                "phase": "complete",
                "agent": explorer["name"],
                "message": f"{explorer['name']} shares discovered locations with other agents",
                "knowledge_shared": 4,
            }
        )

        # Log scenario completion
        self.log_scenario_completion(scenario["id"], explorer["id"])

    async def run_merchant_trade(self, scenario: Dict[str, Any]):
        """Run merchant trading scenario."""
        logger.info("Starting Merchant Trade scenario")

        # Get merchant agents
        merchants = self.get_agents_by_class("merchant")
        if len(merchants) < 2:
            logger.warning("Not enough merchant agents for trading")
            return

        merchant1, merchant2 = merchants[:2]

        # Announce scenario
        self.broadcast_scenario_event(
            {
                "scenario": "merchant_trade",
                "phase": "start",
                "agents": [merchant1["name"], merchant2["name"]],
                "message": f"Trading session begins between {merchant1['name']} and {merchant2['name']}",
            }
        )

        # Phase 1: Market analysis
        await asyncio.sleep(2)
        self.broadcast_scenario_event(
            {
                "scenario": "merchant_trade",
                "phase": "analysis",
                "agent": merchant1["name"],
                "message": f"{merchant1['name']} analyzes market conditions",
            }
        )

        # Phase 2: Negotiation
        await asyncio.sleep(3)
        resources = ["food", "water", "metal", "wood", "gems"]
        resource1 = random.choice(resources)
        resource2 = random.choice([r for r in resources if r != resource1])

        offers = [
            {
                "from": merchant1["name"],
                "offer": f"50 units of {resource1}",
                "wants": f"30 units of {resource2}",
            },
            {
                "from": merchant2["name"],
                "offer": f"25 units of {resource2}",
                "wants": f"45 units of {resource1}",
            },
            {
                "from": merchant1["name"],
                "offer": f"45 units of {resource1}",
                "wants": f"25 units of {resource2}",
            },
            {"from": merchant2["name"], "offer": "Deal accepted!"},
        ]

        for offer in offers:
            self.broadcast_scenario_event(
                {
                    "scenario": "merchant_trade",
                    "phase": "negotiation",
                    "data": offer,
                    "message": f"{offer['from']}: {offer.get('offer', '')}",
                }
            )
            await asyncio.sleep(2)

        # Phase 3: Trade execution
        self.broadcast_scenario_event(
            {
                "scenario": "merchant_trade",
                "phase": "complete",
                "agents": [merchant1["name"], merchant2["name"]],
                "message": "Trade completed successfully!",
                "profit": random.randint(10, 50),
            }
        )

        # Update stats
        self.update_agent_experience(merchant1["id"], "trade")
        self.update_agent_experience(merchant2["id"], "trade")

        # Log completion
        self.log_scenario_completion(scenario["id"], merchant1["id"])

    async def run_scholar_research(self, scenario: Dict[str, Any]):
        """Run scholar research scenario."""
        logger.info("Starting Scholar Research scenario")

        # Get scholar agents
        scholars = self.get_agents_by_class("scholar")
        if not scholars:
            logger.warning("No scholar agents available")
            return

        scholar = scholars[0]

        # Announce scenario
        self.broadcast_scenario_event(
            {
                "scenario": "scholar_research",
                "phase": "start",
                "agent": scholar["name"],
                "message": f"{scholar['name']} begins research into agent collaboration patterns",
            }
        )

        # Phase 1: Data collection
        await asyncio.sleep(3)
        self.broadcast_scenario_event(
            {
                "scenario": "scholar_research",
                "phase": "data_collection",
                "agent": scholar["name"],
                "message": f"{scholar['name']} collects behavioral data from 50 agent interactions",
                "data_points": 50,
            }
        )

        # Phase 2: Pattern analysis
        await asyncio.sleep(4)
        patterns = [
            "Agents cooperate 40% more efficiently in groups of 3-4",
            "Resource sharing increases overall survival rate by 65%",
            "Specialized roles emerge naturally after 100+ interactions",
        ]

        for pattern in patterns:
            self.broadcast_scenario_event(
                {
                    "scenario": "scholar_research",
                    "phase": "analysis",
                    "agent": scholar["name"],
                    "discovery": pattern,
                    "message": f"{scholar['name']} discovers: {pattern}",
                }
            )
            await asyncio.sleep(3)

        # Phase 3: Theory formulation
        self.broadcast_scenario_event(
            {
                "scenario": "scholar_research",
                "phase": "theory",
                "agent": scholar["name"],
                "message": f"{scholar['name']} formulates the 'Emergent Cooperation Theory'",
                "confidence": 0.89,
            }
        )

        await asyncio.sleep(2)

        # Phase 4: Knowledge dissemination
        self.broadcast_scenario_event(
            {
                "scenario": "scholar_research",
                "phase": "complete",
                "agent": scholar["name"],
                "message": f"{scholar['name']} publishes research findings to the knowledge graph",
                "impact": "High",
            }
        )

        # Update stats
        self.update_agent_experience(scholar["id"], "research")

        # Log completion
        self.log_scenario_completion(scenario["id"], scholar["id"])

    async def run_guardian_patrol(self, scenario: Dict[str, Any]):
        """Run guardian patrol scenario."""
        logger.info("Starting Guardian Patrol scenario")

        # Get guardian agents
        guardians = self.get_agents_by_class("guardian")
        if not guardians:
            logger.warning("No guardian agents available")
            return

        guardian = guardians[0]

        # Announce scenario
        self.broadcast_scenario_event(
            {
                "scenario": "guardian_patrol",
                "phase": "start",
                "agent": guardian["name"],
                "message": f"{guardian['name']} begins security patrol of the territory",
            }
        )

        # Phase 1: Perimeter establishment
        await asyncio.sleep(2)
        checkpoints = ["North Gate", "East Tower", "South Bridge", "West Outpost"]

        for checkpoint in checkpoints:
            self.broadcast_scenario_event(
                {
                    "scenario": "guardian_patrol",
                    "phase": "patrol",
                    "agent": guardian["name"],
                    "location": checkpoint,
                    "message": f"{guardian['name']} secures {checkpoint}",
                    "status": "clear",
                }
            )
            await asyncio.sleep(2)

        # Phase 2: Threat detection
        if random.random() > 0.5:
            threat_location = random.choice(checkpoints)
            self.broadcast_scenario_event(
                {
                    "scenario": "guardian_patrol",
                    "phase": "alert",
                    "agent": guardian["name"],
                    "location": threat_location,
                    "message": f"{guardian['name']} detects unusual activity at {threat_location}!",
                    "threat_level": "medium",
                }
            )

            await asyncio.sleep(3)

            # Phase 3: Response
            self.broadcast_scenario_event(
                {
                    "scenario": "guardian_patrol",
                    "phase": "response",
                    "agent": guardian["name"],
                    "message": f"{guardian['name']} investigates and neutralizes the threat",
                    "outcome": "success",
                }
            )

        # Phase 4: Report
        await asyncio.sleep(2)
        self.broadcast_scenario_event(
            {
                "scenario": "guardian_patrol",
                "phase": "complete",
                "agent": guardian["name"],
                "message": f"{guardian['name']} completes patrol. Territory secure.",
                "incidents": 1 if random.random() > 0.5 else 0,
            }
        )

        # Update stats
        self.update_agent_experience(guardian["id"], "patrol")

        # Log completion
        self.log_scenario_completion(scenario["id"], guardian["id"])

    async def run_multi_agent_collaboration(self, scenario: Dict[str, Any]):
        """Run multi-agent collaboration scenario."""
        logger.info("Starting Multi-Agent Collaboration scenario")

        # Get one agent of each class
        team = []
        for agent_class in ["explorer", "merchant", "scholar", "guardian"]:
            agents = self.get_agents_by_class(agent_class)
            if agents:
                team.append(agents[0])

        if len(team) < 3:
            logger.warning("Not enough diverse agents for collaboration")
            return

        team_names = [agent["name"] for agent in team]

        # Announce scenario
        self.broadcast_scenario_event(
            {
                "scenario": "multi_agent_collaboration",
                "phase": "start",
                "team": team_names,
                "message": f"Team forms for complex mission: {', '.join(team_names)}",
            }
        )

        # Phase 1: Mission briefing
        await asyncio.sleep(3)
        self.broadcast_scenario_event(
            {
                "scenario": "multi_agent_collaboration",
                "phase": "planning",
                "message": "Team objective: Establish a new outpost in uncharted territory",
            }
        )

        # Phase 2: Role assignment
        await asyncio.sleep(2)
        roles = {
            "explorer": "Scout and map the area",
            "merchant": "Manage resources and supplies",
            "scholar": "Analyze terrain and optimize placement",
            "guardian": "Provide security and establish perimeter",
        }

        for agent in team:
            role = roles.get(agent["class"], "Support operations")
            self.broadcast_scenario_event(
                {
                    "scenario": "multi_agent_collaboration",
                    "phase": "role_assignment",
                    "agent": agent["name"],
                    "role": role,
                    "message": f"{agent['name']} assigned: {role}",
                }
            )
            await asyncio.sleep(1)

        # Phase 3: Execution
        phases = [
            "Explorer scouts optimal location",
            "Scholar analyzes terrain data",
            "Merchant organizes supply chain",
            "Guardian establishes defensive positions",
            "Team collaborates on construction",
        ]

        for phase in phases:
            self.broadcast_scenario_event(
                {
                    "scenario": "multi_agent_collaboration",
                    "phase": "execution",
                    "message": phase,
                    "progress": phases.index(phase) * 20,
                }
            )
            await asyncio.sleep(3)

        # Phase 4: Success
        self.broadcast_scenario_event(
            {
                "scenario": "multi_agent_collaboration",
                "phase": "complete",
                "team": team_names,
                "message": "Outpost successfully established! Team efficiency: 94%",
                "rewards_distributed": len(team),
            }
        )

        # Update all team members
        for agent in team:
            self.update_agent_experience(agent["id"], "collaboration")

        # Log completion
        self.log_scenario_completion(scenario["id"], team[0]["id"])

    def broadcast_scenario_event(self, event: Dict[str, Any]):
        """Broadcast scenario event via Redis."""
        event["timestamp"] = datetime.utcnow().isoformat()
        event["type"] = "scenario"

        try:
            # Publish to scenario channel
            self.redis_client.publish("demo:scenarios", json.dumps(event))

            # Store in recent scenarios
            self.redis_client.lpush("demo:recent_scenarios", json.dumps(event))
            self.redis_client.ltrim("demo:recent_scenarios", 0, 49)  # Keep last 50

            logger.info(f"Scenario event: {event.get('message', 'Unknown')}")

        except Exception as e:
            logger.error(f"Failed to broadcast scenario event: {e}")

    def update_agent_experience(self, agent_id: str, experience_type: str) -> None:
        """Update agent experience from scenario."""
        try:
            with self.db_conn.cursor() as cursor:
                updates = {
                    "discovery": [
                        "experience_count = experience_count + 5",
                        "pattern_count = LEAST(pattern_count + 1, 100)",
                    ],
                    "trade": [
                        "successful_interactions = successful_interactions + 1",
                        "total_interactions = total_interactions + 1",
                    ],
                    "research": [
                        "knowledge_items_shared = knowledge_items_shared + 3",
                        "avg_pattern_confidence = LEAST(avg_pattern_confidence + 0.02, 0.95)",
                    ],
                    "patrol": [
                        "successful_goals = successful_goals + 1",
                        "total_goals_attempted = total_goals_attempted + 1",
                    ],
                    "collaboration": [
                        "unique_collaborators = LEAST(unique_collaborators + 1, 10)",
                        "successful_interactions = successful_interactions + 2",
                    ],
                }

                update_list = updates.get(experience_type, [])
                if update_list:
                    query = f"""
                        UPDATE agents.agent_stats
                        SET {', '.join(update_list)}, updated_at = CURRENT_TIMESTAMP
                        WHERE agent_id = %s
                    """
                    cursor.execute(query, (agent_id,))
                    self.db_conn.commit()

        except Exception as e:
            logger.error(f"Failed to update agent experience: {e}")
            self.db_conn.rollback()

    def log_scenario_completion(self, scenario_id: str, agent_id: str) -> None:
        """Log scenario completion."""
        try:
            with self.db_conn.cursor() as cursor:
                # Update scenario
                cursor.execute(
                    """
                    UPDATE demo.scenarios
                    SET last_run = CURRENT_TIMESTAMP,
                        run_count = run_count + 1
                    WHERE id = %s
                """,
                    (scenario_id,),
                )

                # Log event
                cursor.execute(
                    """
                    INSERT INTO demo.events (scenario_id, event_type, agent_id, description)
                    VALUES (%s, 'scenario_complete', %s, 'Scenario completed successfully')
                """,
                    (scenario_id, agent_id),
                )

                self.db_conn.commit()

        except Exception as e:
            logger.error(f"Failed to log scenario completion: {e}")
            self.db_conn.rollback()

    async def run_scenario_loop(self):
        """Main scenario execution loop."""
        logger.info(f"Starting scenario runner with interval: {self.scenario_interval}s")

        # Load scenarios
        scenarios = self.load_scenarios()
        logger.info(f"Loaded {len(scenarios)} scenarios")

        # Map scenario names to functions
        scenario_handlers = {
            "explorer_discovery": self.run_explorer_discovery,
            "merchant_trade": self.run_merchant_trade,
            "scholar_research": self.run_scholar_research,
            "guardian_patrol": self.run_guardian_patrol,
            "multi_agent_collaboration": self.run_multi_agent_collaboration,
        }

        scenario_index = 0

        while self.running:
            try:
                if scenarios:
                    # Get next scenario
                    scenario = scenarios[scenario_index % len(scenarios)]
                    handler = scenario_handlers.get(scenario["name"])

                    if handler:
                        logger.info(f"Running scenario: {scenario['name']}")
                        self.current_scenario = scenario["name"]

                        # Run scenario
                        await handler(scenario)

                        # Move to next scenario
                        scenario_index += 1
                    else:
                        logger.warning(f"No handler for scenario: {scenario['name']}")

                    # Wait before next scenario
                    await asyncio.sleep(self.scenario_interval)
                else:
                    logger.warning("No scenarios to run")
                    await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Error in scenario loop: {e}")
                await asyncio.sleep(10)

    def cleanup(self):
        """Clean up resources."""
        self.running = False
        self.db_conn.close()
        self.redis_client.close()


async def main():
    """Main entry point."""
    runner = ScenarioRunner()

    try:
        await runner.run_scenario_loop()
    except KeyboardInterrupt:
        logger.info("Shutting down scenario runner...")
    finally:
        runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
