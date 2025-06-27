#!/usr/bin/env python3
"""
FreeAgentics Demo Simulator.

Runs agents at accelerated speed for compelling demonstrations.
"""

import asyncio
import json
import logging
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

import psycopg2
import redis
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DemoSimulator:
    """Runs accelerated agent simulations for demos"""

    def __init__(self) -> None:
        """Initialize the demo simulator"""
        self.db_url = os.environ.get("DATABASE_URL")
        self.redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        speed_env = os.environ.get("SIMULATION_SPEED", "10").rstrip("x")
        self.simulation_speed = float(speed_env)
        self.auto_play = os.environ.get("AUTO_PLAY", "true").lower() == "true"
        loop_env = os.environ.get("SCENARIO_LOOP", "true").lower()
        self.scenario_loop = loop_env == "true"

        # Connect to database
        self.db_conn = psycopg2.connect(self.db_url)

        # Connect to Redis for real-time updates
        self.redis_client = redis.from_url(self.redis_url)
        self.pubsub = self.redis_client.pubsub()

        # Track active agents
        self.active_agents = {}
        self.running = True

    def load_agents(self) -> List[Dict[str, Any]]:
        """Load active agents from database"""
        with self.db_conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT a.*, s.*
                FROM agents.agents a
                LEFT JOIN agents.agent_stats s ON a.id = s.agent_id
                WHERE a.status IN ('active', 'training', 'ready')
                ORDER BY a.created_at
            """
            )
            return cursor.fetchall()

    def simulate_agent_action(self, agent: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Simulate a single agent action"""
        agent_id = agent["id"]
        agent_class = agent["class"]

        # Choose action based on agent class
        action_weights = {
            "explorer": {
                "explore": 0.4,
                "analyze_location": 0.3,
                "share_discovery": 0.2,
                "rest": 0.1,
            },
            "merchant": {
                "seek_trade": 0.35,
                "negotiate": 0.3,
                "analyze_market": 0.25,
                "rest": 0.1,
            },
            "scholar": {
                "research": 0.4,
                "analyze_data": 0.3,
                "share_knowledge": 0.25,
                "rest": 0.05,
            },
            "guardian": {
                "patrol": 0.4,
                "scan_threats": 0.3,
                "coordinate": 0.2,
                "rest": 0.1,
            },
        }

        weights = action_weights.get(agent_class, {"rest": 1.0})
        action = random.choices(list(weights.keys()), weights=list(weights.values()))[0]

        # Generate action result
        result = {
            "agent_id": agent_id,
            "agent_name": agent["name"],
            "action": action,
            "timestamp": datetime.utcnow().isoformat(),
            "success": random.random() > 0.2,  # 80% success rate
        }

        # Add action-specific data
        if action == "explore":
            result["data"] = {
                "location": f"h3_{random.randint(1000, 9999)}",
                "resources_found": random.choice(["food", "water", "metal", "wood", "none"]),
            }
        elif action == "share_discovery" or action == "share_knowledge":
            result["data"] = {
                "knowledge_type": random.choice(["location", "pattern", "theory", "warning"]),
                "confidence": round(random.uniform(0.6, 0.95), 2),
            }
        elif action == "negotiate":
            result["data"] = {
                "offer_type": random.choice(["buy", "sell", "trade"]),
                "resource": random.choice(["food", "water", "metal", "wood"]),
                "quantity": random.randint(10, 100),
            }

        return result

    def update_agent_stats(self, agent_id: str, action_result: Dict[str, Any]) -> None:
        """Update agent statistics based on action results"""
        try:
            with self.db_conn.cursor() as cursor:
                if action_result["success"]:
                    # Update successful actions
                    updates = []

                    if action_result["action"] in ["explore", "research"]:
                        updates.append("experience_count = experience_count + 1")

                    if action_result["action"] in [
                        "share_discovery",
                        "share_knowledge",
                    ]:
                        updates.append("knowledge_items_shared = " "knowledge_items_shared + 1")

                    if action_result["action"] == "negotiate":
                        updates.append("successful_interactions = " "successful_interactions + 1")
                        updates.append("total_interactions = total_interactions + 1")

                    if random.random() > 0.7:  # 30% chance to complete goal
                        updates.append("successful_goals = successful_goals + 1")
                        updates.append("total_goals_attempted = " "total_goals_attempted + 1")

                    if updates:
                        query = f"""
                            UPDATE agents.agent_stats
                            SET {', '.join(updates)},
                            updated_at = CURRENT_TIMESTAMP
                            WHERE agent_id = %s
                        """
                        cursor.execute(query, (agent_id,))

                # Always update last active
                cursor.execute(
                    """
                    UPDATE agents.agents
                    SET last_active = CURRENT_TIMESTAMP
                    WHERE id = %s
                """,
                    (agent_id,),
                )

                self.db_conn.commit()

        except Exception as e:
            logger.error(f"Failed to update agent stats: {e}")
            self.db_conn.rollback()

    def broadcast_event(self, event: Dict[str, Any]):
        """Broadcast event via Redis for real-time updates"""
        try:
            channel = f"demo:events:{event.get('agent_id', 'global')}"
            self.redis_client.publish(channel, json.dumps(event))

            # Also publish to global channel
            self.redis_client.publish("demo:events:all", json.dumps(event))

            # Store recent events
            self.redis_client.lpush("demo:recent_events", json.dumps(event))
            self.redis_client.ltrim("demo:recent_events", 0, 99)

        except Exception as e:
            logger.error(f"Failed to broadcast event: {e}")

    def log_demo_event(
        self,
        event_type: str,
        agent_id: Optional[str],
        description: str,
        data: Dict[str, Any] = None,
    ) -> None:
        """Log event to demo events table"""
        try:
            with self.db_conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO demo.events
                    (event_type, agent_id, description, data)
                    VALUES (%s, %s, %s, %s)
                """,
                    (event_type, agent_id, description, json.dumps(data or {})),
                )
                self.db_conn.commit()
        except Exception as e:
            logger.error(f"Failed to log demo event: {e}")
            self.db_conn.rollback()

    async def simulate_conversations(self):
        """Simulate agent conversations"""
        while self.running:
            try:
                # Random delay between conversations
                delay = random.uniform(20, 40) / self.simulation_speed
                await asyncio.sleep(delay)

                # Get two random active agents
                agents = list(self.active_agents.values())
                if len(agents) < 2:
                    continue

                agent1, agent2 = random.sample(agents, 2)

                # Create conversation
                conversation_types = [
                    {
                        "type": "trade_negotiation",
                        "messages": [
                            f"Greetings {agent2['name']}, I have resources " f"to trade.",
                            f"Welcome {agent1['name']}, what do you offer?",
                            "I can provide 50 units of food for 20 metal.",
                            "That's acceptable. Let's proceed with the " "exchange.",
                            "Excellent! Trade completed successfully.",
                        ],
                    },
                    {
                        "type": "knowledge_exchange",
                        "messages": [
                            f"Hello {agent2['name']}, I've discovered " f"something interesting.",
                            "Please share your findings!",
                            "Resources regenerate 50% faster near water " "sources.",
                            "Fascinating! I've noticed similar patterns in " "the eastern regions.",
                            "We should collaborate on further research.",
                        ],
                    },
                    {
                        "type": "coordination",
                        "messages": [
                            f"{agent2['name']}, we need to coordinate our " f"efforts.",
                            "Agreed. What do you propose?",
                            "I'll explore the northern territories while you " "secure the south.",
                            "Good plan. I'll establish a base at the southern " "checkpoint.",
                            "Perfect. Let's reconvene in 2 cycles.",
                        ],
                    },
                ]

                conversation = random.choice(conversation_types)

                # Log conversation start
                self.log_demo_event(
                    "conversation_start",
                    agent1["id"],
                    f"{agent1['name']} initiates {conversation['type']} " f"with {agent2['name']}",
                    {
                        "conversation_type": conversation["type"],
                        "participant_ids": [agent1["id"], agent2["id"]],
                    },
                )

                # Simulate message exchange
                for i, message in enumerate(conversation["messages"]):
                    sender = agent1 if i % 2 == 0 else agent2

                    # Broadcast message event
                    self.broadcast_event(
                        {
                            "type": "message",
                            "agent_id": sender["id"],
                            "agent_name": sender["name"],
                            "message": message,
                            "conversation_type": conversation["type"],
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )

                    # Small delay between messages
                    await asyncio.sleep(2 / self.simulation_speed)

                # Update interaction stats
                self.update_agent_stats(agent1["id"], {"action": "negotiate", "success": True})
                self.update_agent_stats(agent2["id"], {"action": "negotiate", "success": True})

            except Exception as e:
                logger.error(f"Error in conversation simulation: {e}")

    async def run_simulation_loop(self):
        """Run main simulation loop"""
        logger.info(f"Starting demo simulator at " f"{self.simulation_speed}x speed")

        # Load agents
        agents = self.load_agents()
        for agent in agents:
            self.active_agents[agent["id"]] = agent
            logger.info(f"Loaded agent: {agent['name']} ({agent['class']})")

        # Log simulation start
        self.log_demo_event(
            "simulation_start",
            None,
            f"Demo simulation started with {len(agents)} agents "
            f"at {self.simulation_speed}x speed",
        )

        # Start conversation simulator
        conversation_task = asyncio.create_task(self.simulate_conversations())

        # Main action loop
        action_interval = 5 / self.simulation_speed

        while self.running:
            try:
                # Simulate actions for each active agent
                for agent in self.active_agents.values():
                    # Simulate action
                    action_result = self.simulate_agent_action(agent)

                    if action_result:
                        # Update stats
                        self.update_agent_stats(agent["id"], action_result)

                        # Broadcast event
                        self.broadcast_event(action_result)

                        # Log significant events
                        if action_result["action"] in [
                            "share_discovery",
                            "share_knowledge",
                        ]:
                            self.log_demo_event(
                                "knowledge_shared",
                                agent["id"],
                                f"{agent['name']} shares "
                                f"{action_result['data']['knowledge_type']}",
                                action_result["data"],
                            )

                # Check for readiness improvements
                if random.random() > 0.9:  # 10% chance per cycle
                    agent = random.choice(list(self.active_agents.values()))
                    self.simulate_readiness_improvement(agent)

                # Wait for next cycle
                await asyncio.sleep(action_interval)

            except Exception as e:
                logger.error(f"Error in simulation loop: {e}")
                await asyncio.sleep(1)

        # Cleanup
        conversation_task.cancel()
        await conversation_task

    def simulate_readiness_improvement(self, agent: Dict[str, Any]):
        """Simulate improvement in agent readiness"""
        try:
            with self.db_conn.cursor() as cursor:
                # Improve random stats
                improvements = [
                    "pattern_count = LEAST(pattern_count + 1, 100)",
                    "avg_pattern_confidence = LEAST(" "avg_pattern_confidence + 0.01, 0.95)",
                    "energy_efficiency = LEAST(" "energy_efficiency + 0.01, 0.95)",
                    "model_update_count = model_update_count + 1",
                ]

                improvement = random.choice(improvements)

                cursor.execute(
                    f"""
                    UPDATE agents.agent_stats
                    SET {improvement}, updated_at = CURRENT_TIMESTAMP
                    WHERE agent_id = %s
                """,
                    (agent["id"],),
                )

                self.db_conn.commit()

                # Broadcast improvement
                self.broadcast_event(
                    {
                        "type": "improvement",
                        "agent_id": agent["id"],
                        "agent_name": agent["name"],
                        "improvement": improvement.split(" = ")[0],
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

                logger.info(f"Agent {agent['name']} improved: {improvement}")

        except Exception as e:
            logger.error(f"Failed to simulate improvement: {e}")
            self.db_conn.rollback()

    def cleanup(self):
        """Clean up resources"""
        self.running = False
        self.db_conn.close()
        self.redis_client.close()


async def main():
    """Run main entry point"""
    simulator = DemoSimulator()

    try:
        await simulator.run_simulation_loop()
    except KeyboardInterrupt:
        logger.info("Shutting down simulator...")
    finally:
        simulator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
