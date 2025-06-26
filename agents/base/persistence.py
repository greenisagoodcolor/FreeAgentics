import logging
import uuid
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from infrastructure.database.connection import get_db as get_db_session
from infrastructure.database.models import Agent as DBAgent

from .data_model import (
    Agent,
    AgentCapability,
    AgentGoal,
    AgentPersonality,
    AgentResources,
    AgentStatus,
    Orientation,
    Position,
    ResourceAgent,
    SocialAgent,
    SocialRelationship,
)

"""
Agent Persistence and Serialization Module
This module provides functionality for saving and loading agent states to/from
the database, including serialization, deserialization, and version management.
"""
logger = logging.getLogger(__name__)
AGENT_SCHEMA_VERSION = "1.0.0"


class AgentPersistence:
    """Handles persistence operations for agents"""

    def __init__(self, session: Optional[Session] = None) -> None:
        """Initialize persistence handler
        Args:
            session: SQLAlchemy session. If None, will create new sessions as needed.
        """
        self.session = session
        self._use_external_session = session is not None

    def _get_session(self) -> Session:
        """Get database session"""
        if self._use_external_session:
            return self.session
        return get_db_session()

    def save_agent(self, agent: Agent, update_if_exists: bool = True) -> bool:
        """Save agent to database
        Args:
            agent: Agent instance to save
            update_if_exists: If True, update existing agent; if False, raise error
        Returns:
            True if successful, False otherwise
        """
        session = self._get_session()
        try:
            db_agent = session.query(DBAgent).filter_by(uuid=agent.agent_id).first()
            if db_agent and (not update_if_exists):
                logger.error(f"Agent {agent.agent_id} already exists")
                return False
            serialized_data = self._serialize_agent(agent)
            if db_agent:
                db_agent.name = agent.name
                db_agent.type = agent.agent_type
                db_agent.state = serialized_data["state"]
                db_agent.config = serialized_data["config"]
                db_agent.beliefs = serialized_data["beliefs"]
                db_agent.location = serialized_data.get("location")
                db_agent.energy_level = agent.resources.energy / 100.0
                db_agent.experience_points = agent.experience_count
                db_agent.last_active_at = datetime.now()
                db_agent.updated_at = datetime.now()
            else:
                db_agent = DBAgent(
                    uuid=agent.agent_id,
                    name=agent.name,
                    type=agent.agent_type,
                    state=serialized_data["state"],
                    config=serialized_data["config"],
                    beliefs=serialized_data["beliefs"],
                    location=serialized_data.get("location"),
                    energy_level=agent.resources.energy / 100.0,
                    experience_points=agent.experience_count,
                    created_at=agent.created_at,
                    last_active_at=datetime.now(),
                )
                session.add(db_agent)
            if not self._use_external_session:
                session.commit()
            logger.info(f"Successfully saved agent {agent.agent_id}")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Database error saving agent {agent.agent_id}: {e}")
            if not self._use_external_session:
                session.rollback()
            return False
        except Exception as e:
            logger.error(f"Error saving agent {agent.agent_id}: {e}")
            if not self._use_external_session:
                session.rollback()
            return False
        finally:
            if not self._use_external_session:
                session.close()

    def load_agent(self, agent_id: str) -> Optional[Agent]:
        """Load agent from database
        Args:
            agent_id: UUID of agent to load
        Returns:
            Agent instance if found, None otherwise
        """
        session = self._get_session()
        try:
            db_agent = session.query(DBAgent).filter_by(uuid=agent_id).first()
            if not db_agent:
                logger.warning(f"Agent {agent_id} not found in database")
                return None
            agent = self._deserialize_agent(db_agent)
            logger.info(f"Successfully loaded agent {agent_id}")
            return agent
        except Exception as e:
            logger.error(f"Error loading agent {agent_id}: {e}")
            return None
        finally:
            if not self._use_external_session:
                session.close()

    def load_all_agents(
        self, agent_type: Optional[str] = None, status: Optional[str] = None
    ) -> List[Agent]:
        """Load all agents matching criteria
        Args:
            agent_type: Filter by agent type
            status: Filter by agent status
        Returns:
            List of Agent instances
        """
        session = self._get_session()
        try:
            query = session.query(DBAgent)
            if agent_type:
                query = query.filter_by(type=agent_type)
            if status:
                query = query.filter_by(status=status)
            db_agents = query.all()
            agents = []
            for db_agent in db_agents:
                try:
                    agent = self._deserialize_agent(db_agent)
                    agents.append(agent)
                except Exception as e:
                    logger.error(f"Error deserializing agent {db_agent.uuid}: {e}")
            logger.info(f"Loaded {len(agents)} agents")
            return agents
        except Exception as e:
            logger.error(f"Error loading agents: {e}")
            return []
        finally:
            if not self._use_external_session:
                session.close()

    def delete_agent(self, agent_id: str) -> bool:
        """Delete agent from database
        Args:
            agent_id: UUID of agent to delete
        Returns:
            True if successful, False otherwise
        """
        session = self._get_session()
        try:
            db_agent = session.query(DBAgent).filter_by(uuid=agent_id).first()
            if not db_agent:
                logger.warning(f"Agent {agent_id} not found for deletion")
                return False
            session.delete(db_agent)
            if not self._use_external_session:
                session.commit()
            logger.info(f"Successfully deleted agent {agent_id}")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Database error deleting agent {agent_id}: {e}")
            if not self._use_external_session:
                session.rollback()
            return False
        finally:
            if not self._use_external_session:
                session.close()

    def _serialize_agent(self, agent: Agent) -> Dict[str, Any]:
        """Serialize agent to dictionary for database storage
        Args:
            agent: Agent instance
        Returns:
            Dictionary with serialized agent data
        """
        state = {
            "position": {
                "x": agent.position.x,
                "y": agent.position.y,
                "z": agent.position.z,
            },
            "orientation": {
                "w": agent.orientation.w,
                "x": agent.orientation.x,
                "y": agent.orientation.y,
                "z": agent.orientation.z,
            },
            "velocity": (
                agent.velocity.tolist()
                if isinstance(agent.velocity, np.ndarray)
                else agent.velocity
            ),
            "status": agent.status.value,
            "resources": asdict(agent.resources),
            "current_goal": (
                self._serialize_goal(agent.current_goal) if agent.current_goal else None
            ),
            "short_term_memory": agent.short_term_memory[-50:],
            "experience_count": agent.experience_count,
            "schema_version": AGENT_SCHEMA_VERSION,
        }
        config = {
            "capabilities": [cap.value for cap in agent.capabilities],
            "personality": asdict(agent.personality),
            "metadata": agent.metadata,
        }
        beliefs = {
            "relationships": {
                agent_id: {
                    "target_agent_id": rel.target_agent_id,
                    "relationship_type": rel.relationship_type,
                    "trust_level": rel.trust_level,
                    "interaction_count": rel.interaction_count,
                    "last_interaction": (
                        rel.last_interaction.isoformat() if rel.last_interaction else None
                    ),
                }
                for agent_id, rel in agent.relationships.items()
            },
            "goals": [self._serialize_goal(goal) for goal in agent.goals],
            "long_term_memory": agent.long_term_memory[-100:],
            "generative_model_params": agent.generative_model_params,
        }
        if agent.belief_state is not None:
            beliefs["belief_state"] = agent.belief_state.tolist()
        location = None
        return {
            "state": state,
            "config": config,
            "beliefs": beliefs,
            "location": location,
        }

    def _deserialize_agent(self, db_agent) -> Agent:
        """Deserialize agent from database representation
        Args:
            db_agent: Database agent model
        Returns:
            Agent instance
        """
        agent_class = Agent
        if db_agent.type == "resource_management":
            agent_class = ResourceAgent
        elif db_agent.type == "social_interaction":
            agent_class = SocialAgent
        agent = agent_class()
        agent.agent_id = db_agent.uuid
        agent.name = db_agent.name
        agent.agent_type = db_agent.type
        agent.created_at = db_agent.created_at
        agent.last_updated = db_agent.updated_at or datetime.now()
        state = db_agent.state or {}
        if "position" in state:
            agent.position = Position(**state["position"])
        if "orientation" in state:
            agent.orientation = Orientation(**state["orientation"])
        if "velocity" in state:
            agent.velocity = np.array(state["velocity"])
        if "status" in state:
            agent.status = AgentStatus(state["status"])
        if "resources" in state:
            agent.resources = AgentResources(**state["resources"])
        if "current_goal" in state and state["current_goal"]:
            agent.current_goal = self._deserialize_goal(state["current_goal"])
        if "short_term_memory" in state:
            agent.short_term_memory = state["short_term_memory"]
        if "experience_count" in state:
            agent.experience_count = state["experience_count"]
        config = db_agent.config or {}
        if "capabilities" in config:
            agent.capabilities = {AgentCapability(cap) for cap in config["capabilities"]}
        if "personality" in config:
            agent.personality = AgentPersonality(**config["personality"])
        if "metadata" in config:
            agent.metadata = config["metadata"]
        beliefs = db_agent.beliefs or {}
        if "relationships" in beliefs:
            for agent_id, rel_data in beliefs["relationships"].items():
                rel = SocialRelationship(
                    target_agent_id=rel_data["target_agent_id"],
                    relationship_type=rel_data["relationship_type"],
                    trust_level=rel_data["trust_level"],
                    interaction_count=rel_data["interaction_count"],
                    last_interaction=(
                        datetime.fromisoformat(rel_data["last_interaction"])
                        if rel_data["last_interaction"]
                        else None
                    ),
                )
                agent.relationships[agent_id] = rel
        if "goals" in beliefs:
            agent.goals = [self._deserialize_goal(goal_data) for goal_data in beliefs["goals"]]
        if "long_term_memory" in beliefs:
            agent.long_term_memory = beliefs["long_term_memory"]
        if "generative_model_params" in beliefs:
            agent.generative_model_params = beliefs["generative_model_params"]
        if "belief_state" in beliefs:
            agent.belief_state = np.array(beliefs["belief_state"])
        return agent

    def _serialize_goal(self, goal: AgentGoal) -> Dict[str, Any]:
        """Serialize AgentGoal to dictionary"""
        serialized = asdict(goal)
        serialized["goal_id"] = goal.goal_id
        return serialized

    def _deserialize_goal(self, goal_data: Dict[str, Any]) -> AgentGoal:
        """Deserialize dictionary to AgentGoal"""
        goal_id = goal_data.pop("goal_id", None)
        if "target_position" in goal_data and goal_data["target_position"]:
            goal_data["target_position"] = Position(**goal_data["target_position"])
        goal = AgentGoal(**goal_data)
        if goal_id:
            goal.goal_id = goal_id
        return goal


class AgentSnapshot:
    """Handles agent state snapshots for versioning and rollback"""

    def __init__(self, persistence: AgentPersistence) -> None:
        """Initialize snapshot handler
        Args:
            persistence: AgentPersistence instance
        """
        self.persistence = persistence

    def create_snapshot(self, agent: Agent, description: str = "") -> str:
        """Create a snapshot of agent state
        Args:
            agent: Agent to snapshot
            description: Optional description of snapshot
        Returns:
            Snapshot ID
        """
        snapshot_id = str(uuid.uuid4())
        snapshot_data = {
            "snapshot_id": snapshot_id,
            "agent_id": agent.agent_id,
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "agent_data": agent.to_dict(),
            "schema_version": AGENT_SCHEMA_VERSION,
        }
        if "snapshots" not in agent.metadata:
            agent.metadata["snapshots"] = []
        agent.metadata["snapshots"].append(snapshot_data)
        agent.metadata["snapshots"] = agent.metadata["snapshots"][-10:]
        self.persistence.save_agent(agent)
        logger.info(f"Created snapshot {snapshot_id} for agent {agent.agent_id}")
        return snapshot_id

    def restore_snapshot(self, agent_id: str, snapshot_id: str) -> Optional[Agent]:
        """Restore agent from a snapshot
        Args:
            agent_id: Agent UUID
            snapshot_id: Snapshot ID to restore
        Returns:
            Restored Agent instance if successful
        """
        current_agent = self.persistence.load_agent(agent_id)
        if not current_agent:
            logger.error(f"Agent {agent_id} not found")
            return None
        snapshots = current_agent.metadata.get("snapshots", [])
        snapshot = next((s for s in snapshots if s["snapshot_id"] == snapshot_id), None)
        if not snapshot:
            logger.error(f"Snapshot {snapshot_id} not found for agent {agent_id}")
            return None
        agent = Agent.from_dict(snapshot["agent_data"])
        logger.info(f"Restored agent {agent_id} from snapshot {snapshot_id}")
        return agent

    def list_snapshots(self, agent_id: str) -> List[Dict[str, Any]]:
        """List available snapshots for an agent
        Args:
            agent_id: Agent UUID
        Returns:
            List of snapshot metadata
        """
        agent = self.persistence.load_agent(agent_id)
        if not agent:
            return []
        snapshots = agent.metadata.get("snapshots", [])
        return [
            {
                "snapshot_id": s["snapshot_id"],
                "timestamp": s["timestamp"],
                "description": s["description"],
            }
            for s in snapshots
        ]
