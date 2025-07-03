"""
Module for FreeAgentics Active Inference implementation.
"""

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


class DatabaseAgentDeserializer:
    """Builder class for deserializing agents from database representation"""

    def __init__(self, persistence: "AgentPersistence", agent: Agent) -> None:
        self.persistence = persistence
        self.agent = agent

    def set_basic_properties(self, db_agent) -> None:
        """Set basic agent properties from database agent"""
        self.agent.agent_id = db_agent.uuid
        self.agent.name = db_agent.name
        self.agent.agent_type = db_agent.type
        self.agent.created_at = db_agent.created_at
        self.agent.last_updated = db_agent.updated_at or datetime.now()

    def set_state_properties(self, state: Dict[str, Any]) -> None:
        """Set state-related properties from database state"""
        if "position" in state:
            self.agent.position = Position(**state["position"])
        if "orientation" in state:
            self.agent.orientation = Orientation(**state["orientation"])
        if "velocity" in state:
            self.agent.velocity = np.array(state["velocity"])
        if "status" in state:
            self.agent.status = AgentStatus(state["status"])
        if "resources" in state:
            self.agent.resources = AgentResources(**state["resources"])
        if "current_goal" in state and state["current_goal"]:
            self.agent.current_goal = self.persistence._deserialize_goal(state["current_goal"])
        if "short_term_memory" in state:
            self.agent.short_term_memory = state["short_term_memory"]
        if "experience_count" in state:
            self.agent.experience_count = state["experience_count"]

    def set_config_properties(self, config: Dict[str, Any]) -> None:
        """Set configuration properties from database config"""
        if "capabilities" in config:
            self.agent.capabilities = {AgentCapability(cap) for cap in config["capabilities"]}
        if "personality" in config:
            self.agent.personality = AgentPersonality(**config["personality"])
        if "metadata" in config:
            self.agent.metadata = config["metadata"]

    def set_belief_properties(self, beliefs: Dict[str, Any]) -> None:
        """Set belief-related properties from database beliefs"""
        self._set_relationships(beliefs)
        self._set_goals(beliefs)
        self._set_memory_and_beliefs(beliefs)

    def _set_relationships(self, beliefs: Dict[str, Any]) -> None:
        """Set relationships from beliefs data"""
        if "relationships" in beliefs:
            for agent_id, rel_data in beliefs["relationships"].items():
                relationship = self.persistence._create_relationship_from_db_data(rel_data)
                self.agent.relationships[agent_id] = relationship

    def _set_goals(self, beliefs: Dict[str, Any]) -> None:
        """Set goals from beliefs data"""
        if "goals" in beliefs:
            self.agent.goals = [
                self.persistence._deserialize_goal(goal_data) for goal_data in beliefs["goals"]
            ]

    def _set_memory_and_beliefs(self, beliefs: Dict[str, Any]) -> None:
        """Set memory and belief state from beliefs data"""
        if "long_term_memory" in beliefs:
            self.agent.long_term_memory = beliefs["long_term_memory"]
        if "generative_model_params" in beliefs:
            self.agent.generative_model_params = beliefs["generative_model_params"]
        if "belief_state" in beliefs:
            self.agent.belief_state = np.array(beliefs["belief_state"])


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
                    logger.error(
                        f"Error deserializing agent {
                            db_agent.uuid}: {e}"
                    )
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
        """Deserialize agent from database representation using Builder pattern"""
        agent = self._create_agent_instance(db_agent.type)
        deserializer = DatabaseAgentDeserializer(self, agent)

        deserializer.set_basic_properties(db_agent)
        deserializer.set_state_properties(db_agent.state or {})
        deserializer.set_config_properties(db_agent.config or {})
        deserializer.set_belief_properties(db_agent.beliefs or {})

        return agent

    def _create_agent_instance(self, agent_type: str) -> Agent:
        """Create appropriate agent instance based on type"""
        if agent_type == "resource_management":
            return ResourceAgent()
        elif agent_type == "social_interaction":
            return SocialAgent()
        else:
            return Agent()

    def _create_relationship_from_db_data(self, rel_data: Dict[str, Any]) -> SocialRelationship:
        """Create relationship from database relationship data"""
        last_interaction = None
        if rel_data.get("last_interaction"):
            last_interaction = datetime.fromisoformat(rel_data["last_interaction"])

        return SocialRelationship(
            target_agent_id=rel_data["target_agent_id"],
            relationship_type=rel_data["relationship_type"],
            trust_level=rel_data["trust_level"],
            interaction_count=rel_data["interaction_count"],
            last_interaction=last_interaction,
        )

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
        logger.info(
            f"Created snapshot {snapshot_id} for agent {
                agent.agent_id}"
        )
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
