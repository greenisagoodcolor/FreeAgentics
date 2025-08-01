"""
Belief State Persistence Service

Provides database-backed persistence for agent belief states to enable
conversation resumption and long-term belief tracking.

Following Nemesis Committee consensus:
- Robert C. Martin: Clean repository pattern with domain separation
- Martin Fowler: Enterprise patterns for data persistence
- Michael Feathers: Proper error handling and fallback strategies
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from inference.active.belief_manager import BeliefState, BeliefStateRepository

logger = logging.getLogger(__name__)

Base = declarative_base()


class BeliefStateRecord(Base):
    """Database model for belief state persistence."""

    __tablename__ = "belief_states"

    id = Column(String, primary_key=True)  # composite key: agent_id + timestamp
    agent_id = Column(String, nullable=False, index=True)
    conversation_id = Column(String, nullable=True, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Serialized belief state data
    beliefs_json = Column(Text, nullable=False)  # JSON array of belief probabilities
    entropy = Column(Float, nullable=False)
    max_confidence = Column(Float, nullable=False)
    effective_states = Column(Integer, nullable=False)
    most_likely_state = Column(Integer, nullable=False)

    # Observation history
    observation_history_json = Column(Text, nullable=False, default="[]")
    observation_count = Column(Integer, nullable=False, default=0)

    # Metadata for monitoring
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    is_current = Column(Boolean, nullable=False, default=True)

    def __repr__(self):
        return f"<BeliefStateRecord(agent_id='{self.agent_id}', timestamp={self.timestamp})>"


@dataclass
class BeliefPersistenceConfig:
    """Configuration for belief state persistence."""

    database_url: str = "sqlite:///./belief_states.db"
    max_history_per_agent: int = 1000
    enable_compression: bool = False
    backup_to_file: bool = True
    backup_directory: str = "./belief_backups"
    cleanup_old_states_days: int = 30


class DatabaseBeliefRepository(BeliefStateRepository):
    """Database-backed implementation of belief state repository."""

    def __init__(self, config: BeliefPersistenceConfig):
        """Initialize database belief repository.

        Args:
            config: Persistence configuration
        """
        self.config = config
        self._engine = None
        self._session_factory = None
        self._initialized = False

        # File backup functionality
        if config.backup_to_file:
            self._backup_dir = Path(config.backup_directory)
            self._backup_dir.mkdir(exist_ok=True)

    async def initialize(self):
        """Initialize database connection and tables."""
        if self._initialized:
            return

        try:
            # Create async engine
            self._engine = create_async_engine(
                self.config.database_url,
                echo=False,  # Set to True for SQL debugging
                future=True,
            )

            # Create session factory
            self._session_factory = sessionmaker(
                self._engine, class_=AsyncSession, expire_on_commit=False
            )

            # Create tables
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            self._initialized = True
            logger.info("Database belief repository initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database belief repository: {e}")
            raise

    async def save_belief_state(self, belief_state: BeliefState, agent_id: str) -> None:
        """Save belief state for an agent."""
        if not self._initialized:
            await self.initialize()

        try:
            # Mark previous states as non-current
            async with self._session_factory() as session:
                # Update previous current states
                await session.execute(
                    f"UPDATE belief_states SET is_current = FALSE WHERE agent_id = '{agent_id}' AND is_current = TRUE"
                )

                # Create new record
                record = BeliefStateRecord(
                    id=f"{agent_id}_{belief_state.timestamp}",
                    agent_id=agent_id,
                    timestamp=datetime.fromtimestamp(belief_state.timestamp),
                    beliefs_json=json.dumps(belief_state.beliefs.tolist()),
                    entropy=belief_state.entropy,
                    max_confidence=belief_state.max_confidence,
                    effective_states=belief_state.effective_states,
                    most_likely_state=belief_state.most_likely_state(),
                    observation_history_json=json.dumps(belief_state.observation_history),
                    observation_count=len(belief_state.observation_history),
                    is_current=True,
                )

                session.add(record)
                await session.commit()

                logger.debug(f"Saved belief state for agent {agent_id}")

                # Create backup if enabled
                if self.config.backup_to_file:
                    await self._backup_belief_state(belief_state, agent_id)

        except Exception as e:
            logger.error(f"Failed to save belief state for agent {agent_id}: {e}")
            raise

    async def get_current_belief_state(self, agent_id: str) -> Optional[BeliefState]:
        """Get current belief state for an agent."""
        if not self._initialized:
            await self.initialize()

        try:
            async with self._session_factory() as session:
                result = await session.execute(
                    f"SELECT * FROM belief_states WHERE agent_id = '{agent_id}' AND is_current = TRUE LIMIT 1"
                )
                record = result.fetchone()

                if not record:
                    return None

                return self._record_to_belief_state(record)

        except Exception as e:
            logger.error(f"Failed to get current belief state for agent {agent_id}: {e}")
            return None

    async def get_belief_history(
        self, agent_id: str, limit: Optional[int] = None
    ) -> List[BeliefState]:
        """Get belief history for an agent."""
        if not self._initialized:
            await self.initialize()

        try:
            query_limit = limit or self.config.max_history_per_agent

            async with self._session_factory() as session:
                result = await session.execute(
                    f"SELECT * FROM belief_states WHERE agent_id = '{agent_id}' "
                    f"ORDER BY timestamp DESC LIMIT {query_limit}"
                )
                records = result.fetchall()

                belief_states = []
                for record in records:
                    belief_state = self._record_to_belief_state(record)
                    if belief_state:
                        belief_states.append(belief_state)

                return belief_states

        except Exception as e:
            logger.error(f"Failed to get belief history for agent {agent_id}: {e}")
            return []

    async def clear_history(self, agent_id: str) -> None:
        """Clear belief history for an agent."""
        if not self._initialized:
            await self.initialize()

        try:
            async with self._session_factory() as session:
                await session.execute(f"DELETE FROM belief_states WHERE agent_id = '{agent_id}'")
                await session.commit()

                logger.debug(f"Cleared belief history for agent {agent_id}")

        except Exception as e:
            logger.error(f"Failed to clear belief history for agent {agent_id}: {e}")
            raise

    def _record_to_belief_state(self, record) -> Optional[BeliefState]:
        """Convert database record to BeliefState object."""
        try:
            beliefs = np.array(json.loads(record.beliefs_json))
            observation_history = json.loads(record.observation_history_json)

            return BeliefState(
                beliefs=beliefs,
                timestamp=record.timestamp.timestamp(),
                observation_history=observation_history,
            )

        except Exception as e:
            logger.error(f"Failed to convert record to belief state: {e}")
            return None

    async def _backup_belief_state(self, belief_state: BeliefState, agent_id: str) -> None:
        """Create file backup of belief state."""
        try:
            backup_data = {
                "agent_id": agent_id,
                "timestamp": belief_state.timestamp,
                "beliefs": belief_state.beliefs.tolist(),
                "entropy": belief_state.entropy,
                "max_confidence": belief_state.max_confidence,
                "effective_states": belief_state.effective_states,
                "most_likely_state": belief_state.most_likely_state(),
                "observation_history": belief_state.observation_history,
                "created_at": datetime.now().isoformat(),
            }

            timestamp_str = datetime.fromtimestamp(belief_state.timestamp).strftime("%Y%m%d_%H%M%S")
            filename = self._backup_dir / f"{agent_id}_{timestamp_str}.json"

            with open(filename, "w") as f:
                json.dump(backup_data, f, indent=2)

            logger.debug(f"Created backup file: {filename}")

        except Exception as e:
            logger.warning(f"Failed to create backup for agent {agent_id}: {e}")

    async def cleanup_old_states(self) -> int:
        """Clean up old belief states based on configuration."""
        if not self._initialized:
            await self.initialize()

        try:
            cutoff_date = datetime.now() - timedelta(days=self.config.cleanup_old_states_days)

            async with self._session_factory() as session:
                result = await session.execute(
                    f"DELETE FROM belief_states WHERE timestamp < '{cutoff_date}' AND is_current = FALSE"
                )
                deleted_count = result.rowcount
                await session.commit()

                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} old belief states")

                return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup old belief states: {e}")
            return 0

    async def get_persistence_stats(self) -> Dict[str, Any]:
        """Get persistence statistics for monitoring."""
        if not self._initialized:
            await self.initialize()

        try:
            async with self._session_factory() as session:
                # Total states
                result = await session.execute("SELECT COUNT(*) FROM belief_states")
                total_states = result.scalar()

                # Current states (active agents)
                result = await session.execute(
                    "SELECT COUNT(*) FROM belief_states WHERE is_current = TRUE"
                )
                current_states = result.scalar()

                # Unique agents
                result = await session.execute("SELECT COUNT(DISTINCT agent_id) FROM belief_states")
                unique_agents = result.scalar()

                # Oldest and newest timestamps
                result = await session.execute(
                    "SELECT MIN(timestamp), MAX(timestamp) FROM belief_states"
                )
                min_time, max_time = result.fetchone()

                return {
                    "total_belief_states": total_states,
                    "current_active_states": current_states,
                    "unique_agents": unique_agents,
                    "oldest_state": min_time.isoformat() if min_time else None,
                    "newest_state": max_time.isoformat() if max_time else None,
                    "database_url": self.config.database_url,
                    "backup_enabled": self.config.backup_to_file,
                    "max_history_per_agent": self.config.max_history_per_agent,
                }

        except Exception as e:
            logger.error(f"Failed to get persistence stats: {e}")
            return {"error": str(e)}

    async def close(self):
        """Close database connections."""
        if self._engine:
            await self._engine.dispose()
            self._initialized = False
            logger.info("Database belief repository closed")


class FallbackFileRepository(BeliefStateRepository):
    """File-based fallback repository when database is unavailable."""

    def __init__(self, base_directory: str = "./belief_states_fallback"):
        """Initialize file-based repository.

        Args:
            base_directory: Directory to store belief state files
        """
        self.base_dir = Path(base_directory)
        self.base_dir.mkdir(exist_ok=True)
        logger.info(f"Initialized fallback file repository at {self.base_dir}")

    async def save_belief_state(self, belief_state: BeliefState, agent_id: str) -> None:
        """Save belief state to file."""
        try:
            agent_dir = self.base_dir / agent_id
            agent_dir.mkdir(exist_ok=True)

            timestamp_str = datetime.fromtimestamp(belief_state.timestamp).strftime(
                "%Y%m%d_%H%M%S_%f"
            )
            filename = agent_dir / f"belief_{timestamp_str}.json"

            data = {
                "timestamp": belief_state.timestamp,
                "beliefs": belief_state.beliefs.tolist(),
                "entropy": belief_state.entropy,
                "max_confidence": belief_state.max_confidence,
                "effective_states": belief_state.effective_states,
                "most_likely_state": belief_state.most_likely_state(),
                "observation_history": belief_state.observation_history,
            }

            with open(filename, "w") as f:
                json.dump(data, f)

            # Create current state symlink
            current_link = agent_dir / "current.json"
            if current_link.exists():
                current_link.unlink()
            current_link.symlink_to(filename.name)

            logger.debug(f"Saved belief state to file: {filename}")

        except Exception as e:
            logger.error(f"Failed to save belief state to file for agent {agent_id}: {e}")
            raise

    async def get_current_belief_state(self, agent_id: str) -> Optional[BeliefState]:
        """Get current belief state from file."""
        try:
            current_file = self.base_dir / agent_id / "current.json"
            if not current_file.exists():
                return None

            with open(current_file, "r") as f:
                data = json.load(f)

            return BeliefState(
                beliefs=np.array(data["beliefs"]),
                timestamp=data["timestamp"],
                observation_history=data["observation_history"],
            )

        except Exception as e:
            logger.error(f"Failed to get current belief state from file for agent {agent_id}: {e}")
            return None

    async def get_belief_history(
        self, agent_id: str, limit: Optional[int] = None
    ) -> List[BeliefState]:
        """Get belief history from files."""
        try:
            agent_dir = self.base_dir / agent_id
            if not agent_dir.exists():
                return []

            # Get all belief files
            belief_files = sorted(
                agent_dir.glob("belief_*.json"), key=lambda f: f.stat().st_mtime, reverse=True
            )

            if limit:
                belief_files = belief_files[:limit]

            belief_states = []
            for file_path in belief_files:
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)

                    belief_state = BeliefState(
                        beliefs=np.array(data["beliefs"]),
                        timestamp=data["timestamp"],
                        observation_history=data["observation_history"],
                    )
                    belief_states.append(belief_state)

                except Exception as e:
                    logger.warning(f"Failed to load belief state from {file_path}: {e}")

            return belief_states

        except Exception as e:
            logger.error(f"Failed to get belief history from files for agent {agent_id}: {e}")
            return []

    async def clear_history(self, agent_id: str) -> None:
        """Clear belief history files for agent."""
        try:
            agent_dir = self.base_dir / agent_id
            if agent_dir.exists():
                for file_path in agent_dir.iterdir():
                    file_path.unlink()
                agent_dir.rmdir()

                logger.debug(f"Cleared belief history files for agent {agent_id}")

        except Exception as e:
            logger.error(f"Failed to clear belief history files for agent {agent_id}: {e}")
            raise


def create_belief_repository(
    config: Optional[BeliefPersistenceConfig] = None, fallback_to_file: bool = True
) -> BeliefStateRepository:
    """Factory function to create appropriate belief repository.

    Args:
        config: Persistence configuration
        fallback_to_file: Whether to fall back to file repository on database failure

    Returns:
        Configured belief state repository
    """
    config = config or BeliefPersistenceConfig()

    try:
        # Try database repository first
        return DatabaseBeliefRepository(config)

    except Exception as e:
        logger.warning(f"Failed to create database repository: {e}")

        if fallback_to_file:
            logger.info("Falling back to file-based repository")
            return FallbackFileRepository()
        else:
            raise
