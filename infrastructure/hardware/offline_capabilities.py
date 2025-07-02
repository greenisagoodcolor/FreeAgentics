"""
Offline Capabilities Management System

This module provides offline operation capabilities for agents and infrastructure
components when network connectivity is limited or unavailable.
"""

import json
import logging
import pickle
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from queue import Queue
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SyncPriority(Enum):
    """Priority levels for synchronization tasks"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class WorkItem:
    """Work item for offline processing queue"""
    id: str
    task_type: str
    data: Dict[str, Any]
    priority: SyncPriority = SyncPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    max_retries: int = 3


class StatePersistence:
    """Manages persistent state storage for offline operations"""
    def __init__(self, storage_path: str = "./offline_state.db"):
        self.storage_path = Path(storage_path)
        self.connection: Optional[sqlite3.Connection] = None
        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database for state storage"""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(str(self.storage_path), check_same_thread=False)
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS agent_states (
                    agent_id TEXT PRIMARY KEY,
                    state_data BLOB,
                    last_updated TIMESTAMP,
                    version INTEGER
                )
            """)
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS work_queue (
                    id TEXT PRIMARY KEY,
                    task_type TEXT,
                    data BLOB,
                    priority TEXT,
                    created_at TIMESTAMP,
                    retry_count INTEGER
                )
            """)
            self.connection.commit()
            logger.info(f"Initialized offline state database at {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to initialize state database: {e}")
            raise

    def save_agent_state(self, agent_id: str, state: Dict[str, Any]) -> bool:
        """Save agent state for offline access"""
        try:
            if not self.connection:
                return False

            state_blob = pickle.dumps(state)
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO agent_states
                (agent_id, state_data, last_updated, version)
                VALUES (?, ?, ?, COALESCE((SELECT version FROM agent_states WHERE agent_id = ?) + 1, 1))
            """, (agent_id, state_blob, datetime.now(), agent_id))
            self.connection.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save agent state for {agent_id}: {e}")
            return False

    def load_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Load agent state from offline storage"""
        try:
            if not self.connection:
                return None

            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT state_data FROM agent_states WHERE agent_id = ?",
                (agent_id,)
            )
            result = cursor.fetchone()
            if result:
                return pickle.loads(result[0])
            return None
        except Exception as e:
            logger.error(f"Failed to load agent state for {agent_id}: {e}")
            return None

    def cleanup_old_states(self, days_old: int = 30) -> int:
        """Remove old agent states to save storage space"""
        try:
            if not self.connection:
                return 0

            cutoff_date = datetime.now() - timedelta(days=days_old)
            cursor = self.connection.cursor()
            cursor.execute(
                "DELETE FROM agent_states WHERE last_updated < ?",
                (cutoff_date,)
            )
            self.connection.commit()
            return cursor.rowcount
        except Exception as e:
            logger.error(f"Failed to cleanup old states: {e}")
            return 0


class WorkQueue:
    """Queue for managing work items during offline operation"""

    def __init__(self, persistence: StatePersistence):
        self.persistence = persistence
        self.memory_queue: Queue = Queue()
        self.is_processing = False
        self._lock = threading.Lock()
        self._load_persisted_items()

    def _load_persisted_items(self) -> None:
        """Load work items from persistent storage"""
        try:
            if not self.persistence.connection:
                return

            cursor = self.persistence.connection.cursor()
            cursor.execute("SELECT id, task_type, data, priority, created_at, retry_count FROM work_queue")

            for row in cursor.fetchall():
                item = WorkItem(
                    id=row[0],
                    task_type=row[1],
                    data=pickle.loads(row[2]),
                    priority=SyncPriority(row[3]),
                    created_at=datetime.fromisoformat(row[4]),
                    retry_count=row[5]
                )
                self.memory_queue.put(item)

            # Clear persisted items after loading
            cursor.execute("DELETE FROM work_queue")
            self.persistence.connection.commit()

        except Exception as e:
            logger.error(f"Failed to load persisted work items: {e}")

    def add_work_item(self, item: WorkItem) -> bool:
        """Add work item to queue"""
        try:
            with self._lock:
                # Add to memory queue
                self.memory_queue.put(item)

                # Persist to database as backup
                if self.persistence.connection:
                    data_blob = pickle.dumps(item.data)
                    cursor = self.persistence.connection.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO work_queue
                        (id, task_type, data, priority, created_at, retry_count)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (item.id, item.task_type, data_blob, item.priority.value,
                          item.created_at.isoformat(), item.retry_count))
                    self.persistence.connection.commit()

                return True
        except Exception as e:
            logger.error(f"Failed to add work item {item.id}: {e}")
            return False

    def get_next_item(self) -> Optional[WorkItem]:
        """Get next work item from queue"""
        try:
            if not self.memory_queue.empty():
                return self.memory_queue.get_nowait()
            return None
        except Exception as e:
            logger.error(f"Failed to get next work item: {e}")
            return None

    def mark_completed(self, item_id: str) -> bool:
        """Mark work item as completed"""
        try:
            # Remove from persistent storage if exists
            if self.persistence.connection:
                cursor = self.persistence.connection.cursor()
                cursor.execute("DELETE FROM work_queue WHERE id = ?", (item_id,))
                self.persistence.connection.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to mark item {item_id} as completed: {e}")
            return False

    def size(self) -> int:
        """Get current queue size"""
        return self.memory_queue.qsize()


class SyncManager:
    """Manages synchronization between offline and online operations"""

    def __init__(self, work_queue: WorkQueue):
        self.work_queue = work_queue
        self.is_online = False
        self.sync_running = False
        self.sync_thread: Optional[threading.Thread] = None
        self.callbacks: Dict[str, callable] = {}

    def register_sync_callback(self, task_type: str, callback: callable) -> None:
        """Register callback for specific task type"""
        self.callbacks[task_type] = callback

    def set_online_status(self, is_online: bool) -> None:
        """Update online status and trigger sync if needed"""
        was_offline = not self.is_online
        self.is_online = is_online

        if is_online and was_offline and not self.sync_running:
            self.start_sync()

    def start_sync(self) -> bool:
        """Start synchronization process"""
        if self.sync_running:
            return False

        try:
            self.sync_running = True
            self.sync_thread = threading.Thread(target=self._sync_worker, daemon=True)
            self.sync_thread.start()
            logger.info("Started sync process")
            return True
        except Exception as e:
            logger.error(f"Failed to start sync: {e}")
            self.sync_running = False
            return False

    def _sync_worker(self) -> None:
        """Worker thread for processing sync queue"""
        try:
            while self.sync_running and self.is_online:
                item = self.work_queue.get_next_item()
                if not item:
                    time.sleep(1)
                    continue

                # Process work item
                callback = self.callbacks.get(item.task_type)
                if callback:
                    try:
                        success = callback(item)
                        if success:
                            self.work_queue.mark_completed(item.id)
                        else:
                            # Re-queue with incremented retry count
                            item.retry_count += 1
                            if item.retry_count <= item.max_retries:
                                self.work_queue.add_work_item(item)
                    except Exception as e:
                        logger.error(f"Error processing sync item {item.id}: {e}")
                        item.retry_count += 1
                        if item.retry_count <= item.max_retries:
                            self.work_queue.add_work_item(item)

        except Exception as e:
            logger.error(f"Sync worker error: {e}")
        finally:
            self.sync_running = False

    def stop_sync(self) -> None:
        """Stop synchronization process"""
        self.sync_running = False
        if self.sync_thread and self.sync_thread.is_alive():
            self.sync_thread.join(timeout=5)


class OfflineManager:
    """Main manager for offline capabilities"""

    def __init__(self, storage_path: str = "./offline_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.persistence = StatePersistence(str(self.storage_path / "state.db"))
        self.work_queue = WorkQueue(self.persistence)
        self.sync_manager = SyncManager(self.work_queue)

        # Offline capabilities state
        self.offline_mode = False
        self.cached_data: Dict[str, Any] = {}

        logger.info("Offline manager initialized")

    def enable_offline_mode(self) -> None:
        """Enable offline mode operation"""
        self.offline_mode = True
        self.sync_manager.set_online_status(False)
        logger.info("Offline mode enabled")

    def disable_offline_mode(self) -> None:
        """Disable offline mode and sync with online systems"""
        self.offline_mode = False
        self.sync_manager.set_online_status(True)
        logger.info("Offline mode disabled, starting sync")

    def cache_data(self, key: str, data: Any, ttl_seconds: int = 3600) -> bool:
        """Cache data for offline access"""
        try:
            cache_entry = {
                'data': data,
                'cached_at': datetime.now(),
                'ttl_seconds': ttl_seconds
            }
            self.cached_data[key] = cache_entry

            # Persist to disk
            cache_file = self.storage_path / f"cache_{key}.json"
            with open(cache_file, 'w') as f:
                json.dump({
                    'data': data,
                    'cached_at': cache_entry['cached_at'].isoformat(),
                    'ttl_seconds': ttl_seconds
                }, f)

            return True
        except Exception as e:
            logger.error(f"Failed to cache data for key {key}: {e}")
            return False

    def get_cached_data(self, key: str) -> Optional[Any]:
        """Retrieve cached data"""
        try:
            # Check memory cache first
            if key in self.cached_data:
                entry = self.cached_data[key]
                age = datetime.now() - entry['cached_at']
                if age.total_seconds() < entry['ttl_seconds']:
                    return entry['data']
                else:
                    del self.cached_data[key]

            # Check disk cache
            cache_file = self.storage_path / f"cache_{key}.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cache_entry = json.load(f)
                    cached_at = datetime.fromisoformat(cache_entry['cached_at'])
                    age = datetime.now() - cached_at
                    if age.total_seconds() < cache_entry['ttl_seconds']:
                        return cache_entry['data']
                    else:
                        cache_file.unlink()  # Remove expired cache

            return None
        except Exception as e:
            logger.error(f"Failed to get cached data for key {key}: {e}")
            return None

    def queue_for_sync(self, task_type: str, data: Dict[str, Any],
                       priority: SyncPriority = SyncPriority.MEDIUM) -> str:
        """Queue task for synchronization when back online"""
        work_item = WorkItem(
            id=f"{task_type}_{int(time.time())}_{id(data)}",
            task_type=task_type,
            data=data,
            priority=priority
        )

        if self.work_queue.add_work_item(work_item):
            return work_item.id
        else:
            raise Exception("Failed to queue work item")

    def register_sync_handler(self, task_type: str, handler: callable) -> None:
        """Register handler for sync task type"""
        self.sync_manager.register_sync_callback(task_type, handler)

    def cleanup(self) -> None:
        """Cleanup offline resources"""
        try:
            self.sync_manager.stop_sync()
            if self.persistence.connection:
                self.persistence.connection.close()
            logger.info("Offline manager cleanup completed")
        except Exception as e:
            logger.error(f"Error during offline manager cleanup: {e}")
