"""
Comprehensive tests for Offline Capabilities Management System
"""

import json
import os
import pickle
import sqlite3
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from infrastructure.hardware.offline_capabilities import (
    OfflineManager,
    StatePersistence,
    SyncManager,
    SyncPriority,
    WorkItem,
    WorkQueue,
)


class TestSyncPriority:
    """Test SyncPriority enum"""

    def test_priority_values(self):
        """Test priority enum values"""
        assert SyncPriority.CRITICAL.value == "critical"
        assert SyncPriority.HIGH.value == "high"
        assert SyncPriority.MEDIUM.value == "medium"
        assert SyncPriority.LOW.value == "low"

    def test_priority_from_string(self):
        """Test creating priority from string value"""
        assert SyncPriority("critical") == SyncPriority.CRITICAL
        assert SyncPriority("high") == SyncPriority.HIGH
        assert SyncPriority("medium") == SyncPriority.MEDIUM
        assert SyncPriority("low") == SyncPriority.LOW


class TestWorkItem:
    """Test WorkItem dataclass"""

    def test_default_work_item(self):
        """Test creating work item with defaults"""
        item = WorkItem(id="test-1", task_type="sync_data", data={"key": "value"})
        assert item.id == "test-1"
        assert item.task_type == "sync_data"
        assert item.data == {"key": "value"}
        assert item.priority == SyncPriority.MEDIUM
        assert isinstance(item.created_at, datetime)
        assert item.retry_count == 0
        assert item.max_retries == 3

    def test_custom_work_item(self):
        """Test creating work item with custom values"""
        created_time = datetime.now()
        item = WorkItem(
            id="test-2",
            task_type="critical_sync",
            data={"important": True},
            priority=SyncPriority.CRITICAL,
            created_at=created_time,
            retry_count=2,
            max_retries=5,
        )
        assert item.priority == SyncPriority.CRITICAL
        assert item.created_at == created_time
        assert item.retry_count == 2
        assert item.max_retries == 5


class TestStatePersistence:
    """Test StatePersistence class"""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield os.path.join(tmp_dir, "test_state.db")

    def test_initialization(self, temp_db_path):
        """Test state persistence initialization"""
        persistence = StatePersistence(temp_db_path)
        assert persistence.storage_path == Path(temp_db_path)
        assert persistence.connection is not None
        assert Path(temp_db_path).exists()

    def test_database_tables_created(self, temp_db_path):
        """Test that required tables are created"""
        persistence = StatePersistence(temp_db_path)

        # Check tables exist
        cursor = persistence.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        assert "agent_states" in tables
        assert "work_queue" in tables

    def test_save_and_load_agent_state(self, temp_db_path):
        """Test saving and loading agent state"""
        persistence = StatePersistence(temp_db_path)

        # Save state
        agent_id = "agent-123"
        state = {
            "position": {"x": 10, "y": 20},
            "status": "active",
            "knowledge": ["fact1", "fact2"],
        }

        result = persistence.save_agent_state(agent_id, state)
        assert result is True

        # Load state
        loaded_state = persistence.load_agent_state(agent_id)
        assert loaded_state == state

    def test_save_agent_state_update(self, temp_db_path):
        """Test updating existing agent state"""
        persistence = StatePersistence(temp_db_path)
        agent_id = "agent-456"

        # Save initial state
        state1 = {"version": 1, "status": "idle"}
        persistence.save_agent_state(agent_id, state1)

        # Update state
        state2 = {"version": 2, "status": "active"}
        persistence.save_agent_state(agent_id, state2)

        # Load should return updated state
        loaded_state = persistence.load_agent_state(agent_id)
        assert loaded_state == state2

    def test_load_nonexistent_agent_state(self, temp_db_path):
        """Test loading state for non-existent agent"""
        persistence = StatePersistence(temp_db_path)
        result = persistence.load_agent_state("nonexistent")
        assert result is None

    def test_save_agent_state_no_connection(self, temp_db_path):
        """Test saving state when connection is closed"""
        persistence = StatePersistence(temp_db_path)
        persistence.connection.close()
        persistence.connection = None

        result = persistence.save_agent_state("agent-789", {"data": "test"})
        assert result is False

    def test_load_agent_state_no_connection(self, temp_db_path):
        """Test loading state when connection is closed"""
        persistence = StatePersistence(temp_db_path)
        persistence.connection.close()
        persistence.connection = None

        result = persistence.load_agent_state("agent-789")
        assert result is None

    def test_cleanup_old_states(self, temp_db_path):
        """Test cleaning up old agent states"""
        persistence = StatePersistence(temp_db_path)

        # Create states with different ages
        now = datetime.now()
        old_date = now - timedelta(days=35)
        recent_date = now - timedelta(days=5)

        # Insert old state
        cursor = persistence.connection.cursor()
        cursor.execute(
            """
            INSERT INTO agent_states (agent_id, state_data, last_updated, version)
            VALUES (?, ?, ?, ?)
        """,
            ("old-agent", pickle.dumps({"old": True}), old_date, 1),
        )

        # Insert recent state
        cursor.execute(
            """
            INSERT INTO agent_states (agent_id, state_data, last_updated, version)
            VALUES (?, ?, ?, ?)
        """,
            ("recent-agent", pickle.dumps({"recent": True}), recent_date, 1),
        )
        persistence.connection.commit()

        # Cleanup old states (30 days)
        deleted_count = persistence.cleanup_old_states(days_old=30)
        assert deleted_count == 1

        # Verify old state is gone
        assert persistence.load_agent_state("old-agent") is None
        # Verify recent state remains
        assert persistence.load_agent_state("recent-agent") is not None

    def test_cleanup_old_states_no_connection(self, temp_db_path):
        """Test cleanup when connection is closed"""
        persistence = StatePersistence(temp_db_path)
        persistence.connection.close()
        persistence.connection = None

        result = persistence.cleanup_old_states()
        assert result == 0

    @patch("infrastructure.hardware.offline_capabilities.logger")
    def test_init_database_error(self, mock_logger):
        """Test database initialization error handling"""
        with patch("sqlite3.connect", side_effect=Exception("DB Error")):
            with pytest.raises(Exception):
                StatePersistence("/invalid/path/test.db")
            mock_logger.error.assert_called()

    @patch("infrastructure.hardware.offline_capabilities.logger")
    def test_save_agent_state_error(self, mock_logger, temp_db_path):
        """Test error handling in save_agent_state"""
        persistence = StatePersistence(temp_db_path)

        # Make pickle.dumps fail
        with patch("pickle.dumps", side_effect=Exception("Pickle Error")):
            result = persistence.save_agent_state("agent-error", {"data": "test"})
            assert result is False
            mock_logger.error.assert_called()

    @patch("infrastructure.hardware.offline_capabilities.logger")
    def test_load_agent_state_error(self, mock_logger, temp_db_path):
        """Test error handling in load_agent_state"""
        persistence = StatePersistence(temp_db_path)

        # Save valid state first
        persistence.save_agent_state("agent-corrupt", {"data": "test"})

        # Make pickle.loads fail
        with patch("pickle.loads", side_effect=Exception("Unpickle Error")):
            result = persistence.load_agent_state("agent-corrupt")
            assert result is None
            mock_logger.error.assert_called()


class TestWorkQueue:
    """Test WorkQueue class"""

    @pytest.fixture
    def persistence_mock(self):
        """Create mock persistence object"""
        mock = Mock(spec=StatePersistence)
        mock.connection = Mock(spec=sqlite3.Connection)
        return mock

    def test_initialization(self, persistence_mock):
        """Test work queue initialization"""
        # Mock empty persisted items
        cursor_mock = Mock()
        cursor_mock.fetchall.return_value = []
        persistence_mock.connection.cursor.return_value = cursor_mock

        queue = WorkQueue(persistence_mock)
        assert queue.persistence == persistence_mock
        assert queue.memory_queue.empty()
        assert not queue.is_processing

    def test_load_persisted_items(self, persistence_mock):
        """Test loading items from persistence"""
        # Mock persisted items
        cursor_mock = Mock()
        created_time = datetime.now()
        cursor_mock.fetchall.return_value = [
            (
                "item-1",
                "sync_data",
                pickle.dumps({"key": "value"}),
                "high",
                created_time.isoformat(),
                1,
            ),
            (
                "item-2",
                "update_state",
                pickle.dumps({"status": "active"}),
                "medium",
                created_time.isoformat(),
                0,
            ),
        ]
        persistence_mock.connection.cursor.return_value = cursor_mock

        queue = WorkQueue(persistence_mock)

        # Check items were loaded
        assert queue.size() == 2

        # Verify items were deleted from persistence
        cursor_mock.execute.assert_any_call("DELETE FROM work_queue")
        persistence_mock.connection.commit.assert_called()

    def test_add_work_item(self, persistence_mock):
        """Test adding work item to queue"""
        cursor_mock = Mock()
        persistence_mock.connection.cursor.return_value = cursor_mock

        queue = WorkQueue(persistence_mock)

        item = WorkItem(
            id="test-item", task_type="sync_data", data={"test": "data"}, priority=SyncPriority.HIGH
        )

        result = queue.add_work_item(item)
        assert result is True
        assert queue.size() == 1

        # Verify persistence
        cursor_mock.execute.assert_called()
        persistence_mock.connection.commit.assert_called()

    def test_add_work_item_no_persistence(self):
        """Test adding work item when persistence is not available"""
        persistence = Mock(spec=StatePersistence)
        persistence.connection = None

        queue = WorkQueue(persistence)
        item = WorkItem(id="test", task_type="sync", data={})

        result = queue.add_work_item(item)
        assert result is True
        assert queue.size() == 1

    def test_get_next_item(self, persistence_mock):
        """Test getting next item from queue"""
        cursor_mock = Mock()
        cursor_mock.fetchall.return_value = []
        persistence_mock.connection.cursor.return_value = cursor_mock

        queue = WorkQueue(persistence_mock)

        # Add item
        item = WorkItem(id="test", task_type="sync", data={})
        queue.add_work_item(item)

        # Get item
        retrieved = queue.get_next_item()
        assert retrieved == item
        assert queue.size() == 0

    def test_get_next_item_empty_queue(self, persistence_mock):
        """Test getting item from empty queue"""
        cursor_mock = Mock()
        cursor_mock.fetchall.return_value = []
        persistence_mock.connection.cursor.return_value = cursor_mock

        queue = WorkQueue(persistence_mock)

        result = queue.get_next_item()
        assert result is None

    def test_mark_completed(self, persistence_mock):
        """Test marking item as completed"""
        cursor_mock = Mock()
        persistence_mock.connection.cursor.return_value = cursor_mock

        queue = WorkQueue(persistence_mock)

        result = queue.mark_completed("item-123")
        assert result is True

        cursor_mock.execute.assert_called_with("DELETE FROM work_queue WHERE id = ?", ("item-123",))
        persistence_mock.connection.commit.assert_called()

    def test_mark_completed_no_persistence(self):
        """Test marking completed when persistence is not available"""
        persistence = Mock(spec=StatePersistence)
        persistence.connection = None

        queue = WorkQueue(persistence)
        result = queue.mark_completed("item-123")
        assert result is True

    @patch("infrastructure.hardware.offline_capabilities.logger")
    def test_add_work_item_error(self, mock_logger, persistence_mock):
        """Test error handling when adding work item"""
        queue = WorkQueue(persistence_mock)

        # Make queue.put fail
        queue.memory_queue.put = Mock(side_effect=Exception("Queue Error"))

        item = WorkItem(id="test", task_type="sync", data={})
        result = queue.add_work_item(item)

        assert result is False
        mock_logger.error.assert_called()

    @patch("infrastructure.hardware.offline_capabilities.logger")
    def test_load_persisted_items_error(self, mock_logger):
        """Test error handling when loading persisted items"""
        persistence = Mock(spec=StatePersistence)
        persistence.connection = Mock()
        persistence.connection.cursor.side_effect = Exception("DB Error")

        queue = WorkQueue(persistence)
        mock_logger.error.assert_called()


class TestSyncManager:
    """Test SyncManager class"""

    @pytest.fixture
    def work_queue_mock(self):
        """Create mock work queue"""
        return Mock(spec=WorkQueue)

    def test_initialization(self, work_queue_mock):
        """Test sync manager initialization"""
        manager = SyncManager(work_queue_mock)
        assert manager.work_queue == work_queue_mock
        assert not manager.is_online
        assert not manager.sync_running
        assert manager.sync_thread is None
        assert manager.callbacks == {}

    def test_register_sync_callback(self, work_queue_mock):
        """Test registering sync callbacks"""
        manager = SyncManager(work_queue_mock)

        callback = Mock()
        manager.register_sync_callback("sync_data", callback)

        assert manager.callbacks["sync_data"] == callback

    def test_set_online_status_to_online(self, work_queue_mock):
        """Test setting online status from offline to online"""
        manager = SyncManager(work_queue_mock)

        # Start offline
        manager.is_online = False

        with patch.object(manager, "start_sync") as mock_start:
            manager.set_online_status(True)

            assert manager.is_online is True
            mock_start.assert_called_once()

    def test_set_online_status_to_offline(self, work_queue_mock):
        """Test setting online status from online to offline"""
        manager = SyncManager(work_queue_mock)

        # Start online
        manager.is_online = True

        with patch.object(manager, "start_sync") as mock_start:
            manager.set_online_status(False)

            assert manager.is_online is False
            mock_start.assert_not_called()

    def test_set_online_status_no_change(self, work_queue_mock):
        """Test setting online status with no change"""
        manager = SyncManager(work_queue_mock)
        manager.is_online = True

        with patch.object(manager, "start_sync") as mock_start:
            manager.set_online_status(True)
            mock_start.assert_not_called()

    def test_start_sync(self, work_queue_mock):
        """Test starting sync process"""
        manager = SyncManager(work_queue_mock)

        # Mock the work queue to return None immediately
        work_queue_mock.get_next_item.return_value = None

        # Set manager online so thread will run
        manager.is_online = True

        result = manager.start_sync()
        assert result is True
        assert manager.sync_thread is not None

        # Give thread a moment to start
        time.sleep(0.1)
        assert manager.sync_thread.is_alive()

        # Stop the sync
        manager.stop_sync()

        # Wait for thread to finish
        manager.sync_thread.join(timeout=2)

    def test_start_sync_already_running(self, work_queue_mock):
        """Test starting sync when already running"""
        manager = SyncManager(work_queue_mock)
        manager.sync_running = True

        result = manager.start_sync()
        assert result is False

    def test_stop_sync(self, work_queue_mock):
        """Test stopping sync process"""
        manager = SyncManager(work_queue_mock)

        # Start sync first
        manager.start_sync()
        time.sleep(0.1)  # Let thread start

        manager.stop_sync()
        assert manager.sync_running is False

    def test_stop_sync_not_running(self, work_queue_mock):
        """Test stopping sync when not running"""
        manager = SyncManager(work_queue_mock)

        # Should not raise exception
        manager.stop_sync()
        assert manager.sync_running is False

    def test_sync_worker_processes_items(self, work_queue_mock):
        """Test sync worker processing items"""
        manager = SyncManager(work_queue_mock)

        # Setup test data
        test_item = WorkItem(id="test-1", task_type="sync_data", data={"test": "data"})

        # Mock work queue behavior
        work_queue_mock.get_next_item.side_effect = [
            test_item,  # First call returns item
            None,  # Second call returns None to exit
        ]

        # Register callback
        callback = Mock(return_value=True)
        manager.register_sync_callback("sync_data", callback)

        # Run sync briefly
        manager.is_online = True
        manager.start_sync()
        time.sleep(0.2)
        manager.sync_running = False
        manager.sync_thread.join(timeout=2)

        # Verify callback was called
        callback.assert_called_once_with(test_item)
        work_queue_mock.mark_completed.assert_called_once_with("test-1")

    def test_sync_worker_retry_on_failure(self, work_queue_mock):
        """Test sync worker retries on failure"""
        manager = SyncManager(work_queue_mock)

        test_item = WorkItem(
            id="test-retry",
            task_type="sync_data",
            data={"test": "data"},
            retry_count=0,
            max_retries=3,
        )

        work_queue_mock.get_next_item.side_effect = [test_item, None]

        # Callback fails
        callback = Mock(return_value=False)
        manager.register_sync_callback("sync_data", callback)

        # Run sync
        manager.is_online = True
        manager.start_sync()
        time.sleep(0.2)
        manager.sync_running = False

        # Verify item was re-queued
        work_queue_mock.add_work_item.assert_called_once()
        re_queued_item = work_queue_mock.add_work_item.call_args[0][0]
        assert re_queued_item.retry_count == 1

    def test_sync_worker_max_retries_exceeded(self, work_queue_mock):
        """Test sync worker doesn't retry when max retries exceeded"""
        manager = SyncManager(work_queue_mock)

        test_item = WorkItem(
            id="test-max-retry",
            task_type="sync_data",
            data={"test": "data"},
            retry_count=3,
            max_retries=3,
        )

        work_queue_mock.get_next_item.side_effect = [test_item, None]

        callback = Mock(return_value=False)
        manager.register_sync_callback("sync_data", callback)

        # Run sync
        manager.is_online = True
        manager.start_sync()
        time.sleep(0.2)
        manager.sync_running = False

        # Verify item was NOT re-queued
        work_queue_mock.add_work_item.assert_not_called()

    def test_sync_worker_exception_handling(self, work_queue_mock):
        """Test sync worker handles exceptions"""
        manager = SyncManager(work_queue_mock)

        test_item = WorkItem(id="test-exception", task_type="sync_data", data={"test": "data"})

        work_queue_mock.get_next_item.side_effect = [test_item, None]

        # Callback raises exception
        callback = Mock(side_effect=Exception("Sync Error"))
        manager.register_sync_callback("sync_data", callback)

        # Run sync
        manager.is_online = True
        manager.start_sync()
        time.sleep(0.2)
        manager.sync_running = False

        # Verify item was re-queued with incremented retry
        work_queue_mock.add_work_item.assert_called_once()

    def test_sync_worker_no_callback(self, work_queue_mock):
        """Test sync worker skips items without callbacks"""
        manager = SyncManager(work_queue_mock)

        test_item = WorkItem(id="test-no-callback", task_type="unknown_type", data={"test": "data"})

        work_queue_mock.get_next_item.side_effect = [test_item, None]

        # No callback registered for this type

        # Run sync
        manager.is_online = True
        manager.start_sync()
        time.sleep(0.2)
        manager.sync_running = False

        # Verify item was not marked completed or re-queued
        work_queue_mock.mark_completed.assert_not_called()
        work_queue_mock.add_work_item.assert_not_called()

    @patch("infrastructure.hardware.offline_capabilities.logger")
    def test_start_sync_error(self, mock_logger, work_queue_mock):
        """Test error handling in start_sync"""
        manager = SyncManager(work_queue_mock)

        with patch("threading.Thread", side_effect=Exception("Thread Error")):
            result = manager.start_sync()
            assert result is False
            assert manager.sync_running is False
            mock_logger.error.assert_called()


class TestOfflineManager:
    """Test OfflineManager class"""

    @pytest.fixture
    def temp_offline_dir(self):
        """Create temporary offline directory"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir

    def test_initialization(self, temp_offline_dir):
        """Test offline manager initialization"""
        manager = OfflineManager(temp_offline_dir)

        assert manager.storage_path == Path(temp_offline_dir)
        assert manager.storage_path.exists()
        assert isinstance(manager.persistence, StatePersistence)
        assert isinstance(manager.work_queue, WorkQueue)
        assert isinstance(manager.sync_manager, SyncManager)
        assert not manager.offline_mode
        assert manager.cached_data == {}

    def test_enable_offline_mode(self, temp_offline_dir):
        """Test enabling offline mode"""
        manager = OfflineManager(temp_offline_dir)

        with patch.object(manager.sync_manager, "set_online_status") as mock_set_status:
            manager.enable_offline_mode()

            assert manager.offline_mode is True
            mock_set_status.assert_called_once_with(False)

    def test_disable_offline_mode(self, temp_offline_dir):
        """Test disabling offline mode"""
        manager = OfflineManager(temp_offline_dir)
        manager.offline_mode = True

        with patch.object(manager.sync_manager, "set_online_status") as mock_set_status:
            manager.disable_offline_mode()

            assert manager.offline_mode is False
            mock_set_status.assert_called_once_with(True)

    def test_cache_data(self, temp_offline_dir):
        """Test caching data"""
        manager = OfflineManager(temp_offline_dir)

        key = "test_key"
        data = {"important": "data", "numbers": [1, 2, 3]}
        ttl = 3600

        result = manager.cache_data(key, data, ttl)
        assert result is True

        # Check memory cache
        assert key in manager.cached_data
        assert manager.cached_data[key]["data"] == data
        assert manager.cached_data[key]["ttl_seconds"] == ttl

        # Check disk cache
        cache_file = manager.storage_path / f"cache_{key}.json"
        assert cache_file.exists()

        with open(cache_file, "r") as f:
            disk_data = json.load(f)
            assert disk_data["data"] == data
            assert disk_data["ttl_seconds"] == ttl

    def test_get_cached_data_from_memory(self, temp_offline_dir):
        """Test retrieving data from memory cache"""
        manager = OfflineManager(temp_offline_dir)

        # Cache data
        key = "memory_test"
        data = {"cached": True}
        manager.cache_data(key, data, 3600)

        # Retrieve immediately
        retrieved = manager.get_cached_data(key)
        assert retrieved == data

    def test_get_cached_data_expired(self, temp_offline_dir):
        """Test retrieving expired cached data"""
        manager = OfflineManager(temp_offline_dir)

        key = "expired_test"
        data = {"old": "data"}

        # Manually add expired entry to memory cache
        manager.cached_data[key] = {
            "data": data,
            "cached_at": datetime.now() - timedelta(seconds=7200),
            "ttl_seconds": 3600,
        }

        retrieved = manager.get_cached_data(key)
        assert retrieved is None
        assert key not in manager.cached_data

    def test_get_cached_data_from_disk(self, temp_offline_dir):
        """Test retrieving data from disk cache"""
        manager = OfflineManager(temp_offline_dir)

        key = "disk_test"
        data = {"from": "disk"}

        # Write directly to disk
        cache_file = manager.storage_path / f"cache_{key}.json"
        with open(cache_file, "w") as f:
            json.dump(
                {"data": data, "cached_at": datetime.now().isoformat(), "ttl_seconds": 3600}, f
            )

        retrieved = manager.get_cached_data(key)
        assert retrieved == data

    def test_get_cached_data_disk_expired(self, temp_offline_dir):
        """Test retrieving expired data from disk"""
        manager = OfflineManager(temp_offline_dir)

        key = "disk_expired"
        data = {"old": "disk_data"}

        # Write expired data to disk
        cache_file = manager.storage_path / f"cache_{key}.json"
        with open(cache_file, "w") as f:
            json.dump(
                {
                    "data": data,
                    "cached_at": (datetime.now() - timedelta(hours=2)).isoformat(),
                    "ttl_seconds": 3600,
                },
                f,
            )

        retrieved = manager.get_cached_data(key)
        assert retrieved is None
        assert not cache_file.exists()  # Should be deleted

    def test_get_cached_data_not_found(self, temp_offline_dir):
        """Test retrieving non-existent cached data"""
        manager = OfflineManager(temp_offline_dir)

        retrieved = manager.get_cached_data("nonexistent")
        assert retrieved is None

    def test_queue_for_sync(self, temp_offline_dir):
        """Test queuing task for sync"""
        manager = OfflineManager(temp_offline_dir)

        task_type = "update_agent"
        data = {"agent_id": "123", "status": "active"}
        priority = SyncPriority.HIGH

        with patch.object(manager.work_queue, "add_work_item", return_value=True) as mock_add:
            item_id = manager.queue_for_sync(task_type, data, priority)

            assert isinstance(item_id, str)
            assert task_type in item_id

            # Verify work item was created correctly
            mock_add.assert_called_once()
            work_item = mock_add.call_args[0][0]
            assert work_item.task_type == task_type
            assert work_item.data == data
            assert work_item.priority == priority

    def test_queue_for_sync_failure(self, temp_offline_dir):
        """Test queue for sync when add fails"""
        manager = OfflineManager(temp_offline_dir)

        with patch.object(manager.work_queue, "add_work_item", return_value=False):
            with pytest.raises(Exception, match="Failed to queue work item"):
                manager.queue_for_sync("test", {})

    def test_register_sync_handler(self, temp_offline_dir):
        """Test registering sync handler"""
        manager = OfflineManager(temp_offline_dir)

        handler = Mock()
        with patch.object(manager.sync_manager, "register_sync_callback") as mock_register:
            manager.register_sync_handler("custom_sync", handler)
            mock_register.assert_called_once_with("custom_sync", handler)

    def test_cleanup(self, temp_offline_dir):
        """Test cleanup of resources"""
        manager = OfflineManager(temp_offline_dir)

        # Store initial connection
        initial_connection = manager.persistence.connection

        with patch.object(manager.sync_manager, "stop_sync") as mock_stop_sync:
            manager.cleanup()

            mock_stop_sync.assert_called_once()
            # Connection should be closed
            if initial_connection:
                # SQLite connection should be closed but not None
                with pytest.raises(sqlite3.ProgrammingError):
                    initial_connection.execute("SELECT 1")

    @patch("infrastructure.hardware.offline_capabilities.logger")
    def test_cleanup_error(self, mock_logger, temp_offline_dir):
        """Test cleanup error handling"""
        manager = OfflineManager(temp_offline_dir)

        with patch.object(
            manager.sync_manager, "stop_sync", side_effect=Exception("Cleanup Error")
        ):
            manager.cleanup()
            mock_logger.error.assert_called()

    @patch("infrastructure.hardware.offline_capabilities.logger")
    def test_cache_data_error(self, mock_logger, temp_offline_dir):
        """Test error handling in cache_data"""
        manager = OfflineManager(temp_offline_dir)

        # Make JSON dump fail
        with patch("json.dump", side_effect=Exception("JSON Error")):
            result = manager.cache_data("test", {"data": "test"})
            assert result is False
            mock_logger.error.assert_called()

    @patch("infrastructure.hardware.offline_capabilities.logger")
    def test_get_cached_data_error(self, mock_logger, temp_offline_dir):
        """Test error handling in get_cached_data"""
        manager = OfflineManager(temp_offline_dir)

        # Add data to cache
        manager.cached_data["test"] = {
            "data": {"test": "data"},
            "cached_at": datetime.now(),
            "ttl_seconds": 3600,
        }

        # Make datetime operations fail
        with patch("infrastructure.hardware.offline_capabilities.datetime") as mock_dt:
            mock_dt.now.side_effect = Exception("DateTime Error")

            result = manager.get_cached_data("test")
            assert result is None
            mock_logger.error.assert_called()


class TestOfflineIntegration:
    """Integration tests for offline capabilities"""

    def test_full_offline_online_cycle(self):
        """Test complete offline/online cycle"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = OfflineManager(tmp_dir)

            # Register sync handler
            sync_results = []

            def sync_handler(item):
                sync_results.append(item)
                return True

            manager.register_sync_handler("agent_update", sync_handler)

            # Enable offline mode
            manager.enable_offline_mode()

            # Queue some work while offline
            item_id1 = manager.queue_for_sync(
                "agent_update", {"agent_id": "001", "status": "active"}, SyncPriority.HIGH
            )

            item_id2 = manager.queue_for_sync(
                "agent_update", {"agent_id": "002", "status": "idle"}, SyncPriority.MEDIUM
            )

            # Cache some data
            manager.cache_data("agent_001", {"position": {"x": 10, "y": 20}})

            # Go back online
            manager.disable_offline_mode()

            # Wait for sync to process
            time.sleep(0.5)

            # Verify sync happened
            assert len(sync_results) == 2
            assert sync_results[0].task_type == "agent_update"
            assert sync_results[1].task_type == "agent_update"

            # Verify cached data still accessible
            cached = manager.get_cached_data("agent_001")
            assert cached == {"position": {"x": 10, "y": 20}}

            # Cleanup
            manager.cleanup()

    def test_state_persistence_across_restarts(self):
        """Test that state persists across manager restarts"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # First session
            manager1 = OfflineManager(tmp_dir)

            # Save agent state
            manager1.persistence.save_agent_state(
                "agent-123", {"status": "active", "knowledge": ["fact1", "fact2"]}
            )

            # Queue work item
            manager1.queue_for_sync("sync_data", {"data": "important"})

            # Cache data
            manager1.cache_data("cached_key", {"cached": "value"})

            # Cleanup
            manager1.cleanup()

            # Second session
            manager2 = OfflineManager(tmp_dir)

            # Verify agent state persisted
            state = manager2.persistence.load_agent_state("agent-123")
            assert state == {"status": "active", "knowledge": ["fact1", "fact2"]}

            # Verify work queue has item
            assert manager2.work_queue.size() > 0

            # Verify cached data persisted
            cached = manager2.get_cached_data("cached_key")
            assert cached == {"cached": "value"}

            manager2.cleanup()

    def test_concurrent_operations(self):
        """Test concurrent offline operations"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = OfflineManager(tmp_dir)

            results = []
            errors = []
            lock = threading.Lock()

            def worker(worker_id):
                try:
                    # Each worker performs multiple operations
                    for i in range(5):
                        # Use lock for database operations to avoid SQLite threading issues
                        with lock:
                            # Save state
                            manager.persistence.save_agent_state(
                                f"agent-{worker_id}-{i}", {"worker": worker_id, "iteration": i}
                            )

                            # Queue work
                            manager.queue_for_sync("worker_task", {"worker": worker_id, "task": i})

                        # Cache data (can be done without lock)
                        manager.cache_data(
                            f"cache-{worker_id}-{i}", {"data": f"worker-{worker_id}-{i}"}
                        )

                    results.append(worker_id)
                except Exception as e:
                    errors.append(str(e))

            # Start multiple workers
            threads = []
            for i in range(5):
                thread = threading.Thread(target=worker, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for completion
            for thread in threads:
                thread.join()

            # Verify no errors
            if len(errors) > 0:
                print(f"Errors encountered: {errors}")
            assert len(errors) == 0
            assert len(results) == 5

            # Verify all operations succeeded
            # 5 workers * 5 iterations = 25 items each
            assert manager.work_queue.size() == 25

            # Spot check some data
            state = manager.persistence.load_agent_state("agent-2-3")
            assert state == {"worker": 2, "iteration": 3}

            cached = manager.get_cached_data("cache-4-2")
            assert cached == {"data": "worker-4-2"}

            manager.cleanup()

    def test_error_recovery(self):
        """Test recovery from various error conditions"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = OfflineManager(tmp_dir)

            # Simulate sync handler that fails first time
            attempt_count = 0

            def flaky_handler(item):
                nonlocal attempt_count
                attempt_count += 1
                if attempt_count == 1:
                    return False  # Fail first attempt
                return True  # Succeed on retry

            manager.register_sync_handler("flaky_task", flaky_handler)

            # Queue work
            manager.queue_for_sync("flaky_task", {"retry": "test"})

            # Go online to trigger sync
            manager.disable_offline_mode()

            # Wait for retries
            time.sleep(1)

            # Verify handler was called multiple times
            assert attempt_count >= 2

            manager.cleanup()
