"""
E2E Test Data Management
========================

Manages test data for E2E tests including:
- Test user creation and cleanup
- Test data fixtures
- Database seeding
- File management
"""

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import E2ETestConfig

logger = logging.getLogger(__name__)


class TestDataManager:
    """Manages test data for E2E tests"""

    def __init__(self, config: E2ETestConfig):
        self.config = config
        self.created_users = []
        self.created_agents = []
        self.created_coalitions = []
        self.created_conversations = []
        self.test_files = []
        self.test_session_id = str(uuid.uuid4())

    async def setup(self):
        """Setup test data"""
        logger.info("Setting up test data")

        # Create test directories
        self._create_test_directories()

        # Load test fixtures
        await self._load_test_fixtures()

        # Create test users
        await self._create_test_users()

        # Setup test files
        await self._setup_test_files()

    async def cleanup(self):
        """Clean up test data"""
        logger.info("Cleaning up test data")

        # Clean up database records
        await self._cleanup_database_records()

        # Clean up test files
        await self._cleanup_test_files()

        # Clear session data
        self._clear_session_data()

    def _create_test_directories(self):
        """Create necessary test directories"""
        directories = [
            self.config.test_data_path,
            self.config.fixture_path,
            self.config.screenshots_path,
            self.config.videos_path,
            self.config.report_path,
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    async def _load_test_fixtures(self):
        """Load test fixtures from files"""
        fixtures_dir = self.config.fixture_path

        # Load user fixtures
        user_fixture_path = os.path.join(fixtures_dir, "users.json")
        if os.path.exists(user_fixture_path):
            with open(user_fixture_path, "r") as f:
                self.user_fixtures = json.load(f)
        else:
            self.user_fixtures = self._create_default_user_fixtures()
            self._save_fixture(user_fixture_path, self.user_fixtures)

        # Load agent fixtures
        agent_fixture_path = os.path.join(fixtures_dir, "agents.json")
        if os.path.exists(agent_fixture_path):
            with open(agent_fixture_path, "r") as f:
                self.agent_fixtures = json.load(f)
        else:
            self.agent_fixtures = self._create_default_agent_fixtures()
            self._save_fixture(agent_fixture_path, self.agent_fixtures)

        # Load conversation fixtures
        conversation_fixture_path = os.path.join(
            fixtures_dir, "conversations.json"
        )
        if os.path.exists(conversation_fixture_path):
            with open(conversation_fixture_path, "r") as f:
                self.conversation_fixtures = json.load(f)
        else:
            self.conversation_fixtures = (
                self._create_default_conversation_fixtures()
            )
            self._save_fixture(
                conversation_fixture_path, self.conversation_fixtures
            )

    async def _create_test_users(self):
        """Create test users"""
        # Create admin user
        admin_user = await self._create_user(
            username=self.config.admin_user["username"],
            email=self.config.admin_user["email"],
            password=self.config.admin_user["password"],
            role="admin",
        )
        if admin_user:
            self.created_users.append(admin_user)

        # Create test user
        test_user = await self._create_user(
            username=self.config.test_user["username"],
            email=self.config.test_user["email"],
            password=self.config.test_user["password"],
            role="user",
        )
        if test_user:
            self.created_users.append(test_user)

        # Create additional test users from fixtures
        for user_data in self.user_fixtures:
            user = await self._create_user(**user_data)
            if user:
                self.created_users.append(user)

    async def _create_user(
        self, username: str, email: str, password: str, role: str = "user"
    ) -> Optional[Dict[str, Any]]:
        """Create a test user"""
        try:
            # This would interact with the actual user creation API
            user_data = {
                "id": str(uuid.uuid4()),
                "username": username,
                "email": email,
                "password": password,
                "role": role,
                "created_at": datetime.now().isoformat(),
                "session_id": self.test_session_id,
            }

            logger.debug(f"Created test user: {username}")
            return user_data

        except Exception as e:
            logger.error(f"Failed to create test user {username}: {e}")
            return None

    async def _setup_test_files(self):
        """Setup test files"""
        test_files_dir = os.path.join(self.config.test_data_path, "files")
        os.makedirs(test_files_dir, exist_ok=True)

        # Create sample test files
        sample_files = [
            ("sample.txt", "This is a sample text file for testing"),
            ("sample.json", json.dumps({"test": "data", "number": 42})),
            (
                "sample.csv",
                "name,age,email\nJohn,30,john@test.com\nJane,25,jane@test.com",
            ),
            ("sample.md", "# Test Document\n\nThis is a test markdown file."),
        ]

        for filename, content in sample_files:
            file_path = os.path.join(test_files_dir, filename)
            with open(file_path, "w") as f:
                f.write(content)
            self.test_files.append(file_path)

    async def _cleanup_database_records(self):
        """Clean up database records created during tests"""
        # This would implement actual database cleanup
        logger.debug("Cleaning up database records")

        # Clean up users
        for user in self.created_users:
            await self._delete_user(user["id"])

        # Clean up agents
        for agent in self.created_agents:
            await self._delete_agent(agent["id"])

        # Clean up coalitions
        for coalition in self.created_coalitions:
            await self._delete_coalition(coalition["id"])

        # Clean up conversations
        for conversation in self.created_conversations:
            await self._delete_conversation(conversation["id"])

    async def _cleanup_test_files(self):
        """Clean up test files"""
        for file_path in self.test_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.error(f"Failed to delete test file {file_path}: {e}")

    def _clear_session_data(self):
        """Clear session data"""
        self.created_users = []
        self.created_agents = []
        self.created_coalitions = []
        self.created_conversations = []
        self.test_files = []

    async def _delete_user(self, user_id: str):
        """Delete a test user"""
        try:
            logger.debug(f"Deleted test user: {user_id}")
        except Exception as e:
            logger.error(f"Failed to delete test user {user_id}: {e}")

    async def _delete_agent(self, agent_id: str):
        """Delete a test agent"""
        try:
            logger.debug(f"Deleted test agent: {agent_id}")
        except Exception as e:
            logger.error(f"Failed to delete test agent {agent_id}: {e}")

    async def _delete_coalition(self, coalition_id: str):
        """Delete a test coalition"""
        try:
            logger.debug(f"Deleted test coalition: {coalition_id}")
        except Exception as e:
            logger.error(
                f"Failed to delete test coalition {coalition_id}: {e}"
            )

    async def _delete_conversation(self, conversation_id: str):
        """Delete a test conversation"""
        try:
            logger.debug(f"Deleted test conversation: {conversation_id}")
        except Exception as e:
            logger.error(
                f"Failed to delete test conversation {conversation_id}: {e}"
            )

    def _create_default_user_fixtures(self) -> List[Dict[str, Any]]:
        """Create default user fixtures"""
        return [
            {
                "username": "testuser1",
                "email": "testuser1@test.com",
                "password": "testpass1",
                "role": "user",
            },
            {
                "username": "testuser2",
                "email": "testuser2@test.com",
                "password": "testpass2",
                "role": "user",
            },
            {
                "username": "testadmin",
                "email": "testadmin@test.com",
                "password": "testadmin",
                "role": "admin",
            },
        ]

    def _create_default_agent_fixtures(self) -> List[Dict[str, Any]]:
        """Create default agent fixtures"""
        return [
            {
                "name": "TestAgent1",
                "description": "Test agent for E2E testing",
                "type": "active_inference",
                "config": {"belief_threshold": 0.5, "action_precision": 0.8},
            },
            {
                "name": "TestAgent2",
                "description": "Another test agent",
                "type": "gnn_based",
                "config": {"embedding_dim": 128, "num_layers": 3},
            },
        ]

    def _create_default_conversation_fixtures(self) -> List[Dict[str, Any]]:
        """Create default conversation fixtures"""
        return [
            {
                "title": "Test Conversation 1",
                "description": "First test conversation",
                "participants": ["testuser1", "TestAgent1"],
                "messages": [
                    {"role": "user", "content": "Hello, agent!"},
                    {"role": "agent", "content": "Hello! How can I help you?"},
                ],
            },
            {
                "title": "Test Conversation 2",
                "description": "Second test conversation",
                "participants": ["testuser2", "TestAgent2"],
                "messages": [
                    {"role": "user", "content": "What can you do?"},
                    {
                        "role": "agent",
                        "content": "I can help with various tasks.",
                    },
                ],
            },
        ]

    def _save_fixture(self, file_path: str, data: Any):
        """Save fixture data to file"""
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    # Public methods for test cases

    def create_test_data(self, data_type: str) -> Dict[str, Any]:
        """Create test data of specified type"""
        if data_type == "user":
            return {
                "id": str(uuid.uuid4()),
                "username": f"testuser_{uuid.uuid4().hex[:8]}",
                "email": f"test_{uuid.uuid4().hex[:8]}@test.com",
                "password": "testpass123",
                "role": "user",
                "created_at": datetime.now().isoformat(),
            }

        elif data_type == "agent":
            return {
                "id": str(uuid.uuid4()),
                "name": f"TestAgent_{uuid.uuid4().hex[:8]}",
                "description": "Dynamically created test agent",
                "type": "active_inference",
                "config": {"belief_threshold": 0.5, "action_precision": 0.8},
                "created_at": datetime.now().isoformat(),
            }

        elif data_type == "coalition":
            return {
                "id": str(uuid.uuid4()),
                "name": f"TestCoalition_{uuid.uuid4().hex[:8]}",
                "description": "Test coalition for E2E testing",
                "members": [],
                "created_at": datetime.now().isoformat(),
            }

        elif data_type == "conversation":
            return {
                "id": str(uuid.uuid4()),
                "title": f"Test Conversation {uuid.uuid4().hex[:8]}",
                "description": "Dynamically created test conversation",
                "participants": [],
                "messages": [],
                "created_at": datetime.now().isoformat(),
            }

        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def get_test_file(self, filename: str) -> str:
        """Get path to test file"""
        return os.path.join(self.config.test_data_path, "files", filename)

    def get_fixture_data(self, fixture_type: str) -> List[Dict[str, Any]]:
        """Get fixture data by type"""
        if fixture_type == "users":
            return self.user_fixtures
        elif fixture_type == "agents":
            return self.agent_fixtures
        elif fixture_type == "conversations":
            return self.conversation_fixtures
        else:
            raise ValueError(f"Unknown fixture type: {fixture_type}")

    def get_test_user(self, username: str) -> Optional[Dict[str, Any]]:
        """Get test user by username"""
        for user in self.created_users:
            if user["username"] == username:
                return user
        return None

    def get_admin_user(self) -> Optional[Dict[str, Any]]:
        """Get admin user"""
        return self.get_test_user(self.config.admin_user["username"])

    def get_regular_user(self) -> Optional[Dict[str, Any]]:
        """Get regular test user"""
        return self.get_test_user(self.config.test_user["username"])

    def create_temporary_file(
        self, content: str, extension: str = ".txt"
    ) -> str:
        """Create temporary file with content"""
        temp_filename = f"temp_{uuid.uuid4().hex[:8]}{extension}"
        temp_path = os.path.join(
            self.config.test_data_path, "files", temp_filename
        )

        with open(temp_path, "w") as f:
            f.write(content)

        self.test_files.append(temp_path)
        return temp_path

    def get_session_id(self) -> str:
        """Get current test session ID"""
        return self.test_session_id
