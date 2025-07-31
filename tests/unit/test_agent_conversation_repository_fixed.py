"""Unit tests for Agent Conversation Repository.

Tests the repository pattern implementation for agent conversations
following TDD principles as recommended by Kent Beck.
"""

import uuid
from unittest.mock import Mock

import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.v1.schemas.agent_conversation_schemas import (
    ConversationQueryParams,
    ConversationStatusEnum,
)
from database.models import AgentConversationMessage, AgentConversationSession, ConversationStatus
from database.repositories.agent_conversation_repository import (
    AgentConversationMessageRepository,
    AgentConversationRepository,
)


class TestAgentConversationRepository:
    """Unit tests for AgentConversationRepository."""

    @pytest.fixture
    def mock_db(self):
        """Mock database session."""
        return Mock(spec=Session)

    @pytest.fixture
    def repository(self, mock_db):
        """Repository instance with mocked database."""
        return AgentConversationRepository(mock_db)

    @pytest.fixture
    def message_repository(self, mock_db):
        """Message repository instance with mocked database."""
        return AgentConversationMessageRepository(mock_db)

    @pytest.fixture
    def sample_conversation_id(self):
        """Sample conversation ID for testing."""
        return uuid.uuid4()

    @pytest.fixture
    def sample_agent_id(self):
        """Sample agent ID for testing."""
        return uuid.uuid4()

    @pytest.mark.asyncio
    async def test_create_conversation_success(self, repository, mock_db):
        """Test successful conversation creation."""
        # Arrange
        prompt = "Test conversation prompt"
        title = "Test Conversation"
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()

        # Act
        result = await repository.create_conversation(prompt=prompt, title=title, max_turns=5)

        # Assert
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        mock_db.refresh.assert_called_once()
        assert result.prompt == prompt
        assert result.title == title
        assert result.status == ConversationStatus.PENDING

    @pytest.mark.asyncio
    async def test_create_conversation_database_error(self, repository, mock_db):
        """Test conversation creation with database error."""
        # Arrange
        prompt = "Test conversation prompt"
        mock_db.add = Mock()
        mock_db.commit = Mock(side_effect=SQLAlchemyError("Database error"))
        mock_db.rollback = Mock()

        # Act & Assert
        with pytest.raises(Exception, match="Failed to create conversation"):
            await repository.create_conversation(prompt=prompt)

        mock_db.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_conversation_by_id_exists(self, repository, mock_db, sample_conversation_id):
        """Test retrieving existing conversation by ID."""
        # Arrange
        expected_conversation = AgentConversationSession(
            id=sample_conversation_id, prompt="Test prompt", status=ConversationStatus.PENDING
        )

        mock_query = Mock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = expected_conversation

        # Act
        result = await repository.get_conversation_by_id(sample_conversation_id)

        # Assert
        mock_db.query.assert_called_once_with(AgentConversationSession)
        assert result == expected_conversation

    @pytest.mark.asyncio
    async def test_get_conversation_by_id_not_exists(
        self, repository, mock_db, sample_conversation_id
    ):
        """Test retrieving non-existent conversation by ID."""
        # Arrange
        mock_query = Mock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = None

        # Act
        result = await repository.get_conversation_by_id(sample_conversation_id)

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_get_conversations_with_filters(self, repository, mock_db):
        """Test retrieving conversations with filtering and pagination."""
        # Arrange
        user_id = "test_user_123"
        query_params = ConversationQueryParams(
            user_id=user_id, status=ConversationStatusEnum.ACTIVE, page=1, page_size=10
        )

        mock_conversations = [
            AgentConversationSession(
                id=uuid.uuid4(),
                prompt=f"Conversation {i}",
                user_id=user_id,
                status=ConversationStatus.ACTIVE,
            )
            for i in range(3)
        ]

        mock_query = Mock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 3
        mock_query.order_by.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = mock_conversations

        # Act
        conversations, total_count = await repository.get_conversations(query_params)

        # Assert
        assert len(conversations) == 3
        assert total_count == 3
        assert all(conv.user_id == user_id for conv in conversations)

    @pytest.mark.asyncio
    async def test_update_conversation_status_success(
        self, repository, mock_db, sample_conversation_id
    ):
        """Test successfully updating conversation status."""
        # Arrange
        conversation = AgentConversationSession(
            id=sample_conversation_id, prompt="Test", status=ConversationStatus.PENDING
        )

        mock_query = Mock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = conversation
        mock_db.commit = Mock()
        mock_db.refresh = Mock()

        # Act
        result = await repository.update_conversation_status(
            sample_conversation_id, ConversationStatusEnum.ACTIVE
        )

        # Assert
        assert result.status == ConversationStatus.ACTIVE
        assert result.started_at is not None
        mock_db.commit.assert_called_once()
        mock_db.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_agent_to_conversation_success(
        self, repository, mock_db, sample_conversation_id, sample_agent_id
    ):
        """Test successfully adding agent to conversation."""
        # Arrange
        conversation = AgentConversationSession(
            id=sample_conversation_id, prompt="Test", agent_count=0
        )

        # Mock existing agent check (returns None = no existing agent)
        mock_query = Mock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.first.side_effect = [
            None,
            conversation,
        ]  # First call: no existing, second: get conversation
        mock_query.count.return_value = 1  # New agent count

        mock_db.execute = Mock()
        mock_db.commit = Mock()

        # Act
        result = await repository.add_agent_to_conversation(
            sample_conversation_id, sample_agent_id, role="participant"
        )

        # Assert
        assert result is True
        mock_db.execute.assert_called_once()
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_agent_to_conversation_already_exists(
        self, repository, mock_db, sample_conversation_id, sample_agent_id
    ):
        """Test adding agent that's already in conversation."""
        # Arrange
        existing_association = Mock()  # Non-None return means agent already exists

        mock_query = Mock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = existing_association

        # Act
        result = await repository.add_agent_to_conversation(sample_conversation_id, sample_agent_id)

        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_conversation_success(self, repository, mock_db, sample_conversation_id):
        """Test successful conversation deletion."""
        # Arrange
        conversation = AgentConversationSession(id=sample_conversation_id, prompt="Test")

        mock_query = Mock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = conversation
        mock_db.delete = Mock()
        mock_db.commit = Mock()

        # Act
        result = await repository.delete_conversation(sample_conversation_id)

        # Assert
        assert result is True
        mock_db.delete.assert_called_once_with(conversation)
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_conversation_not_found(self, repository, mock_db, sample_conversation_id):
        """Test deleting non-existent conversation."""
        # Arrange
        mock_query = Mock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = None

        # Act
        result = await repository.delete_conversation(sample_conversation_id)

        # Assert
        assert result is False


class TestAgentConversationMessageRepository:
    """Unit tests for AgentConversationMessageRepository."""

    @pytest.fixture
    def mock_db(self):
        """Mock database session."""
        return Mock(spec=Session)

    @pytest.fixture
    def message_repository(self, mock_db):
        """Message repository instance with mocked database."""
        return AgentConversationMessageRepository(mock_db)

    @pytest.fixture
    def sample_conversation_id(self):
        """Sample conversation ID for testing."""
        return uuid.uuid4()

    @pytest.fixture
    def sample_agent_id(self):
        """Sample agent ID for testing."""
        return uuid.uuid4()

    @pytest.fixture
    def sample_message_id(self):
        """Sample message ID for testing."""
        return uuid.uuid4()

    @pytest.mark.asyncio
    async def test_create_message_success(
        self, message_repository, mock_db, sample_conversation_id, sample_agent_id
    ):
        """Test successful message creation."""
        # Arrange
        content = "Test message content"
        conversation = AgentConversationSession(
            id=sample_conversation_id, current_turn=0, message_count=0
        )

        mock_query = Mock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.scalar.return_value = 0  # No existing messages
        mock_query.first.return_value = conversation

        mock_db.add = Mock()
        mock_db.execute = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()

        # Act
        result = await message_repository.create_message(
            conversation_id=sample_conversation_id, agent_id=sample_agent_id, content=content
        )

        # Assert
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        mock_db.refresh.assert_called_once()
        assert result.content == content
        assert result.message_order == 1
        assert result.conversation_id == sample_conversation_id
        assert result.agent_id == sample_agent_id

    @pytest.mark.asyncio
    async def test_create_message_conversation_not_found(
        self, message_repository, mock_db, sample_conversation_id, sample_agent_id
    ):
        """Test message creation when conversation doesn't exist."""
        # Arrange
        mock_query = Mock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.scalar.return_value = 0
        mock_query.first.return_value = None  # Conversation not found

        # Act & Assert
        with pytest.raises(Exception, match="Conversation not found"):
            await message_repository.create_message(
                conversation_id=sample_conversation_id,
                agent_id=sample_agent_id,
                content="Test message",
            )

    @pytest.mark.asyncio
    async def test_get_message_by_id_exists(self, message_repository, mock_db, sample_message_id):
        """Test retrieving existing message by ID."""
        # Arrange
        expected_message = AgentConversationMessage(
            id=sample_message_id, content="Test message", message_order=1
        )

        mock_query = Mock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = expected_message

        # Act
        result = await message_repository.get_message_by_id(sample_message_id)

        # Assert
        assert result == expected_message

    @pytest.mark.asyncio
    async def test_get_message_by_id_not_exists(
        self, message_repository, mock_db, sample_message_id
    ):
        """Test retrieving non-existent message by ID."""
        # Arrange
        mock_query = Mock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = None

        # Act
        result = await message_repository.get_message_by_id(sample_message_id)

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_create_message_database_error(
        self, message_repository, mock_db, sample_conversation_id, sample_agent_id
    ):
        """Test message creation with database error."""
        # Arrange
        conversation = AgentConversationSession(
            id=sample_conversation_id,
            current_turn=0,
            message_count=0,  # Initialize message_count to avoid NoneType error
        )

        mock_query = Mock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.scalar.return_value = 0
        mock_query.first.return_value = conversation

        mock_db.add = Mock()
        mock_db.commit = Mock(side_effect=SQLAlchemyError("Database error"))
        mock_db.rollback = Mock()

        # Act & Assert
        with pytest.raises(Exception, match="Failed to create message"):
            await message_repository.create_message(
                conversation_id=sample_conversation_id,
                agent_id=sample_agent_id,
                content="Test message",
            )

        mock_db.rollback.assert_called_once()
