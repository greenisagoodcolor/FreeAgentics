"""Unit tests for Agent Conversation Repository.

Tests the repository pattern implementation for agent conversations
following TDD principles as recommended by Kent Beck.
"""

import uuid
from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from sqlalchemy.orm import Session

from database.models import Agent, AgentConversationSession, AgentConversationMessage, ConversationStatus
from database.repositories.agent_conversation_repository import AgentConversationRepository


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
    def sample_agent(self):
        """Sample agent for testing."""
        return Agent(
            id=uuid.uuid4(),
            name="Test Agent",
            template="test_template",
            gmn_spec='{"test": "spec"}',
            beliefs={"initial": "belief"},
            preferences={"test": "preference"}
        )

    @pytest.fixture
    def sample_conversation(self):
        """Sample conversation for testing."""
        return AgentConversationSession(
            id=uuid.uuid4(),
            prompt="Test conversation prompt",
            title="Test Conversation",
            status=ConversationStatus.PENDING,
            message_count=0,
            agent_count=1,
            config={"test": "config"}
        )

    @pytest.mark.asyncio
    async def test_create_conversation_success(self, repository, mock_db):
        """Test successful conversation creation."""
        # Arrange
        prompt = "Test conversation prompt"
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()

        # Act
        result = await repository.create_conversation(prompt=prompt, title="Test")

        # Assert
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        mock_db.refresh.assert_called_once()
        assert result.prompt == prompt

    def test_create_conversation_database_error(self, repository, mock_db, sample_conversation):
        """Test conversation creation with database error."""
        # Arrange
        mock_db.add = Mock()
        mock_db.commit = Mock(side_effect=Exception("Database error"))
        mock_db.rollback = Mock()

        # Act & Assert
        with pytest.raises(Exception, match="Database error"):
            repository.create_conversation(sample_conversation)
        
        mock_db.rollback.assert_called_once()

    def test_get_conversation_by_id_exists(self, repository, mock_db, sample_conversation):
        """Test retrieving existing conversation by ID."""
        # Arrange
        mock_query = Mock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = sample_conversation

        # Act
        result = repository.get_conversation_by_id(sample_conversation.id)

        # Assert
        mock_db.query.assert_called_once_with(AgentConversationSession)
        assert result == sample_conversation

    def test_get_conversation_by_id_not_exists(self, repository, mock_db):
        """Test retrieving non-existent conversation by ID."""
        # Arrange
        conversation_id = uuid.uuid4()
        mock_query = Mock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = None

        # Act
        result = repository.get_conversation_by_id(conversation_id)

        # Assert
        assert result is None

    def test_add_message_to_conversation_success(self, repository, mock_db, sample_conversation):
        """Test successfully adding message to conversation."""
        # Arrange
        message = AgentConversationMessage(
            id=uuid.uuid4(),
            conversation_id=sample_conversation.id,
            agent_id=uuid.uuid4(),
            content="Test message",
            message_order=1
        )
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()

        # Act
        result = repository.add_message_to_conversation(sample_conversation.id, message)

        # Assert
        mock_db.add.assert_called_once_with(message)
        mock_db.commit.assert_called_once()
        mock_db.refresh.assert_called_once_with(message)
        assert result == message

    def test_get_conversation_messages_with_pagination(self, repository, mock_db, sample_conversation):
        """Test retrieving conversation messages with pagination."""
        # Arrange
        messages = [
            AgentConversationMessage(
                id=uuid.uuid4(),
                conversation_id=sample_conversation.id,
                agent_id=uuid.uuid4(),
                content=f"Message {i}",
                message_order=i
            ) for i in range(1, 6)
        ]
        
        mock_query = Mock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = messages[:3]  # First 3 messages

        # Act
        result = repository.get_conversation_messages(
            sample_conversation.id, 
            skip=0, 
            limit=3
        )

        # Assert
        mock_db.query.assert_called_once_with(AgentConversationMessage)
        assert len(result) == 3
        assert all(isinstance(msg, AgentConversationMessage) for msg in result)

    def test_update_conversation_status(self, repository, mock_db, sample_conversation):
        """Test updating conversation status."""
        # Arrange
        new_status = ConversationStatus.ACTIVE
        mock_query = Mock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = sample_conversation
        mock_db.commit = Mock()

        # Act
        result = repository.update_conversation_status(sample_conversation.id, new_status)

        # Assert
        assert result.status == new_status
        mock_db.commit.assert_called_once()

    def test_get_active_conversations_for_user(self, repository, mock_db):
        """Test retrieving active conversations for a user."""
        # Arrange
        user_id = "test_user_123"
        conversations = [
            AgentConversationSession(
                id=uuid.uuid4(),
                prompt=f"Conversation {i}",
                user_id=user_id,
                status=ConversationStatus.ACTIVE
            ) for i in range(3)
        ]
        
        mock_query = Mock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = conversations

        # Act
        result = repository.get_active_conversations_for_user(user_id)

        # Assert
        assert len(result) == 3
        assert all(conv.user_id == user_id for conv in result)
        assert all(conv.status == ConversationStatus.ACTIVE for conv in result)

    def test_delete_conversation_success(self, repository, mock_db, sample_conversation):
        """Test successful conversation deletion."""
        # Arrange
        mock_query = Mock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = sample_conversation
        mock_db.delete = Mock()
        mock_db.commit = Mock()

        # Act
        result = repository.delete_conversation(sample_conversation.id)

        # Assert
        assert result is True
        mock_db.delete.assert_called_once_with(sample_conversation)
        mock_db.commit.assert_called_once()

    def test_delete_conversation_not_found(self, repository, mock_db):
        """Test deleting non-existent conversation."""
        # Arrange
        conversation_id = uuid.uuid4()
        mock_query = Mock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = None

        # Act
        result = repository.delete_conversation(conversation_id)

        # Assert
        assert result is False

    def test_get_conversation_statistics(self, repository, mock_db):
        """Test retrieving conversation statistics."""
        # Arrange
        mock_result = Mock()
        mock_result.fetchone.return_value = (10, 5, 3, 2)  # total, active, completed, failed
        mock_db.execute.return_value = mock_result

        # Act
        result = repository.get_conversation_statistics()

        # Assert
        expected = {
            "total_conversations": 10,
            "active_conversations": 5,
            "completed_conversations": 3,
            "failed_conversations": 2
        }
        assert result == expected

    @pytest.mark.asyncio
    async def test_conversation_lifecycle_integration(self, repository, mock_db):
        """Integration test for complete conversation lifecycle."""
        # This tests the interaction between multiple repository methods
        
        # Arrange
        conversation_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        
        conversation = AgentConversationSession(
            id=conversation_id,
            prompt="Integration test conversation",
            status=ConversationStatus.PENDING
        )
        
        message1 = AgentConversationMessage(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            agent_id=agent_id,
            content="First message",
            message_order=1
        )
        
        message2 = AgentConversationMessage(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            agent_id=agent_id,
            content="Second message", 
            message_order=2
        )

        # Mock database operations
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()
        
        # Mock queries
        mock_query = Mock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = conversation
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = [message1, message2]

        # Act & Assert
        # 1. Create conversation
        created_conv = repository.create_conversation(conversation)
        assert created_conv == conversation
        
        # 2. Add messages
        repository.add_message_to_conversation(conversation_id, message1)
        repository.add_message_to_conversation(conversation_id, message2)
        
        # 3. Update status
        updated_conv = repository.update_conversation_status(conversation_id, ConversationStatus.ACTIVE)
        assert updated_conv.status == ConversationStatus.ACTIVE
        
        # 4. Retrieve messages
        messages = repository.get_conversation_messages(conversation_id)
        assert len(messages) == 2
        
        # Verify all database operations were called
        assert mock_db.add.call_count == 3  # 1 conversation + 2 messages
        assert mock_db.commit.call_count == 4  # 3 creates + 1 update