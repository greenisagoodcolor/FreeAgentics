"""Unit tests for Conversation Service.

Tests the service layer for agent conversations following Clean Code principles
and separation of concerns as recommended by Robert C. Martin.
"""

import uuid
from unittest.mock import AsyncMock, Mock

import pytest

from api.v1.models.agent_conversation import (
    AgentConversationRequest,
    AgentStatus,
    ConversationConfig,
)
from api.v1.services.conversation_service import ConversationService
from database.models import Agent, AgentConversationSession, ConversationStatus


class TestConversationService:
    """Unit tests for ConversationService."""

    @pytest.fixture
    def mock_repository(self):
        """Mock conversation repository."""
        return Mock()

    @pytest.fixture
    def mock_llm_service(self):
        """Mock LLM service."""
        return AsyncMock()

    @pytest.fixture
    def mock_gmn_parser(self):
        """Mock GMN parser service."""
        return AsyncMock()

    @pytest.fixture
    def mock_pymdp_service(self):
        """Mock PyMDP service."""
        return AsyncMock()

    @pytest.fixture
    def service(self, mock_repository, mock_llm_service, mock_gmn_parser, mock_pymdp_service):
        """Conversation service with mocked dependencies."""
        return ConversationService(
            repository=mock_repository,
            llm_service=mock_llm_service,
            gmn_parser=mock_gmn_parser,
            pymdp_service=mock_pymdp_service,
        )

    @pytest.fixture
    def sample_request(self):
        """Sample conversation request."""
        return AgentConversationRequest(
            prompt="Create agents to discuss sustainable energy",
            config={"agent_count": 2, "conversation_turns": 5},
            metadata={"session_type": "exploration"},
        )

    @pytest.fixture
    def sample_agent(self):
        """Sample agent for testing."""
        return Agent(
            id=uuid.uuid4(),
            name="Test Agent",
            template="advocate",
            status="ready",
            gmn_spec='{"states": ["listening", "thinking"], "actions": ["respond"]}',
            beliefs={"sustainability": 0.8},
            preferences={"energy_type": "renewable"},
        )

    @pytest.fixture
    def sample_conversation(self):
        """Sample conversation session."""
        return AgentConversationSession(
            id=uuid.uuid4(),
            prompt="Test conversation prompt",
            title="Test Conversation",
            status=ConversationStatus.PENDING,
            message_count=0,
            agent_count=1,
        )

    @pytest.mark.asyncio
    async def test_create_conversation_success(
        self,
        service,
        mock_repository,
        mock_llm_service,
        mock_gmn_parser,
        sample_request,
        sample_agent,
    ):
        """Test successful conversation creation."""
        # Arrange
        conversation_id = uuid.uuid4()
        agent_id = uuid.uuid4()

        # Mock GMN generation
        mock_gmn_spec = {
            "name": "advocate_agent",
            "states": ["listening", "thinking", "responding"],
            "actions": ["listen", "respond", "question"],
        }
        mock_gmn_parser.generate_gmn_from_prompt.return_value = mock_gmn_spec

        # Mock agent creation
        created_agent = Agent(
            id=agent_id,
            name="Advocate Agent",
            template="advocate",
            gmn_spec=str(mock_gmn_spec),
            beliefs={"sustainability": 0.8},
        )
        mock_repository.create_agent.return_value = created_agent

        # Mock conversation creation
        created_conversation = AgentConversationSession(
            id=conversation_id,
            prompt=sample_request.prompt,
            status=ConversationStatus.PENDING,
            agent_count=1,
        )
        mock_repository.create_conversation.return_value = created_conversation

        # Act
        result = await service.create_conversation(sample_request)

        # Assert
        assert result.conversation_id == conversation_id
        assert result.agent_id == agent_id
        assert result.status == AgentStatus.READY
        assert result.gmn_structure == mock_gmn_spec

        # Verify service calls
        mock_gmn_parser.generate_gmn_from_prompt.assert_called_once_with(sample_request.prompt)
        mock_repository.create_agent.assert_called_once()
        mock_repository.create_conversation.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_conversation_llm_failure(self, service, mock_gmn_parser, sample_request):
        """Test conversation creation when LLM service fails."""
        # Arrange
        mock_gmn_parser.generate_gmn_from_prompt.side_effect = Exception("LLM service unavailable")

        # Act & Assert
        with pytest.raises(Exception, match="LLM service unavailable"):
            await service.create_conversation(sample_request)

    @pytest.mark.asyncio
    async def test_create_conversation_database_failure(
        self, service, mock_repository, mock_gmn_parser, sample_request
    ):
        """Test conversation creation when database fails."""
        # Arrange
        mock_gmn_spec = {"name": "test_agent"}
        mock_gmn_parser.generate_gmn_from_prompt.return_value = mock_gmn_spec
        mock_repository.create_agent.side_effect = Exception("Database connection failed")

        # Act & Assert
        with pytest.raises(Exception, match="Database connection failed"):
            await service.create_conversation(sample_request)

    @pytest.mark.asyncio
    async def test_get_conversation_success(self, service, mock_repository, sample_conversation):
        """Test successful conversation retrieval."""
        # Arrange
        mock_repository.get_conversation_by_id.return_value = sample_conversation

        # Act
        result = await service.get_conversation(sample_conversation.id)

        # Assert
        assert result == sample_conversation
        mock_repository.get_conversation_by_id.assert_called_once_with(sample_conversation.id)

    @pytest.mark.asyncio
    async def test_get_conversation_not_found(self, service, mock_repository):
        """Test conversation retrieval when conversation doesn't exist."""
        # Arrange
        conversation_id = uuid.uuid4()
        mock_repository.get_conversation_by_id.return_value = None

        # Act
        result = await service.get_conversation(conversation_id)

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_start_conversation_success(
        self, service, mock_repository, mock_pymdp_service, sample_conversation
    ):
        """Test successfully starting a conversation."""
        # Arrange
        mock_repository.get_conversation_by_id.return_value = sample_conversation
        mock_repository.update_conversation_status.return_value = sample_conversation
        mock_pymdp_service.initialize_agent.return_value = True

        # Act
        result = await service.start_conversation(sample_conversation.id)

        # Assert
        assert result.status == ConversationStatus.ACTIVE
        mock_repository.update_conversation_status.assert_called_once_with(
            sample_conversation.id, ConversationStatus.ACTIVE
        )

    @pytest.mark.asyncio
    async def test_start_conversation_already_active(self, service, mock_repository):
        """Test starting a conversation that's already active."""
        # Arrange
        active_conversation = AgentConversationSession(
            id=uuid.uuid4(), prompt="Test", status=ConversationStatus.ACTIVE
        )
        mock_repository.get_conversation_by_id.return_value = active_conversation

        # Act & Assert
        with pytest.raises(ValueError, match="Conversation is already active"):
            await service.start_conversation(active_conversation.id)

    @pytest.mark.asyncio
    async def test_add_message_success(
        self, service, mock_repository, sample_conversation, sample_agent
    ):
        """Test successfully adding a message to conversation."""
        # Arrange
        message_content = "This is a test message"
        mock_repository.get_conversation_by_id.return_value = sample_conversation
        mock_repository.add_message_to_conversation.return_value = Mock(
            id=uuid.uuid4(), content=message_content, agent_id=sample_agent.id, message_order=1
        )

        # Act
        result = await service.add_message(sample_conversation.id, sample_agent.id, message_content)

        # Assert
        assert result.content == message_content
        assert result.agent_id == sample_agent.id
        mock_repository.add_message_to_conversation.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_message_conversation_not_found(self, service, mock_repository):
        """Test adding message to non-existent conversation."""
        # Arrange
        conversation_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        mock_repository.get_conversation_by_id.return_value = None

        # Act & Assert
        with pytest.raises(ValueError, match="Conversation not found"):
            await service.add_message(conversation_id, agent_id, "Test message")

    @pytest.mark.asyncio
    async def test_get_conversation_messages_with_pagination(
        self, service, mock_repository, sample_conversation
    ):
        """Test retrieving conversation messages with pagination."""
        # Arrange
        mock_messages = [
            Mock(id=uuid.uuid4(), content=f"Message {i}", message_order=i) for i in range(1, 6)
        ]
        mock_repository.get_conversation_messages.return_value = mock_messages[:3]

        # Act
        result = await service.get_conversation_messages(sample_conversation.id, skip=0, limit=3)

        # Assert
        assert len(result) == 3
        mock_repository.get_conversation_messages.assert_called_once_with(
            sample_conversation.id, skip=0, limit=3
        )

    @pytest.mark.asyncio
    async def test_complete_conversation_success(
        self, service, mock_repository, sample_conversation
    ):
        """Test successfully completing a conversation."""
        # Arrange
        active_conversation = AgentConversationSession(
            id=sample_conversation.id,
            prompt=sample_conversation.prompt,
            status=ConversationStatus.ACTIVE,
            message_count=5,
        )
        mock_repository.get_conversation_by_id.return_value = active_conversation
        mock_repository.update_conversation_status.return_value = active_conversation

        # Act
        result = await service.complete_conversation(sample_conversation.id)

        # Assert
        assert result.status == ConversationStatus.COMPLETED
        mock_repository.update_conversation_status.assert_called_once_with(
            sample_conversation.id, ConversationStatus.COMPLETED
        )

    @pytest.mark.asyncio
    async def test_complete_conversation_not_active(
        self, service, mock_repository, sample_conversation
    ):
        """Test completing a conversation that's not active."""
        # Arrange - conversation is still pending
        mock_repository.get_conversation_by_id.return_value = sample_conversation

        # Act & Assert
        with pytest.raises(ValueError, match="Conversation is not active"):
            await service.complete_conversation(sample_conversation.id)

    @pytest.mark.asyncio
    async def test_conversation_error_handling(self, service, mock_repository, sample_conversation):
        """Test error handling and rollback on conversation failure."""
        # Arrange
        mock_repository.get_conversation_by_id.return_value = sample_conversation
        mock_repository.update_conversation_status.side_effect = Exception("Database error")

        # Act & Assert
        with pytest.raises(Exception, match="Database error"):
            await service.start_conversation(sample_conversation.id)

    @pytest.mark.asyncio
    async def test_concurrent_conversation_access(
        self, service, mock_repository, sample_conversation
    ):
        """Test handling concurrent access to the same conversation."""
        # This test would verify that the service handles concurrent modifications properly
        # In a real implementation, this might involve locking or optimistic concurrency control

        # Arrange
        mock_repository.get_conversation_by_id.return_value = sample_conversation

        # Simulate concurrent modification by having the status change between calls
        def side_effect(*args, **kwargs):
            sample_conversation.status = ConversationStatus.ACTIVE
            return sample_conversation

        mock_repository.update_conversation_status.side_effect = side_effect

        # Act
        result1 = await service.start_conversation(sample_conversation.id)

        # The second call should detect the conversation is already active
        with pytest.raises(ValueError, match="Conversation is already active"):
            await service.start_conversation(sample_conversation.id)

        # Assert
        assert result1.status == ConversationStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_service_initialization_validation(self):
        """Test that service properly validates its dependencies on initialization."""
        # This tests the service constructor validation

        # Act & Assert - should raise error if required dependencies are None
        with pytest.raises(ValueError, match="Repository is required"):
            ConversationService(repository=None)

    def test_service_configuration_validation(self, service):
        """Test that service validates configuration parameters."""
        # Arrange
        invalid_config = ConversationConfig(
            agent_count=0,  # Invalid: must be >= 1
            max_turns=-1,  # Invalid: must be >= 1
        )

        # Act & Assert
        with pytest.raises(ValueError):
            service._validate_configuration(invalid_config)

    @pytest.mark.asyncio
    async def test_service_cleanup_on_error(
        self, service, mock_repository, mock_gmn_parser, sample_request
    ):
        """Test that service properly cleans up resources on error."""
        # Arrange
        mock_gmn_parser.generate_gmn_from_prompt.return_value = {"name": "test"}

        # Create agent succeeds
        created_agent = Mock(id=uuid.uuid4())
        mock_repository.create_agent.return_value = created_agent

        # But conversation creation fails
        mock_repository.create_conversation.side_effect = Exception("DB error")

        # Act & Assert
        with pytest.raises(Exception, match="DB error"):
            await service.create_conversation(sample_request)

        # Verify cleanup - agent should be deleted if conversation creation fails
        mock_repository.delete_agent.assert_called_once_with(created_agent.id)
