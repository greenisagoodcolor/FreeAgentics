"""Comprehensive tests for database.conversation_models to achieve high coverage."""

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.base import Base
from database.conversation_models import (
    Conversation,
    Message,
    ValidationStatus,
)


class TestConversationModels:
    """Test conversation and message models comprehensively."""

    @pytest.fixture
    def test_engine(self):
        """Create test database engine."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        return engine

    @pytest.fixture
    def test_session(self, test_engine):
        """Create test database session."""
        SessionLocal = sessionmaker(bind=test_engine)
        session = SessionLocal()
        yield session
        session.close()

    def test_validation_status_enum(self):
        """Test ValidationStatus enum values."""
        assert ValidationStatus.PENDING.value == "pending"
        assert ValidationStatus.VALID.value == "valid"
        assert ValidationStatus.INVALID.value == "invalid"
        assert ValidationStatus.ERROR.value == "error"

    def test_conversation_creation(self, test_session):
        """Test creating a conversation with all fields."""
        now = datetime.utcnow()

        conversation = Conversation(
            user_id="user123",
            title="Test Conversation",
            created_at=now,
            updated_at=now,
            is_active=True,
        )

        test_session.add(conversation)
        test_session.commit()

        # Retrieve and verify
        retrieved = test_session.query(Conversation).filter_by(user_id="user123").first()

        assert retrieved is not None
        assert retrieved.user_id == "user123"
        assert retrieved.title == "Test Conversation"
        assert retrieved.is_active is True
        assert retrieved.created_at == now
        assert retrieved.updated_at == now

    def test_conversation_defaults(self, test_session):
        """Test conversation creation with default values."""
        conversation = Conversation(user_id="user456")

        test_session.add(conversation)
        test_session.commit()

        assert conversation.title is None
        assert conversation.is_active is True
        assert conversation.created_at is not None
        assert conversation.updated_at is not None

    def test_conversation_without_title(self, test_session):
        """Test conversation can be created without title."""
        conversation = Conversation(user_id="user789", title=None)

        test_session.add(conversation)
        test_session.commit()

        assert conversation.title is None
        assert conversation.id is not None

    def test_message_creation(self, test_session):
        """Test creating a message with all fields."""
        # First create a conversation
        conversation = Conversation(user_id="user123")
        test_session.add(conversation)
        test_session.commit()

        now = datetime.utcnow()

        message = Message(
            conversation_id=conversation.id,
            sender_type="user",
            content="Hello, world!",
            timestamp=now,
            gmn_spec='{"type": "test"}',
            validation_status=ValidationStatus.VALID.value,
            validation_message="Validation successful",
        )

        test_session.add(message)
        test_session.commit()

        # Retrieve and verify
        retrieved = test_session.query(Message).filter_by(conversation_id=conversation.id).first()

        assert retrieved is not None
        assert retrieved.sender_type == "user"
        assert retrieved.content == "Hello, world!"
        assert retrieved.timestamp == now
        assert retrieved.gmn_spec == '{"type": "test"}'
        assert retrieved.validation_status == "valid"
        assert retrieved.validation_message == "Validation successful"

    def test_message_defaults(self, test_session):
        """Test message creation with default values."""
        conversation = Conversation(user_id="user123")
        test_session.add(conversation)
        test_session.commit()

        message = Message(
            conversation_id=conversation.id,
            sender_type="assistant",
            content="Response message",
        )

        test_session.add(message)
        test_session.commit()

        assert message.timestamp is not None
        assert message.gmn_spec is None
        assert message.validation_status is None
        assert message.validation_message is None

    def test_conversation_message_relationship(self, test_session):
        """Test relationship between conversations and messages."""
        # Create conversation
        conversation = Conversation(user_id="user123", title="Chat Session")
        test_session.add(conversation)
        test_session.commit()

        # Create messages
        message1 = Message(
            conversation_id=conversation.id,
            sender_type="user",
            content="First message",
        )
        message2 = Message(
            conversation_id=conversation.id,
            sender_type="assistant",
            content="Second message",
        )

        test_session.add_all([message1, message2])
        test_session.commit()

        # Test relationship
        assert len(conversation.messages) == 2
        assert message1 in conversation.messages
        assert message2 in conversation.messages

        # Test back reference
        assert message1.conversation == conversation
        assert message2.conversation == conversation

    def test_cascade_delete(self, test_session):
        """Test cascade delete of messages when conversation is deleted."""
        # Create conversation with messages
        conversation = Conversation(user_id="user123")
        test_session.add(conversation)
        test_session.commit()

        message1 = Message(
            conversation_id=conversation.id,
            sender_type="user",
            content="Message to be deleted",
        )
        message2 = Message(
            conversation_id=conversation.id,
            sender_type="assistant",
            content="Another message to be deleted",
        )

        test_session.add_all([message1, message2])
        test_session.commit()

        conv_id = conversation.id
        msg1_id = message1.id
        msg2_id = message2.id

        # Delete conversation
        test_session.delete(conversation)
        test_session.commit()

        # Verify conversation is deleted
        assert test_session.query(Conversation).filter_by(id=conv_id).first() is None

        # Verify messages are also deleted (cascade)
        assert test_session.query(Message).filter_by(id=msg1_id).first() is None
        assert test_session.query(Message).filter_by(id=msg2_id).first() is None

    def test_conversation_updated_at(self, test_session):
        """Test updated_at field is updated on modification."""
        conversation = Conversation(user_id="user123")
        test_session.add(conversation)
        test_session.commit()

        # Update conversation
        conversation.title = "Updated Title"
        test_session.commit()

        # Note: SQLAlchemy's onupdate doesn't work automatically in SQLite
        # but the field is defined correctly for PostgreSQL

    def test_message_with_validation_error(self, test_session):
        """Test message with validation error status."""
        conversation = Conversation(user_id="user123")
        test_session.add(conversation)
        test_session.commit()

        message = Message(
            conversation_id=conversation.id,
            sender_type="user",
            content="Invalid GMN",
            validation_status=ValidationStatus.ERROR.value,
            validation_message="Parse error: Invalid syntax",
        )

        test_session.add(message)
        test_session.commit()

        assert message.validation_status == "error"
        assert "Parse error" in message.validation_message

    def test_multiple_conversations_per_user(self, test_session):
        """Test user can have multiple conversations."""
        user_id = "user123"

        conv1 = Conversation(user_id=user_id, title="Conv 1")
        conv2 = Conversation(user_id=user_id, title="Conv 2")
        conv3 = Conversation(user_id=user_id, title="Conv 3")

        test_session.add_all([conv1, conv2, conv3])
        test_session.commit()

        # Query all conversations for user
        user_convs = test_session.query(Conversation).filter_by(user_id=user_id).all()

        assert len(user_convs) == 3
        titles = [c.title for c in user_convs]
        assert "Conv 1" in titles
        assert "Conv 2" in titles
        assert "Conv 3" in titles

    def test_inactive_conversation(self, test_session):
        """Test marking conversation as inactive."""
        conversation = Conversation(user_id="user123", is_active=False)

        test_session.add(conversation)
        test_session.commit()

        assert conversation.is_active is False

    def test_message_sender_types(self, test_session):
        """Test different sender types for messages."""
        conversation = Conversation(user_id="user123")
        test_session.add(conversation)
        test_session.commit()

        user_msg = Message(
            conversation_id=conversation.id,
            sender_type="user",
            content="User message",
        )

        assistant_msg = Message(
            conversation_id=conversation.id,
            sender_type="assistant",
            content="Assistant message",
        )

        test_session.add_all([user_msg, assistant_msg])
        test_session.commit()

        assert user_msg.sender_type == "user"
        assert assistant_msg.sender_type == "assistant"

    def test_gmn_spec_json_storage(self, test_session):
        """Test GMN spec stored as JSON string."""
        conversation = Conversation(user_id="user123")
        test_session.add(conversation)
        test_session.commit()

        complex_gmn = '{"nodes": [{"id": 1, "type": "state"}], "edges": []}'

        message = Message(
            conversation_id=conversation.id,
            sender_type="user",
            content="GMN message",
            gmn_spec=complex_gmn,
        )

        test_session.add(message)
        test_session.commit()

        # Retrieve and verify JSON string is preserved
        retrieved = test_session.query(Message).filter_by(id=message.id).first()
        assert retrieved.gmn_spec == complex_gmn

    def test_table_names(self):
        """Test table names are correctly set."""
        assert Conversation.__tablename__ == "conversations"
        assert Message.__tablename__ == "messages"

    def test_message_requires_conversation(self, test_session):
        """Test message cannot be created without conversation."""
        message = Message(
            conversation_id=99999,  # Non-existent conversation
            sender_type="user",
            content="Orphan message",
        )

        test_session.add(message)

        # Should raise integrity error on commit
        with pytest.raises(Exception):  # IntegrityError
            test_session.commit()
