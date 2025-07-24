"""
Test Conversation Monitoring System for Knowledge Graph Auto-Updates
Following TDD principles - write tests first, then implementation
"""

import asyncio
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest
from knowledge_graph.conversation_monitoring import (
    ConversationEvent,
    ConversationEventType,
    ConversationMonitor,
)
from knowledge_graph.graph_engine import KnowledgeGraph as GraphEngine

from database.conversation_models import Conversation, Message


class TestConversationMonitoring:
    """Test suite for conversation monitoring system"""

    def test_monitor_initialization(self):
        """Test that ConversationMonitor initializes correctly"""
        graph_engine = Mock(spec=GraphEngine)
        monitor = ConversationMonitor(graph_engine=graph_engine)

        assert monitor.graph_engine == graph_engine
        assert monitor.is_running is False
        assert monitor.event_queue.empty()

    def test_conversation_event_creation(self):
        """Test ConversationEvent data structure"""
        conversation_id = str(uuid.uuid4())
        message_id = str(uuid.uuid4())

        event = ConversationEvent(
            type=ConversationEventType.MESSAGE_ADDED,
            conversation_id=conversation_id,
            message_id=message_id,
            content="Hello, how are you?",
            timestamp=datetime.utcnow(),
        )

        assert event.type == ConversationEventType.MESSAGE_ADDED
        assert event.conversation_id == conversation_id
        assert event.message_id == message_id
        assert event.content == "Hello, how are you?"
        assert isinstance(event.timestamp, datetime)

    @pytest.mark.asyncio
    async def test_start_monitoring(self):
        """Test starting the conversation monitor"""
        graph_engine = Mock(spec=GraphEngine)
        monitor = ConversationMonitor(graph_engine=graph_engine)

        # Start monitoring
        await monitor.start()

        assert monitor.is_running is True

        # Stop monitoring
        await monitor.stop()
        assert monitor.is_running is False

    @pytest.mark.asyncio
    async def test_process_new_message(self):
        """Test processing a new message event"""
        graph_engine = Mock(spec=GraphEngine)
        monitor = ConversationMonitor(graph_engine=graph_engine)

        # Create a test message with UUID
        message_id = uuid.uuid4()
        conversation_id = uuid.uuid4()

        message = Message(
            id=message_id,
            conversation_id=conversation_id,
            content="I need help with Python programming",
            role="user",
            created_at=datetime.utcnow(),
        )

        # Process the message
        await monitor.process_message(message)

        # Verify event was queued
        assert not monitor.event_queue.empty()
        event = monitor.event_queue.get_nowait()
        assert event.type == ConversationEventType.MESSAGE_ADDED
        assert event.message_id == str(message_id)
        assert event.content == "I need help with Python programming"

    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch processing of multiple messages"""
        graph_engine = Mock(spec=GraphEngine)
        monitor = ConversationMonitor(graph_engine=graph_engine)

        conversation_id = uuid.uuid4()

        # Add multiple messages
        messages = [
            Message(
                id=uuid.uuid4(),
                conversation_id=conversation_id,
                content=f"Message {i}",
                role="user" if i % 2 == 0 else "assistant",
                created_at=datetime.utcnow(),
            )
            for i in range(5)
        ]

        # Process all messages
        for message in messages:
            await monitor.process_message(message)

        # Process batch
        processed_count = await monitor.process_batch()

        assert processed_count == 5
        assert monitor.event_queue.empty()

    @pytest.mark.asyncio
    async def test_conversation_update_trigger(self):
        """Test that conversation updates trigger graph updates"""
        graph_engine = Mock(spec=GraphEngine)
        graph_engine.update_from_conversation = AsyncMock()

        monitor = ConversationMonitor(graph_engine=graph_engine)

        # Create a conversation with messages
        conversation_id = uuid.uuid4()
        message_id = uuid.uuid4()

        conversation = Conversation(
            id=conversation_id,
            title="Python Help",
            created_at=datetime.utcnow(),
        )

        message = Message(
            id=message_id,
            conversation_id=conversation_id,
            content="What are Python decorators?",
            role="user",
            created_at=datetime.utcnow(),
        )

        # Process the conversation update
        await monitor.on_conversation_update(conversation, message)

        # Verify graph engine was called
        graph_engine.update_from_conversation.assert_called_once()
        call_args = graph_engine.update_from_conversation.call_args
        assert call_args[0][0] == conversation
        assert call_args[0][1] == message

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in conversation monitoring"""
        graph_engine = Mock(spec=GraphEngine)
        graph_engine.update_from_conversation = AsyncMock(
            side_effect=Exception("Graph update failed")
        )

        monitor = ConversationMonitor(graph_engine=graph_engine)

        message_id = uuid.uuid4()
        conversation_id = uuid.uuid4()

        message = Message(
            id=message_id,
            conversation_id=conversation_id,
            content="Test message",
            role="user",
            created_at=datetime.utcnow(),
        )

        # Process should not raise exception
        await monitor.process_message(message)

        # Error should be logged (we'd check logs in real implementation)
        # For now, just verify the system continues running
        assert monitor.event_queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_real_time_subscription(self):
        """Test real-time subscription to conversation updates"""
        graph_engine = Mock(spec=GraphEngine)
        monitor = ConversationMonitor(graph_engine=graph_engine)

        # Subscribe to updates
        callback_called = False

        async def callback(event: ConversationEvent):
            nonlocal callback_called
            callback_called = True
            assert event.type == ConversationEventType.MESSAGE_ADDED

        monitor.subscribe(callback)

        # Trigger an event
        message_id = uuid.uuid4()
        conversation_id = uuid.uuid4()

        message = Message(
            id=message_id,
            conversation_id=conversation_id,
            content="Test subscription",
            role="user",
            created_at=datetime.utcnow(),
        )

        await monitor.process_message(message)
        await monitor.notify_subscribers()

        assert callback_called

    def test_conversation_event_types(self):
        """Test all conversation event types"""
        assert ConversationEventType.MESSAGE_ADDED.value == "message_added"
        assert ConversationEventType.MESSAGE_UPDATED.value == "message_updated"
        assert ConversationEventType.MESSAGE_DELETED.value == "message_deleted"
        assert ConversationEventType.CONVERSATION_CREATED.value == "conversation_created"
        assert ConversationEventType.CONVERSATION_UPDATED.value == "conversation_updated"

    @pytest.mark.asyncio
    async def test_concurrent_message_processing(self):
        """Test handling concurrent message updates"""
        graph_engine = Mock(spec=GraphEngine)
        graph_engine.update_from_conversation = AsyncMock()

        monitor = ConversationMonitor(graph_engine=graph_engine)

        conversation_id = uuid.uuid4()

        # Create multiple concurrent messages
        tasks = []
        for i in range(10):
            message = Message(
                id=uuid.uuid4(),
                conversation_id=conversation_id,
                content=f"Concurrent message {i}",
                role="user",
                created_at=datetime.utcnow(),
            )
            tasks.append(monitor.process_message(message))

        # Process all messages concurrently
        await asyncio.gather(*tasks)

        # Verify all messages were queued
        assert monitor.event_queue.qsize() == 10
