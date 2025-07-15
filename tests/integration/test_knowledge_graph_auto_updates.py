"""
Integration test for Knowledge Graph Auto-Updates
Tests the complete pipeline from conversation to knowledge graph using real NLP
"""

import asyncio
import uuid
from datetime import datetime

import pytest

from database.conversation_models import Conversation, Message
from knowledge_graph.conversation_auto_updater import ConversationAutoUpdater


class TestKnowledgeGraphAutoUpdates:
    """Integration tests for the complete auto-update pipeline"""

    @pytest.mark.asyncio
    async def test_complete_pipeline_real_nlp(self):
        """Test the complete pipeline with real NLP extraction"""
        # Initialize the auto-updater
        auto_updater = ConversationAutoUpdater()

        # Create a test message with technical content
        message = Message(
            id=uuid.uuid4(),
            conversation_id=uuid.uuid4(),
            content="I'm using Python and Django to build a web application with React frontend.",
            role="user",
            created_at=datetime.utcnow(),
        )

        # Process the message
        extraction_result = await auto_updater.process_message(message)

        # Verify entities were extracted
        assert len(extraction_result.entities) > 0

        # Check that we found expected technologies
        entity_texts = [e.text for e in extraction_result.entities]
        assert "Python" in entity_texts
        assert "Django" in entity_texts
        assert "React" in entity_texts

        # Verify relationships were extracted
        assert len(extraction_result.relationships) > 0

        # Check statistics
        stats = auto_updater.get_statistics()
        assert stats["messages_processed"] == 1
        assert stats["entities_extracted"] > 0
        assert stats["nodes_created"] > 0

    @pytest.mark.asyncio
    async def test_bidirectional_sync(self):
        """Test bidirectional sync between conversations and knowledge graph"""
        auto_updater = ConversationAutoUpdater()

        conversation_id = str(uuid.uuid4())

        # Process a message to populate the knowledge graph
        message = Message(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            content="TensorFlow is great for machine learning projects.",
            role="user",
            created_at=datetime.utcnow(),
        )

        await auto_updater.process_message(message)

        # Query knowledge for the conversation
        knowledge = await auto_updater.query_knowledge_for_conversation(conversation_id)

        # Should find entities related to this conversation
        assert "direct_entities" in knowledge or "connected_entities" in knowledge

        # Test context suggestions
        suggestions = await auto_updater.suggest_conversation_context(
            conversation_id, "What about PyTorch?"
        )

        # Should provide suggestions based on existing knowledge
        assert isinstance(suggestions, list)

    @pytest.mark.asyncio
    async def test_conversation_processing(self):
        """Test processing an entire conversation"""
        auto_updater = ConversationAutoUpdater()

        # Create a conversation with multiple messages
        conversation_id = uuid.uuid4()
        conversation = Conversation(
            id=conversation_id, title="AI Discussion", created_at=datetime.utcnow()
        )

        # Mock messages in the conversation
        messages = [
            Message(
                id=uuid.uuid4(),
                conversation_id=conversation_id,
                content="I'm interested in artificial intelligence.",
                role="user",
                created_at=datetime.utcnow(),
            ),
            Message(
                id=uuid.uuid4(),
                conversation_id=conversation_id,
                content="Specifically, I want to learn about neural networks.",
                role="user",
                created_at=datetime.utcnow(),
            ),
            Message(
                id=uuid.uuid4(),
                conversation_id=conversation_id,
                content="PyTorch and TensorFlow are popular frameworks.",
                role="assistant",
                created_at=datetime.utcnow(),
            ),
        ]

        conversation.messages = messages

        # Process the entire conversation
        stats = await auto_updater.process_conversation(conversation)

        # Verify processing stats
        assert stats["messages_processed"] == 3
        assert stats["entities_extracted"] > 0
        assert "processing_time" in stats

    @pytest.mark.asyncio
    async def test_real_time_monitoring(self):
        """Test real-time conversation monitoring"""
        auto_updater = ConversationAutoUpdater()

        # Start the auto-updater
        await auto_updater.start()

        try:
            # Simulate a new message being added
            message = Message(
                id=uuid.uuid4(),
                conversation_id=uuid.uuid4(),
                content="JavaScript and Node.js are great for web development.",
                role="user",
                created_at=datetime.utcnow(),
            )

            # Process the message through the monitor
            await auto_updater.conversation_monitor.process_message(message)

            # Allow some time for processing
            await asyncio.sleep(0.1)

            # Check that the message was processed
            stats = auto_updater.get_statistics()
            assert stats["messages_processed"] >= 1

        finally:
            # Stop the auto-updater
            await auto_updater.stop()

    @pytest.mark.asyncio
    async def test_knowledge_export(self):
        """Test exporting knowledge derived from a conversation"""
        auto_updater = ConversationAutoUpdater()

        conversation_id = str(uuid.uuid4())

        # Process some messages
        messages = [
            "I'm learning about Docker containers.",
            "Kubernetes is used for orchestration.",
            "Both are important for DevOps.",
        ]

        for content in messages:
            message = Message(
                id=uuid.uuid4(),
                conversation_id=conversation_id,
                content=content,
                role="user",
                created_at=datetime.utcnow(),
            )
            await auto_updater.process_message(message)

        # Export the knowledge
        exported_knowledge = await auto_updater.export_conversation_knowledge(conversation_id)

        # Verify export structure
        assert "extraction_metadata" in exported_knowledge
        assert "nlp_model" in exported_knowledge["extraction_metadata"]
        assert "entity_types_detected" in exported_knowledge["extraction_metadata"]

    @pytest.mark.asyncio
    async def test_entity_deduplication(self):
        """Test that duplicate entities are properly handled"""
        auto_updater = ConversationAutoUpdater()

        conversation_id = str(uuid.uuid4())

        # Process messages with overlapping entities
        messages = [
            "Python is a great programming language.",
            "I use Python for data science.",
            "Python has excellent libraries.",
        ]

        for content in messages:
            message = Message(
                id=uuid.uuid4(),
                conversation_id=conversation_id,
                content=content,
                role="user",
                created_at=datetime.utcnow(),
            )
            await auto_updater.process_message(message)

        # Query the knowledge graph
        knowledge = await auto_updater.query_knowledge_for_conversation(conversation_id)

        # Python should appear as a single entity, not duplicated
        direct_entities = knowledge.get("direct_entities", [])
        python_entities = [e for e in direct_entities if "python" in e.get("label", "").lower()]

        # Should have at most one Python entity (due to deduplication)
        assert len(python_entities) <= 1

    @pytest.mark.asyncio
    async def test_performance_with_large_content(self):
        """Test performance with larger content"""
        auto_updater = ConversationAutoUpdater()

        # Create a longer, more complex message
        long_content = """
        In modern software development, we use various technologies and frameworks.
        Python is excellent for backend development with Django or Flask frameworks.
        JavaScript powers the frontend with React, Angular, or Vue.js.
        For databases, we might use PostgreSQL, MongoDB, or Redis.
        Cloud platforms like AWS, Azure, and Google Cloud provide infrastructure.
        DevOps tools include Docker, Kubernetes, Jenkins, and Terraform.
        Machine learning frameworks like TensorFlow and PyTorch are becoming essential.
        """

        message = Message(
            id=uuid.uuid4(),
            conversation_id=uuid.uuid4(),
            content=long_content,
            role="user",
            created_at=datetime.utcnow(),
        )

        # Process the message and measure time
        import time

        start_time = time.time()
        extraction_result = await auto_updater.process_message(message)
        processing_time = time.time() - start_time

        # Should complete reasonably quickly
        assert processing_time < 5.0  # Less than 5 seconds

        # Should extract many entities
        assert len(extraction_result.entities) >= 10

        # Should extract some relationships
        assert len(extraction_result.relationships) > 0
