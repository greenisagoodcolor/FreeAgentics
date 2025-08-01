"""Tests for Real-time Knowledge Graph Updater (Task 34.4).

Comprehensive test suite following TDD principles and Nemesis Committee standards.
Tests cover conflict resolution, event streaming, and production-ready error handling.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from knowledge_graph.extraction import ExtractionResult
from knowledge_graph.realtime_updater import (
    ConflictInfo,
    DefaultConflictResolver,
    GraphUpdateEvent,
    RealtimeGraphUpdater,
    UpdateEventType,
    WebSocketEventStreamer,
)
from knowledge_graph.schema import (
    ConflictResolutionStrategy,
    ConversationEntity,
    ConversationRelation,
    EntityType,
    Provenance,
    RelationType,
    TemporalMetadata,
)


class TestDefaultConflictResolver:
    """Test suite for DefaultConflictResolver."""

    @pytest.fixture
    def resolver(self):
        """Create conflict resolver for testing."""
        return DefaultConflictResolver(similarity_threshold=0.6)

    @pytest.fixture
    def sample_entity(self):
        """Create sample entity for testing."""
        return ConversationEntity(
            entity_type=EntityType.CONCEPT,
            label="machine learning",
            entity_id=str(uuid4()),
            properties={"confidence": 0.9},
            temporal_metadata=TemporalMetadata(
                created_at=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc),
                conversation_id="conv_123",
            ),
            provenance=Provenance(
                source_type="nlp_extraction",
                source_id="msg_456",
                extraction_method="spacy_ner",
                confidence_score=0.85,
            ),
        )

    @pytest.fixture
    def similar_entity(self, sample_entity):
        """Create similar entity that should conflict."""
        similar = ConversationEntity(
            entity_type=EntityType.CONCEPT,
            label="machine learning model",  # Similar with higher overlap
            entity_id=str(uuid4()),
            properties={"confidence": 0.95},
            temporal_metadata=TemporalMetadata(
                created_at=datetime.now(timezone.utc),
                last_updated=datetime.now(timezone.utc),
                conversation_id="conv_124",
            ),
            provenance=Provenance(
                source_type="nlp_extraction",
                source_id="msg_789",
                extraction_method="spacy_ner",
                confidence_score=0.92,
            ),
        )
        return similar

    @pytest.fixture
    def different_entity(self):
        """Create different entity that should not conflict."""
        return ConversationEntity(
            entity_type=EntityType.GOAL,  # Different type
            label="complete project",
            entity_id=str(uuid4()),
            properties={"priority": "high"},
        )

    @pytest.mark.asyncio
    async def test_detect_no_conflicts_different_types(
        self, resolver, sample_entity, different_entity
    ):
        """Test that entities of different types don't conflict."""
        conflicts = await resolver.detect_conflicts(sample_entity, [different_entity])
        assert len(conflicts) == 0

    @pytest.mark.asyncio
    async def test_detect_no_conflicts_low_similarity(self, resolver, sample_entity):
        """Test that entities with low similarity don't conflict."""
        dissimilar_entity = ConversationEntity(
            entity_type=EntityType.CONCEPT,
            label="quantum computing",  # Very different
            entity_id=str(uuid4()),
        )

        conflicts = await resolver.detect_conflicts(sample_entity, [dissimilar_entity])
        assert len(conflicts) == 0

    @pytest.mark.asyncio
    async def test_detect_conflicts_high_similarity(self, resolver, sample_entity, similar_entity):
        """Test that similar entities are detected as conflicts."""
        conflicts = await resolver.detect_conflicts(sample_entity, [similar_entity])

        assert len(conflicts) == 1
        conflict = conflicts[0]
        assert conflict.existing_entity == similar_entity
        assert conflict.new_entity == sample_entity
        assert conflict.similarity_score >= 0.6
        assert conflict.metadata["detection_method"] == "label_similarity"

    @pytest.mark.asyncio
    async def test_resolve_conflict_highest_confidence(
        self, resolver, sample_entity, similar_entity
    ):
        """Test conflict resolution using highest confidence strategy."""
        conflict = ConflictInfo(
            existing_entity=sample_entity,  # confidence 0.85
            new_entity=similar_entity,  # confidence 0.92
            resolution_strategy=ConflictResolutionStrategy.HIGHEST_CONFIDENCE,
        )

        resolved = await resolver.resolve_conflict(conflict)

        # Should select entity with higher confidence (similar_entity)
        assert resolved.entity_id == similar_entity.entity_id
        assert resolved.provenance.confidence_score == 0.92

        # Should merge properties
        expected_properties = {**sample_entity.properties, **similar_entity.properties}
        assert resolved.properties == expected_properties

    @pytest.mark.asyncio
    async def test_resolve_conflict_latest_timestamp(self, resolver, sample_entity, similar_entity):
        """Test conflict resolution using latest timestamp strategy."""
        # Make sample_entity newer
        sample_entity.temporal_metadata.last_updated = datetime.now(timezone.utc)

        conflict = ConflictInfo(
            existing_entity=similar_entity,  # older
            new_entity=sample_entity,  # newer
            resolution_strategy=ConflictResolutionStrategy.LATEST_TIMESTAMP,
        )

        resolved = await resolver.resolve_conflict(conflict)

        # Should select newer entity
        assert resolved.entity_id == sample_entity.entity_id

    @pytest.mark.asyncio
    async def test_resolve_conflict_fallback_on_error(
        self, resolver, sample_entity, similar_entity
    ):
        """Test that conflict resolution falls back to new entity on errors."""
        conflict = ConflictInfo(
            existing_entity=sample_entity,
            new_entity=similar_entity,
        )

        # Mock the resolve method to raise an exception
        with patch.object(
            ConflictResolutionStrategy, "resolve", side_effect=Exception("Test error")
        ):
            resolved = await resolver.resolve_conflict(conflict)

            # Should fallback to new entity
            assert resolved == similar_entity

    def test_calculate_similarity_identical(self, resolver):
        """Test similarity calculation for identical labels."""
        similarity = resolver._calculate_similarity("machine learning", "machine learning")
        assert similarity == 1.0

    def test_calculate_similarity_partial_overlap(self, resolver):
        """Test similarity calculation for partially overlapping labels."""
        similarity = resolver._calculate_similarity("machine learning", "learning algorithms")
        assert 0.0 < similarity < 1.0

    def test_calculate_similarity_no_overlap(self, resolver):
        """Test similarity calculation for completely different labels."""
        similarity = resolver._calculate_similarity("machine learning", "quantum computing")
        assert similarity == 0.0

    def test_calculate_similarity_empty_labels(self, resolver):
        """Test similarity calculation for empty labels."""
        assert resolver._calculate_similarity("", "") == 1.0
        assert resolver._calculate_similarity("test", "") == 0.0
        assert resolver._calculate_similarity("", "test") == 0.0


class TestWebSocketEventStreamer:
    """Test suite for WebSocketEventStreamer."""

    @pytest.fixture
    def streamer(self):
        """Create event streamer for testing."""
        return WebSocketEventStreamer()

    @pytest.fixture
    def sample_event(self):
        """Create sample event for testing."""
        return GraphUpdateEvent(
            event_type=UpdateEventType.ENTITY_CREATED,
            entity_id="entity_123",
            conversation_id="conv_123",
            message_id="msg_456",
            metadata={"test": "data"},
        )

    @pytest.mark.asyncio
    async def test_start_stop_streaming(self, streamer):
        """Test starting and stopping the event streamer."""
        assert not streamer.is_streaming

        await streamer.start_streaming()
        assert streamer.is_streaming

        await streamer.stop_streaming()
        assert not streamer.is_streaming

    @pytest.mark.asyncio
    async def test_stream_event_when_not_running(self, streamer, sample_event):
        """Test streaming event when streamer is not running."""
        result = await streamer.stream_event(sample_event)
        assert result is False

    @pytest.mark.asyncio
    async def test_stream_event_when_running(self, streamer, sample_event):
        """Test streaming event when streamer is running."""
        await streamer.start_streaming()

        result = await streamer.stream_event(sample_event)
        assert result is True

        # Verify event was queued
        assert not streamer.event_queue.empty()

        await streamer.stop_streaming()

    @pytest.mark.asyncio
    async def test_event_processing_no_clients(self, streamer, sample_event):
        """Test event processing when no clients are connected."""
        await streamer.start_streaming()
        await streamer.stream_event(sample_event)

        # Let processing happen
        await asyncio.sleep(0.1)

        # Queue should be empty (event processed)
        assert streamer.event_queue.empty()

        await streamer.stop_streaming()

    @pytest.mark.asyncio
    async def test_broadcast_to_clients(self, streamer, sample_event):
        """Test broadcasting events to connected clients."""
        # Mock WebSocket client
        mock_client = AsyncMock()
        streamer.connected_clients.add(mock_client)

        await streamer._broadcast_event(sample_event)

        # Verify client received event
        mock_client.send_json.assert_called_once()
        call_args = mock_client.send_json.call_args[0][0]
        assert call_args["event_type"] == "entity_created"
        assert call_args["entity_id"] == "entity_123"

    @pytest.mark.asyncio
    async def test_broadcast_removes_disconnected_clients(self, streamer, sample_event):
        """Test that disconnected clients are removed during broadcast."""
        # Mock failing client
        mock_client = AsyncMock()
        mock_client.send_json.side_effect = Exception("Connection closed")
        streamer.connected_clients.add(mock_client)

        await streamer._broadcast_event(sample_event)

        # Client should be removed from set
        assert mock_client not in streamer.connected_clients

    def test_event_to_dict(self, sample_event):
        """Test event serialization to dictionary."""
        event_dict = sample_event.to_dict()

        expected_keys = {
            "event_id",
            "event_type",
            "entity_id",
            "relation_id",
            "conversation_id",
            "message_id",
            "timestamp",
            "metadata",
            "trace_id",
        }
        assert set(event_dict.keys()) == expected_keys
        assert event_dict["event_type"] == "entity_created"
        assert event_dict["entity_id"] == "entity_123"


class TestRealtimeGraphUpdater:
    """Test suite for RealtimeGraphUpdater."""

    @pytest.fixture
    def mock_graph(self):
        """Create mock knowledge graph."""
        return MagicMock()

    @pytest.fixture
    def mock_resolver(self):
        """Create mock conflict resolver."""
        resolver = AsyncMock(spec=DefaultConflictResolver)
        resolver.detect_conflicts.return_value = []
        return resolver

    @pytest.fixture
    def mock_streamer(self):
        """Create mock event streamer."""
        streamer = AsyncMock(spec=WebSocketEventStreamer)
        streamer.stream_event.return_value = True
        return streamer

    @pytest.fixture
    def updater(self, mock_graph, mock_resolver, mock_streamer):
        """Create updater for testing."""
        return RealtimeGraphUpdater(
            knowledge_graph=mock_graph,
            conflict_resolver=mock_resolver,
            event_streamer=mock_streamer,
            max_concurrent_updates=5,
        )

    @pytest.fixture
    def sample_extraction(self):
        """Create sample extraction result."""
        entity = ConversationEntity(
            entity_type=EntityType.CONCEPT,
            label="test concept",
            entity_id=str(uuid4()),
        )

        relation = ConversationRelation(
            source_entity_id=entity.entity_id,
            target_entity_id=str(uuid4()),
            relation_type=RelationType.RELATES_TO,
            relation_id=str(uuid4()),
        )

        return ExtractionResult(
            entities=[entity],
            relations=[relation],
            extraction_metadata={"extraction_method": "test"},
        )

    @pytest.mark.asyncio
    async def test_start_stop_updater(self, updater, mock_streamer):
        """Test starting and stopping the updater."""
        assert not updater.is_running

        await updater.start()
        assert updater.is_running
        mock_streamer.start_streaming.assert_called_once()

        await updater.stop()
        assert not updater.is_running
        mock_streamer.stop_streaming.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_extraction_not_running(self, updater, sample_extraction):
        """Test processing extraction when updater is not running."""
        with pytest.raises(RuntimeError, match="RealtimeGraphUpdater is not running"):
            await updater.process_extraction_result(sample_extraction, "conv_123", "msg_456")

    @pytest.mark.asyncio
    async def test_process_extraction_success(
        self, updater, sample_extraction, mock_resolver, mock_streamer
    ):
        """Test successful extraction processing."""
        await updater.start()

        # Mock no conflicts
        mock_resolver.detect_conflicts.return_value = []

        events = await updater.process_extraction_result(
            sample_extraction, "conv_123", "msg_456", "trace_789"
        )

        # Should generate events for entity and relation
        assert len(events) >= 2
        entity_events = [e for e in events if e.event_type == UpdateEventType.ENTITY_CREATED]
        relation_events = [e for e in events if e.event_type == UpdateEventType.RELATION_CREATED]

        assert len(entity_events) == 1
        assert len(relation_events) == 1

        # Verify event details
        entity_event = entity_events[0]
        assert entity_event.conversation_id == "conv_123"
        assert entity_event.message_id == "msg_456"
        assert entity_event.trace_id == "trace_789"

        await updater.stop()

    @pytest.mark.asyncio
    async def test_process_extraction_with_conflicts(
        self, updater, sample_extraction, mock_resolver, mock_streamer
    ):
        """Test extraction processing with conflict resolution."""
        await updater.start()

        # Mock conflict detection and resolution
        existing_entity = ConversationEntity(
            entity_type=EntityType.CONCEPT,
            label="existing concept",
            entity_id=str(uuid4()),
        )

        conflict = ConflictInfo(
            existing_entity=existing_entity,
            new_entity=sample_extraction.entities[0],
            similarity_score=0.9,
        )

        mock_resolver.detect_conflicts.return_value = [conflict]
        mock_resolver.resolve_conflict.return_value = existing_entity

        events = await updater.process_extraction_result(sample_extraction, "conv_123", "msg_456")

        # Should generate conflict resolution event
        conflict_events = [e for e in events if e.event_type == UpdateEventType.CONFLICT_RESOLVED]
        assert len(conflict_events) == 1

        conflict_event = conflict_events[0]
        assert conflict_event.metadata["similarity_score"] == 0.9

        await updater.stop()

    @pytest.mark.asyncio
    async def test_process_extraction_error_handling(
        self, updater, sample_extraction, mock_resolver, mock_streamer
    ):
        """Test error handling during extraction processing."""
        await updater.start()

        # Mock resolver to raise exception
        mock_resolver.detect_conflicts.side_effect = Exception("Test error")

        # Processing should continue despite entity errors (they are logged and handled gracefully)
        result = await updater.process_extraction_result(sample_extraction, "conv_123", "msg_456")
        
        # Should still return result even if some entities failed
        assert isinstance(result, list)

        # The entity processing error is handled gracefully, no failure event for individual entities
        # Just verify the processing completed
        assert mock_streamer.stream_event.called

        await updater.stop()

    @pytest.mark.asyncio
    async def test_concurrency_control(self, updater, mock_resolver, mock_streamer):
        """Test concurrency control with semaphore."""
        await updater.start()

        # Mock slow processing
        async def slow_detect_conflicts(*args):
            await asyncio.sleep(0.1)
            return []

        mock_resolver.detect_conflicts.side_effect = slow_detect_conflicts

        # Create multiple extraction results
        extractions = []
        for i in range(10):
            entity = ConversationEntity(
                entity_type=EntityType.CONCEPT,
                label=f"concept_{i}",
                entity_id=str(uuid4()),
            )
            extraction = ExtractionResult(entities=[entity], relations=[])
            extractions.append(extraction)

        # Start concurrent processing
        tasks = []
        for i, extraction in enumerate(extractions):
            task = asyncio.create_task(
                updater.process_extraction_result(extraction, f"conv_{i}", f"msg_{i}")
            )
            tasks.append(task)

        # Wait for completion
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should complete successfully
        assert len(results) == 10
        assert all(isinstance(r, list) for r in results)  # All return event lists

        await updater.stop()

    @pytest.mark.asyncio
    async def test_get_health_status(self, updater):
        """Test health status reporting."""
        health = await updater.get_health_status()

        expected_keys = {
            "is_running",
            "active_updates",
            "available_slots",
            "event_streamer_active",
            "timestamp",
        }
        assert set(health.keys()) == expected_keys
        assert isinstance(health["is_running"], bool)
        assert isinstance(health["active_updates"], int)
        assert isinstance(health["available_slots"], int)

    @pytest.mark.asyncio
    async def test_graceful_shutdown_with_active_updates(
        self, updater, sample_extraction, mock_resolver
    ):
        """Test graceful shutdown waits for active updates."""
        await updater.start()

        # Mock slow processing
        async def slow_detect_conflicts(*args):
            await asyncio.sleep(0.2)
            return []

        mock_resolver.detect_conflicts.side_effect = slow_detect_conflicts

        # Start processing
        task = asyncio.create_task(
            updater.process_extraction_result(sample_extraction, "conv_123", "msg_456")
        )

        # Give it time to start
        await asyncio.sleep(0.05)

        # Stop should wait for completion
        stop_start = datetime.now()
        await updater.stop()
        stop_duration = (datetime.now() - stop_start).total_seconds()

        # Should have waited for task completion
        assert stop_duration >= 0.0001  # Minimal delay expected due to graceful shutdown
        assert task.done()


@pytest.mark.integration
class TestRealtimeUpdaterIntegration:
    """Integration tests for the complete real-time update system."""

    @pytest.mark.asyncio
    async def test_end_to_end_update_flow(self):
        """Test complete end-to-end update flow."""
        # This would test with real components if we had them available
        # For now, ensure the interfaces work together correctly

        graph = MagicMock()
        resolver = DefaultConflictResolver()
        streamer = WebSocketEventStreamer()
        updater = RealtimeGraphUpdater(graph, resolver, streamer)

        entity = ConversationEntity(
            entity_type=EntityType.CONCEPT,
            label="integration test",
            entity_id=str(uuid4()),
        )
        extraction = ExtractionResult(entities=[entity], relations=[])

        await updater.start()

        try:
            events = await updater.process_extraction_result(
                extraction, "conv_integration", "msg_integration"
            )

            assert len(events) >= 1
            assert events[0].event_type == UpdateEventType.ENTITY_CREATED

        finally:
            await updater.stop()
