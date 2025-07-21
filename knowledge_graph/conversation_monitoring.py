"""
Conversation Monitoring System for Knowledge Graph Auto-Updates.
Monitors conversations in real-time and triggers knowledge graph updates
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional

from database.conversation_models import Conversation, Message
from knowledge_graph.graph_engine import KnowledgeGraph as GraphEngine

logger = logging.getLogger(__name__)


class ConversationEventType(Enum):
    """Types of conversation events."""

    MESSAGE_ADDED = "message_added"
    MESSAGE_UPDATED = "message_updated"
    MESSAGE_DELETED = "message_deleted"
    CONVERSATION_CREATED = "conversation_created"
    CONVERSATION_UPDATED = "conversation_updated"


@dataclass
class ConversationEvent:
    """Represents a conversation event."""

    type: ConversationEventType
    conversation_id: str
    message_id: Optional[str] = None
    content: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class ConversationMonitor:
    """Monitors conversations and triggers knowledge graph updates."""

    def __init__(self, graph_engine: GraphEngine):
        self.graph_engine = graph_engine
        self.is_running = False
        self.event_queue = Queue()
        self._subscribers: List[Callable] = []
        self._processing_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start monitoring conversations."""
        if self.is_running:
            logger.warning("ConversationMonitor is already running")
            return

        self.is_running = True
        logger.info("Starting ConversationMonitor")

        # Start background processing task
        self._processing_task = asyncio.create_task(self._process_events())

    async def stop(self):
        """Stop monitoring conversations."""
        if not self.is_running:
            return

        self.is_running = False
        logger.info("Stopping ConversationMonitor")

        # Cancel processing task
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

    async def process_message(self, message: Message):
        """Process a new message."""
        event = ConversationEvent(
            type=ConversationEventType.MESSAGE_ADDED,
            conversation_id=str(message.conversation_id),
            message_id=str(message.id),
            content=message.content,
            timestamp=message.created_at,
        )

        self.event_queue.put(event)
        logger.debug(f"Queued message event: {message.id}")

    async def process_batch(self) -> int:
        """Process a batch of events from the queue."""
        processed_count = 0
        batch_size = 10
        events = []

        # Collect events from queue
        while len(events) < batch_size and not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                events.append(event)
            except Empty:
                break

        # Process events
        for event in events:
            try:
                await self._process_event(event)
                processed_count += 1
            except Exception as e:
                logger.error(f"Error processing event: {e}")

        return processed_count

    async def _process_event(self, event: ConversationEvent):
        """Process a single event."""
        if event.type == ConversationEventType.MESSAGE_ADDED:
            # Trigger graph update for new message
            if event.content:
                await self._update_graph_from_message(event)

        # Notify subscribers
        await self.notify_subscribers(event)

    async def _update_graph_from_message(self, event: ConversationEvent):
        """Update knowledge graph from message content."""
        # This will be expanded to use NLP entity extraction
        logger.debug(f"Updating graph from message: {event.message_id}")

    async def on_conversation_update(self, conversation: Conversation, message: Message):
        """Handle conversation update event."""
        try:
            await self.graph_engine.update_from_conversation(conversation, message)
        except Exception as e:
            logger.error(f"Error updating graph from conversation: {e}")
            raise

    def subscribe(self, callback: Callable):
        """Subscribe to conversation events."""
        self._subscribers.append(callback)

    async def notify_subscribers(self, event: ConversationEvent = None):
        """Notify all subscribers of an event."""
        if event:
            # Process single event
            for callback in self._subscribers:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Error in subscriber callback: {e}")
        else:
            # Process all queued events
            while not self.event_queue.empty():
                try:
                    event = self.event_queue.get_nowait()
                    for callback in self._subscribers:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(event)
                        else:
                            callback(event)
                except Empty:
                    break
                except Exception as e:
                    logger.error(f"Error notifying subscribers: {e}")

    async def _process_events(self):
        """Background task to process events."""
        while self.is_running:
            try:
                # Process batch every second
                await asyncio.sleep(1.0)
                await self.process_batch()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
                await asyncio.sleep(5.0)  # Back off on error
