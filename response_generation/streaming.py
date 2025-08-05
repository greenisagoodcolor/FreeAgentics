"""Response streaming implementations for real-time user feedback."""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .models import ResponseData, ResponseOptions

logger = logging.getLogger(__name__)


class ResponseStreamer(ABC):
    """Abstract base class for response streamers."""
    
    @abstractmethod
    async def stream_response(
        self,
        response_data: ResponseData,
        options: ResponseOptions,
    ) -> None:
        """Stream response data to clients.
        
        Args:
            response_data: Complete response data to stream
            options: Response options including streaming preferences
        """
        pass
    
    @abstractmethod
    async def stream_partial_response(
        self,
        partial_data: Dict[str, Any],
        response_id: str,
        options: ResponseOptions,
    ) -> None:
        """Stream partial response updates.
        
        Args:
            partial_data: Partial response data
            response_id: Unique response identifier
            options: Response options
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get streaming performance metrics."""
        pass


class WebSocketResponseStreamer(ResponseStreamer):
    """WebSocket-based response streamer for real-time updates.
    
    This implementation follows the established WebSocket patterns in the
    codebase, providing real-time streaming of response generation progress
    to connected clients.
    """
    
    def __init__(self, websocket_manager=None):
        """Initialize the WebSocket response streamer.
        
        Args:
            websocket_manager: WebSocket connection manager instance
        """
        self.websocket_manager = websocket_manager
        self._active_streams: Dict[str, Dict[str, Any]] = {}
        
        # Performance metrics
        self._metrics = {
            "streams_started": 0,
            "streams_completed": 0,
            "streams_failed": 0,
            "partial_updates_sent": 0,
            "messages_sent": 0,
            "avg_stream_duration_ms": 0.0,
        }
        
        logger.debug("WebSocketResponseStreamer initialized")
    
    async def stream_response(
        self,
        response_data: ResponseData,
        options: ResponseOptions,
    ) -> None:
        """Stream complete response data to WebSocket clients."""
        if not self.websocket_manager:
            logger.warning("WebSocket manager not available for streaming")
            return
        
        response_id = response_data.metadata.response_id
        self._metrics["streams_started"] += 1
        
        try:
            # Prepare streaming data
            stream_data = self._prepare_stream_data(response_data, options)
            
            # Send initial response structure
            await self._send_stream_message(
                message_type="response_start",
                data={
                    "response_id": response_id,
                    "conversation_id": options.conversation_id,
                    "trace_id": options.trace_id,
                    "response_type": response_data.response_type.value,
                },
                options=options,
            )
            
            # Stream response components progressively
            await self._stream_components(stream_data, response_id, options)
            
            # Send completion message
            await self._send_stream_message(
                message_type="response_complete",
                data={
                    "response_id": response_id,
                    "generation_time_ms": response_data.metadata.generation_time_ms,
                    "cached": response_data.metadata.cached,
                    "enhanced": response_data.metadata.nlg_enhanced,
                },
                options=options,
            )
            
            self._metrics["streams_completed"] += 1
            logger.debug(f"Response streaming completed: {response_id}")
            
        except Exception as e:
            logger.error(f"Response streaming failed: {e}")
            self._metrics["streams_failed"] += 1
            
            # Send error message to client
            try:
                await self._send_stream_message(
                    message_type="response_error",
                    data={
                        "response_id": response_id,
                        "error": str(e),
                    },
                    options=options,
                )
            except Exception as send_error:
                logger.error(f"Failed to send error message: {send_error}")
    
    async def stream_partial_response(
        self,
        partial_data: Dict[str, Any],
        response_id: str,
        options: ResponseOptions,
    ) -> None:
        """Stream partial response updates."""
        if not self.websocket_manager:
            return
        
        try:
            await self._send_stream_message(
                message_type="response_update",
                data={
                    "response_id": response_id,
                    "partial_data": partial_data,
                    "timestamp": partial_data.get("timestamp"),
                },
                options=options,
            )
            
            self._metrics["partial_updates_sent"] += 1
            
        except Exception as e:
            logger.error(f"Partial response streaming failed: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get streaming performance metrics."""
        metrics = self._metrics.copy()
        
        # Add computed metrics
        total_streams = metrics["streams_started"]
        if total_streams > 0:
            metrics["success_rate"] = metrics["streams_completed"] / total_streams
            metrics["failure_rate"] = metrics["streams_failed"] / total_streams
        else:
            metrics["success_rate"] = 0.0
            metrics["failure_rate"] = 0.0
        
        metrics["active_streams"] = len(self._active_streams)
        return metrics
    
    def _prepare_stream_data(
        self,
        response_data: ResponseData,
        options: ResponseOptions,
    ) -> List[Dict[str, Any]]:
        """Prepare response data for progressive streaming."""
        components = []
        
        # Component 1: Action explanation
        components.append({
            "type": "action_explanation",
            "data": response_data.action_explanation.to_dict(),
            "priority": 1,
        })
        
        # Component 2: Confidence rating
        components.append({
            "type": "confidence_rating", 
            "data": response_data.confidence_rating.to_dict(),
            "priority": 2,
        })
        
        # Component 3: Main message
        components.append({
            "type": "message",
            "data": {"message": response_data.message},
            "priority": 3,
        })
        
        # Component 4: Belief summary (if detailed)
        if options.include_technical_details:
            components.append({
                "type": "belief_summary",
                "data": response_data.belief_summary.to_dict(),
                "priority": 4,
            })
        
        # Component 5: Knowledge graph updates (if available)
        if response_data.knowledge_graph_updates:
            components.append({
                "type": "knowledge_graph_updates",
                "data": response_data.knowledge_graph_updates,
                "priority": 5,
            })
        
        # Component 6: Related concepts and suggestions
        if response_data.related_concepts or response_data.suggested_actions:
            components.append({
                "type": "enrichment_data",
                "data": {
                    "related_concepts": response_data.related_concepts,
                    "suggested_actions": response_data.suggested_actions,
                },
                "priority": 6,
            })
        
        return components
    
    async def _stream_components(
        self,
        components: List[Dict[str, Any]],
        response_id: str,
        options: ResponseOptions,
    ) -> None:
        """Stream response components progressively."""
        for i, component in enumerate(components):
            try:
                await self._send_stream_message(
                    message_type="response_component",
                    data={
                        "response_id": response_id,
                        "component_type": component["type"],
                        "component_data": component["data"],
                        "sequence": i + 1,
                        "total_components": len(components),
                    },
                    options=options,
                )
                
                # Small delay between components for better UX
                if i < len(components) - 1:  # Don't delay after last component
                    await asyncio.sleep(0.1)  # 100ms delay
                    
            except Exception as e:
                logger.error(f"Failed to stream component {component['type']}: {e}")
                # Continue with other components
    
    async def _send_stream_message(
        self,
        message_type: str,
        data: Dict[str, Any],
        options: ResponseOptions,
    ) -> None:
        """Send a streaming message via WebSocket."""
        if not self.websocket_manager:
            return
        
        message = {
            "type": message_type,
            "data": data,
            "timestamp": asyncio.get_event_loop().time(),
        }
        
        try:
            # In a real implementation, this would use the actual WebSocket manager
            # For now, we'll simulate sending to connected clients
            await self._simulate_websocket_send(message, options)
            
            self._metrics["messages_sent"] += 1
            
        except Exception as e:
            logger.error(f"WebSocket message send failed: {e}")
            raise
    
    async def _simulate_websocket_send(
        self,
        message: Dict[str, Any],
        options: ResponseOptions,
    ) -> None:
        """Simulate WebSocket message sending (placeholder for real implementation)."""
        # This is a placeholder for the actual WebSocket sending logic
        # In production, this would:
        # 1. Find connected clients for the conversation/user
        # 2. Send the message to each connected WebSocket
        # 3. Handle connection failures gracefully
        
        logger.debug(
            f"Simulating WebSocket send: {message['type']} "
            f"for conversation {options.conversation_id}"
        )
        
        # Simulate network delay
        await asyncio.sleep(0.01)  # 10ms simulated network delay


class NoOpStreamer(ResponseStreamer):
    """No-operation streamer for when streaming is disabled.
    
    This implementation provides a null object pattern for streaming,
    allowing the response generator to work without modification when
    streaming is disabled.
    """
    
    def __init__(self):
        """Initialize the no-op streamer.""" 
        self._metrics = {
            "streams_started": 0,
            "implementation": "noop",
        }
        
        logger.debug("NoOpStreamer initialized")
    
    async def stream_response(
        self,
        response_data: ResponseData,
        options: ResponseOptions,
    ) -> None:
        """No-op streaming - just log the attempt."""
        logger.debug(f"NoOp streaming for response {response_data.metadata.response_id}")
        self._metrics["streams_started"] += 1
    
    async def stream_partial_response(
        self,
        partial_data: Dict[str, Any],
        response_id: str,
        options: ResponseOptions,
    ) -> None:
        """No-op partial streaming."""
        logger.debug(f"NoOp partial streaming for response {response_id}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get no-op streamer metrics."""
        return self._metrics.copy()


class BufferedStreamer(ResponseStreamer):
    """Buffered streamer for testing and development.
    
    This implementation captures streaming data in memory buffers,
    useful for testing streaming behavior without actual WebSocket
    connections.
    """
    
    def __init__(self, buffer_size: int = 100):
        """Initialize the buffered streamer.
        
        Args:
            buffer_size: Maximum number of messages to buffer
        """
        self.buffer_size = buffer_size
        self._message_buffer: List[Dict[str, Any]] = []
        self._response_buffers: Dict[str, List[Dict[str, Any]]] = {}
        
        self._metrics = {
            "streams_started": 0,
            "streams_completed": 0,
            "messages_buffered": 0,
            "implementation": "buffered",
        }
        
        logger.debug(f"BufferedStreamer initialized with buffer_size={buffer_size}")
    
    async def stream_response(
        self,
        response_data: ResponseData,
        options: ResponseOptions,
    ) -> None:
        """Buffer complete response streaming."""
        response_id = response_data.metadata.response_id
        self._metrics["streams_started"] += 1
        
        # Initialize response buffer
        self._response_buffers[response_id] = []
        
        try:
            # Buffer start message
            await self._buffer_message(
                message_type="response_start",
                data={
                    "response_id": response_id,
                    "response_type": response_data.response_type.value,
                },
                response_id=response_id,
            )
            
            # Buffer main response data
            await self._buffer_message(
                message_type="response_data",
                data=response_data.to_dict(),
                response_id=response_id,
            )
            
            # Buffer completion message
            await self._buffer_message(
                message_type="response_complete",
                data={
                    "response_id": response_id,
                    "generation_time_ms": response_data.metadata.generation_time_ms,
                },
                response_id=response_id,
            )
            
            self._metrics["streams_completed"] += 1
            
        except Exception as e:
            logger.error(f"Buffered streaming failed: {e}")
            
            # Buffer error message
            await self._buffer_message(
                message_type="response_error", 
                data={
                    "response_id": response_id,
                    "error": str(e),
                },
                response_id=response_id,
            )
    
    async def stream_partial_response(
        self,
        partial_data: Dict[str, Any],
        response_id: str,
        options: ResponseOptions,
    ) -> None:
        """Buffer partial response updates."""
        await self._buffer_message(
            message_type="response_update",
            data={
                "response_id": response_id,
                "partial_data": partial_data,
            },
            response_id=response_id,
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get buffered streamer metrics."""
        metrics = self._metrics.copy()
        metrics["buffer_size"] = len(self._message_buffer)
        metrics["response_buffers"] = len(self._response_buffers)
        return metrics
    
    def get_messages(self, response_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get buffered messages.
        
        Args:
            response_id: If provided, get messages for specific response
            
        Returns:
            List of buffered messages
        """
        if response_id:
            return self._response_buffers.get(response_id, [])
        else:
            return self._message_buffer.copy()
    
    def clear_buffers(self) -> None:
        """Clear all message buffers."""
        self._message_buffer.clear()
        self._response_buffers.clear()
        logger.debug("Streaming buffers cleared")
    
    async def _buffer_message(
        self,
        message_type: str,
        data: Dict[str, Any],
        response_id: str,
    ) -> None:
        """Buffer a streaming message."""
        message = {
            "type": message_type,
            "data": data,
            "timestamp": asyncio.get_event_loop().time(),
            "response_id": response_id,
        }
        
        # Add to main buffer with size limit
        self._message_buffer.append(message)
        if len(self._message_buffer) > self.buffer_size:
            self._message_buffer.pop(0)  # Remove oldest
        
        # Add to response-specific buffer
        if response_id in self._response_buffers:
            self._response_buffers[response_id].append(message)
        
        self._metrics["messages_buffered"] += 1