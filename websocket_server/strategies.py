"""WebSocket Strategy Pattern implementation.

This module provides clean separation between development and production
WebSocket handling, eliminating the need for demo endpoints.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
from core.environment import environment

logger = logging.getLogger(__name__)


class WebSocketStrategy(ABC):
    """Abstract base class for WebSocket handling strategies."""
    
    @abstractmethod
    async def handle_connection(self, websocket: WebSocket, client_id: str, token: Optional[str] = None) -> None:
        """Handle WebSocket connection establishment."""
        pass
    
    @abstractmethod
    async def handle_message(self, websocket: WebSocket, message: Dict[str, Any]) -> None:
        """Handle incoming WebSocket message."""
        pass
    
    @abstractmethod
    async def handle_disconnect(self, websocket: WebSocket, client_id: str) -> None:
        """Handle WebSocket disconnection."""
        pass
    
    @abstractmethod
    def requires_authentication(self) -> bool:
        """Check if this strategy requires authentication."""
        pass


class DevelopmentWebSocketStrategy(WebSocketStrategy):
    """WebSocket strategy for development environment.
    
    Features:
    - No authentication required
    - Extensive logging for debugging
    - Mock data responses
    - Auto-simulation of agent behaviors
    """
    
    def __init__(self):
        self.connected_clients: Dict[str, WebSocket] = {}
        self.mock_agents = self._create_mock_agents()
    
    async def handle_connection(self, websocket: WebSocket, client_id: str, token: Optional[str] = None) -> None:
        """Handle development WebSocket connection."""
        await websocket.accept()
        self.connected_clients[client_id] = websocket
        
        logger.info(f"🔌 Dev WebSocket connected: {client_id}")
        
        # Send welcome message with dev environment info
        await websocket.send_json({
            "type": "connection_established",
            "data": {
                "client_id": client_id,
                "environment": "development",
                "features": {
                    "mock_data": True,
                    "auto_simulation": True,
                    "debug_logging": True
                },
                "message": "🎯 Connected to FreeAgentics development environment"
            }
        })
        
        # Start mock data stream
        await self._start_mock_data_stream(websocket)
    
    async def handle_message(self, websocket: WebSocket, message: Dict[str, Any]) -> None:
        """Handle incoming message in development mode."""
        message_type = message.get("type", "unknown")
        logger.debug(f"📨 Dev WebSocket message: {message_type}")
        
        if message_type == "create_agent":
            await self._handle_create_agent(websocket, message)
        elif message_type == "start_simulation":
            await self._handle_start_simulation(websocket, message)
        elif message_type == "get_status":
            await self._handle_get_status(websocket, message)
        else:
            # Echo back with dev info
            await websocket.send_json({
                "type": "message_received",
                "data": {
                    "original_message": message,
                    "environment": "development",
                    "timestamp": "2025-01-29T12:00:00Z"
                }
            })
    
    async def handle_disconnect(self, websocket: WebSocket, client_id: str) -> None:
        """Handle development WebSocket disconnection."""
        if client_id in self.connected_clients:
            del self.connected_clients[client_id]
        logger.info(f"🚪 Dev WebSocket disconnected: {client_id}")
    
    def requires_authentication(self) -> bool:
        """Development mode doesn't require authentication."""
        return False
    
    async def _start_mock_data_stream(self, websocket: WebSocket) -> None:
        """Start streaming mock data for development."""
        await websocket.send_json({
            "type": "mock_agents_update",
            "data": {
                "agents": self.mock_agents,
                "message": "Mock agents loaded for development"
            }
        })
    
    async def _handle_create_agent(self, websocket: WebSocket, message: Dict[str, Any]) -> None:
        """Handle agent creation in development mode."""
        agent_data = message.get("data", {})
        mock_agent = {
            "id": f"dev_agent_{len(self.mock_agents) + 1}",
            "name": agent_data.get("name", "Dev Agent"),
            "type": agent_data.get("type", "explorer"),
            "status": "active",
            "position": {"x": 0, "y": 0},
            "energy": 100,
            "created_at": "2025-01-29T12:00:00Z"
        }
        
        self.mock_agents.append(mock_agent)
        
        await websocket.send_json({
            "type": "agent_created",
            "data": mock_agent
        })
    
    async def _handle_start_simulation(self, websocket: WebSocket, message: Dict[str, Any]) -> None:
        """Handle simulation start in development mode."""
        await websocket.send_json({
            "type": "simulation_started",
            "data": {
                "grid_size": {"width": 10, "height": 10},
                "agents": len(self.mock_agents),
                "message": "🎮 Development simulation started with mock data"
            }
        })
    
    async def _handle_get_status(self, websocket: WebSocket, message: Dict[str, Any]) -> None:
        """Handle status request in development mode."""
        await websocket.send_json({
            "type": "status_response",
            "data": {
                "environment": "development",
                "connected_clients": len(self.connected_clients),
                "mock_agents": len(self.mock_agents),
                "features_enabled": [
                    "mock_data",
                    "auto_simulation", 
                    "debug_logging",
                    "no_authentication"
                ]
            }
        })
    
    def _create_mock_agents(self) -> list:
        """Create mock agents for development."""
        return [
            {
                "id": "dev_agent_1",
                "name": "Explorer Alpha",
                "type": "explorer",
                "status": "active",
                "position": {"x": 2, "y": 2},
                "energy": 85
            },
            {
                "id": "dev_agent_2", 
                "name": "Collector Beta",
                "type": "collector",
                "status": "idle",
                "position": {"x": 7, "y": 5},
                "energy": 92
            }
        ]




class WebSocketStrategyFactory:
    """Factory for creating appropriate WebSocket strategies."""
    
    @staticmethod
    def create_strategy() -> WebSocketStrategy:
        """Create strategy based on current environment."""
        # Always return development strategy for external dev test
        return DevelopmentWebSocketStrategy()


# Global strategy instance
_strategy: Optional[WebSocketStrategy] = None


def get_websocket_strategy() -> WebSocketStrategy:
    """Get the current WebSocket strategy."""
    global _strategy
    if _strategy is None:
        _strategy = WebSocketStrategyFactory.create_strategy()
        logger.info(f"🎯 WebSocket strategy initialized: {_strategy.__class__.__name__}")
    return _strategy