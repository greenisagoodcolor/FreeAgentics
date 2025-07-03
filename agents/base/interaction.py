"""
Module for FreeAgentics Active Inference implementation.
"""

import logging
import queue
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .data_model import Agent, Position

"""
Agent Interaction System
This module provides mechanisms for agents to interact with other agents and the environment,
including communication protocols, resource exchange, and conflict resolution.
"""
logger = logging.getLogger(__name__)


class InteractionType(Enum):
    """Types of agent interactions"""

    COMMUNICATION = "communication"
    RESOURCE_EXCHANGE = "resource_exchange"
    INFORMATION_SHARING = "information_sharing"
    CONFLICT = "conflict"
    COOPERATION = "cooperation"
    COMPETITION = "competition"
    NEGOTIATION = "negotiation"
    SOCIAL = "social"


class MessageType(Enum):
    """Types of messages agents can send"""

    GREETING = "greeting"
    REQUEST = "request"
    OFFER = "offer"
    ACCEPT = "accept"
    REJECT = "reject"
    INFORM = "inform"
    QUERY = "query"
    COMMAND = "command"
    WARNING = "warning"
    ACKNOWLEDGMENT = "acknowledgment"
    TRADE_REQUEST = "trade_request"
    KNOWLEDGE_SHARE = "knowledge_share"
    SYSTEM_ALERT = "system_alert"


class ResourceType(Enum):
    """Types of resources that can be exchanged"""

    ENERGY = "energy"
    INFORMATION = "information"
    MATERIAL = "material"
    TERRITORY = "territory"
    SOCIAL_CAPITAL = "social_capital"


@dataclass
class Message:
    """Represents a message between agents"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    receiver_id: Optional[str] = None
    message_type: MessageType = MessageType.INFORM
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    requires_response: bool = False
    response_deadline: Optional[datetime] = None

    def is_broadcast(self) -> bool:
        """Check if this is a broadcast message"""
        return self.receiver_id is None


@dataclass
class InteractionRequest:
    """Request to initiate an interaction"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    initiator_id: str = ""
    target_id: str = ""
    interaction_type: InteractionType = InteractionType.COMMUNICATION
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    timeout: float = 30.0

    def is_expired(self) -> bool:
        """Check if the request has timed out"""
        return (datetime.now() - self.timestamp).total_seconds() > self.timeout


@dataclass
class InteractionResult:
    """Result of an interaction"""

    request_id: str = ""
    success: bool = False
    outcome: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None


@dataclass
class ResourceExchange:
    """Represents a resource exchange between agents"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    from_agent_id: str = ""
    to_agent_id: str = ""
    resource_type: ResourceType = ResourceType.ENERGY
    amount: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    completed: bool = False

    def execute(self, from_agent: Agent, to_agent: Agent) -> bool:
        """Execute the resource exchange"""
        if self.resource_type == ResourceType.ENERGY:
            if from_agent.resources.energy >= self.amount:
                from_agent.resources.energy -= self.amount
                to_agent.resources.energy += self.amount
                self.completed = True
                return True
        return False


class CommunicationProtocol:
    """Manages communication between agents"""

    def __init__(self) -> None:
        self.message_queue: queue.Queue = queue.Queue()
        self.conversation_history: Dict[tuple[str, str], List[Message]] = {}
        self.pending_responses: Dict[str, Message] = {}
        # Store broadcast messages separately
        self.broadcast_messages: List[Message] = []
        # Track which agents received broadcasts
        self.broadcast_received: Dict[str, set[str]] = {}
        self._lock = threading.Lock()

    def send_message(self, message: Message) -> bool:
        """Send a message to another agent or broadcast"""
        with self._lock:
            if message.is_broadcast():
                # Store broadcast messages separately
                self.broadcast_messages.append(message)
                self.broadcast_received[message.id] = set()
            else:
                self.message_queue.put(message)
                # Create conversation key safely, handling None values
                sender = message.sender_id or "unknown"
                receiver = message.receiver_id or "unknown"
                sorted_ids = sorted([sender, receiver])
                key = (sorted_ids[0], sorted_ids[1])
                if key not in self.conversation_history:
                    self.conversation_history[key] = []
                self.conversation_history[key].append(message)
            if message.requires_response:
                self.pending_responses[message.id] = message
            logger.debug(f"Message sent: {message.sender_id} -> {message.receiver_id or 'ALL'}")
            return True

    def receive_messages(self, agent_id: str, max_messages: int = 10) -> List[Message]:
        """Receive messages for a specific agent"""
        messages: List[Message] = []
        with self._lock:
            # First, collect any broadcast messages not yet received by this
            # agent
            for broadcast_msg in self.broadcast_messages:
                if agent_id not in self.broadcast_received.get(broadcast_msg.id, set()):
                    messages.append(broadcast_msg)
                    self.broadcast_received[broadcast_msg.id].add(agent_id)
                    if len(messages) >= max_messages:
                        break

            # Then collect direct messages
            temp_queue: queue.Queue[Message] = queue.Queue()
            while not self.message_queue.empty() and len(messages) < max_messages:
                try:
                    msg = self.message_queue.get_nowait()
                    if msg.receiver_id == agent_id:
                        messages.append(msg)
                    else:
                        temp_queue.put(msg)
                except queue.Empty:
                    break
            while not temp_queue.empty():
                self.message_queue.put(temp_queue.get())
        return messages

    def get_conversation_history(self, agent1_id: str, agent2_id: str) -> List[Message]:
        """Get conversation history between two agents"""
        with self._lock:
            sorted_ids = sorted([agent1_id, agent2_id])
            key: Tuple[str, str] = (sorted_ids[0], sorted_ids[1])
            return self.conversation_history.get(key, []).copy()

    def check_pending_responses(self) -> List[Message]:
        """Check for messages that need responses and have timed out"""
        with self._lock:
            timed_out = []
            for msg_id, msg in list(self.pending_responses.items()):
                if msg.response_deadline and datetime.now() > msg.response_deadline:
                    timed_out.append(msg)
                    del self.pending_responses[msg_id]
            return timed_out


class ResourceManager:
    """Manages resource exchanges between agents"""

    def __init__(self) -> None:
        self.pending_exchanges: Dict[str, ResourceExchange] = {}
        self.exchange_history: List[ResourceExchange] = []
        self._lock = threading.Lock()

    def propose_exchange(self, exchange: ResourceExchange) -> str:
        """Propose a resource exchange"""
        with self._lock:
            self.pending_exchanges[exchange.id] = exchange
            logger.debug(f"Exchange proposed: {exchange.from_agent_id} -> {exchange.to_agent_id}")
            return exchange.id

    def execute_exchange(self, exchange_id: str, from_agent: Agent, to_agent: Agent) -> bool:
        """Execute a pending exchange"""
        with self._lock:
            if exchange_id not in self.pending_exchanges:
                return False
            exchange = self.pending_exchanges[exchange_id]
            success = exchange.execute(from_agent, to_agent)
            if success:
                self.exchange_history.append(exchange)
                del self.pending_exchanges[exchange_id]
                logger.info(f"Exchange completed: {exchange.id}")
            return success

    def cancel_exchange(self, exchange_id: str) -> bool:
        """Cancel a pending exchange"""
        with self._lock:
            if exchange_id in self.pending_exchanges:
                del self.pending_exchanges[exchange_id]
                logger.debug(f"Exchange cancelled: {exchange_id}")
                return True
            return False

    def get_exchange_history(self, agent_id: str) -> List[ResourceExchange]:
        """Get exchange history for an agent"""
        with self._lock:
            return [
                ex
                for ex in self.exchange_history
                if ex.from_agent_id == agent_id or ex.to_agent_id == agent_id
            ]


class ConflictResolver:
    """Resolves conflicts between agents"""

    def __init__(self) -> None:
        self.conflict_handlers: Dict[str, Callable] = {}
        self.conflict_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def register_handler(self, conflict_type: str, handler: Callable) -> None:
        """Register a conflict resolution handler"""
        with self._lock:
            self.conflict_handlers[conflict_type] = handler

    def resolve_resource_conflict(
        self,
        agent1: Agent,
        agent2: Agent,
        resource_type: ResourceType,
        disputed_amount: float,
    ) -> Dict[str, float]:
        """Resolve a resource conflict between two agents"""
        total_priority = agent1.personality.openness + agent2.personality.openness
        if total_priority == 0:
            share1 = disputed_amount / 2
            share2 = disputed_amount / 2
        else:
            share1 = disputed_amount * (agent1.personality.openness / total_priority)
            share2 = disputed_amount * (agent2.personality.openness / total_priority)
        if resource_type == ResourceType.ENERGY:
            # Inverse energy weighting - agents with less energy get more
            total_energy = agent1.resources.energy + agent2.resources.energy + 0.001
            # Weight inversely by energy (those with less energy need more)
            weight1 = agent2.resources.energy / total_energy
            weight2 = agent1.resources.energy / total_energy
            # Normalize weights to ensure they sum to 1
            total_weight = weight1 + weight2
            if total_weight > 0:
                share1 = disputed_amount * (weight1 / total_weight)
                share2 = disputed_amount * (weight2 / total_weight)
        result = {agent1.agent_id: share1, agent2.agent_id: share2}
        with self._lock:
            self.conflict_history.append(
                {
                    "type": "resource_conflict",
                    "agents": [agent1.agent_id, agent2.agent_id],
                    "resource_type": resource_type,
                    "disputed_amount": disputed_amount,
                    "resolution": result,
                    "timestamp": datetime.now(),
                }
            )
        return result

    def resolve_spatial_conflict(
        self, agent1: Agent, agent2: Agent, disputed_position: Position
    ) -> str:
        """Resolve a spatial conflict (who gets to occupy a position)"""
        factors = {agent1.agent_id: 0, agent2.agent_id: 0}
        dist1 = agent1.position.distance_to(disputed_position)
        dist2 = agent2.position.distance_to(disputed_position)
        if dist1 < dist2:
            factors[agent1.agent_id] += 1
        else:
            factors[agent2.agent_id] += 1
        if agent1.personality.openness > agent2.personality.openness:
            factors[agent1.agent_id] += 1
        elif agent2.personality.openness > agent1.personality.openness:
            factors[agent2.agent_id] += 1
        if agent1.resources.energy > agent2.resources.energy:
            factors[agent1.agent_id] += 1
        else:
            factors[agent2.agent_id] += 1
        winner_id = max(factors.items(), key=lambda x: x[1])[0]
        with self._lock:
            self.conflict_history.append(
                {
                    "type": "spatial_conflict",
                    "agents": [agent1.agent_id, agent2.agent_id],
                    "disputed_position": disputed_position,
                    "winner": winner_id,
                    "factors": factors,
                    "timestamp": datetime.now(),
                }
            )
        return winner_id


class InteractionSystem:
    """Main system for managing all agent interactions"""

    def __init__(self) -> None:
        self.communication = CommunicationProtocol()
        self.resource_manager = ResourceManager()
        self.conflict_resolver = ConflictResolver()
        self.active_interactions: Dict[str, InteractionRequest] = {}
        self.interaction_history: List[InteractionResult] = []
        self.registered_agents: Dict[str, Agent] = {}
        self._lock = threading.Lock()
        self.interaction_callbacks: Dict[InteractionType, List[Callable]] = {
            interaction_type: [] for interaction_type in InteractionType
        }

    def register_agent(self, agent: Agent) -> None:
        """Register an agent with the interaction system"""
        with self._lock:
            self.registered_agents[agent.agent_id] = agent
            logger.info(f"Agent registered: {agent.agent_id}")

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the interaction system"""
        with self._lock:
            if agent_id in self.registered_agents:
                del self.registered_agents[agent_id]
                logger.info(f"Agent unregistered: {agent_id}")

    def register_interaction_callback(
        self, interaction_type: InteractionType, callback: Callable
    ) -> None:
        """Register a callback for specific interaction types"""
        with self._lock:
            self.interaction_callbacks[interaction_type].append(callback)

    def initiate_interaction(self, request: InteractionRequest) -> str:
        """Initiate an interaction between agents"""
        with self._lock:
            self.active_interactions[request.id] = request
            for callback in self.interaction_callbacks[request.interaction_type]:
                try:
                    callback(request)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
            logger.debug(f"Interaction initiated: {request.id}")
            return request.id

    def process_interaction(self, interaction_id: str) -> InteractionResult:
        """Process an active interaction"""
        with self._lock:
            if interaction_id not in self.active_interactions:
                return InteractionResult(
                    request_id=interaction_id,
                    success=False,
                    error_message="Interaction not found",
                )
            request = self.active_interactions[interaction_id]
            if request.is_expired():
                del self.active_interactions[interaction_id]
                return InteractionResult(
                    request_id=interaction_id,
                    success=False,
                    error_message="Interaction timed out",
                )
            result = self._process_interaction_type(request)
            self.interaction_history.append(result)
            del self.active_interactions[interaction_id]
            return result

    def _process_interaction_type(self, request: InteractionRequest) -> InteractionResult:
        """Process specific interaction types"""
        if request.interaction_type == InteractionType.COMMUNICATION:
            message = Message(
                sender_id=request.initiator_id,
                receiver_id=request.target_id,
                message_type=request.parameters.get("message_type", MessageType.INFORM),
                content=request.parameters.get("content", {}),
                requires_response=request.parameters.get("requires_response", False),
            )
            success = self.communication.send_message(message)
            return InteractionResult(
                request_id=request.id,
                success=success,
                outcome={"message_id": message.id},
            )
        elif request.interaction_type == InteractionType.RESOURCE_EXCHANGE:
            exchange = ResourceExchange(
                from_agent_id=request.initiator_id,
                to_agent_id=request.target_id,
                resource_type=request.parameters.get("resource_type", ResourceType.ENERGY),
                amount=request.parameters.get("amount", 0.0),
            )
            exchange_id = self.resource_manager.propose_exchange(exchange)
            return InteractionResult(
                request_id=request.id,
                success=True,
                outcome={"exchange_id": exchange_id},
            )
        elif request.interaction_type == InteractionType.CONFLICT:
            if request.initiator_id not in self.registered_agents:
                return InteractionResult(
                    request_id=request.id,
                    success=False,
                    error_message="Initiator not registered",
                )
            if request.target_id not in self.registered_agents:
                return InteractionResult(
                    request_id=request.id,
                    success=False,
                    error_message="Target not registered",
                )
            agent1 = self.registered_agents[request.initiator_id]
            agent2 = self.registered_agents[request.target_id]
            conflict_type = request.parameters.get("conflict_type", "resource")
            resolution: Dict[str, Any]
            if conflict_type == "resource":
                resolution = self.conflict_resolver.resolve_resource_conflict(
                    agent1,
                    agent2,
                    request.parameters.get("resource_type", ResourceType.ENERGY),
                    request.parameters.get("disputed_amount", 0.0),
                )
            elif conflict_type == "spatial":
                winner = self.conflict_resolver.resolve_spatial_conflict(
                    agent1,
                    agent2,
                    request.parameters.get("disputed_position", agent1.position),
                )
                resolution = {"winner": winner}
            else:
                resolution = {}
            return InteractionResult(request_id=request.id, success=True, outcome=resolution)
        else:
            return InteractionResult(
                request_id=request.id,
                success=True,
                outcome={"interaction_type": request.interaction_type.value},
            )

    def get_interaction_history(self, agent_id: str) -> List[InteractionResult]:
        """Get interaction history for an agent"""
        with self._lock:
            return [
                result
                for result in self.interaction_history
                if any(
                    req.initiator_id == agent_id or req.target_id == agent_id
                    for req in self.active_interactions.values()
                    if req.id == result.request_id
                )
            ]

    def cleanup_expired_interactions(self) -> int:
        """Clean up expired interactions"""
        with self._lock:
            expired = [
                req_id for req_id, req in self.active_interactions.items() if req.is_expired()
            ]
            for req_id in expired:
                result = InteractionResult(
                    request_id=req_id,
                    success=False,
                    error_message="Interaction expired",
                )
                self.interaction_history.append(result)
                del self.active_interactions[req_id]
            return len(expired)
