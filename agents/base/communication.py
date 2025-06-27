"""
Agent Conversation System.

Inter-agent communication with Active Inference goals.
"""

import json
import logging
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from knowledge.knowledge_graph import KnowledgeGraph


# Define BeliefNode locally since it's not available in the imported module
@dataclass
class BeliefNode:
    """A node representing a belief in the agent's knowledge graph"""

    id: str
    statement: str
    confidence: float
    supporting_patterns: List[str]
    contradicting_patterns: List[str]


logger = logging.getLogger(__name__)


class ConversationIntent(Enum):
    """Intent behind agent communication"""

    SHARE_DISCOVERY = "share_discovery"
    PROPOSE_TRADE = "propose_trade"
    FORM_ALLIANCE = "form_alliance"
    SEEK_INFORMATION = "seek_information"
    WARN_DANGER = "warn_danger"
    CASUAL_GREETING = "casual_greeting"


@dataclass
class ConversationMessage:
    """Single message in a conversation"""

    id: str
    sender_id: str
    recipient_id: Optional[str]  # None for broadcast
    content: str
    intent: ConversationIntent
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "content": self.content,
            "intent": self.intent.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ConversationTurn:
    """Represents a turn in the conversation"""

    agent_id: str
    action: str  # "speak", "listen", "think"
    message: Optional[ConversationMessage] = None
    internal_state: Dict[str, Any] = field(default_factory=dict)


class AgentConversation:
    """
    Manages goal-driven conversations between agents.

    Conversations are driven by Active Inference goals where
    agents communicate to reduce uncertainty and achieve objectives.
    """

    def __init__(self, conversation_id: Optional[str] = None, max_turns: int = 10) -> None:
        """
        Initialize a conversation.

        Args:
            conversation_id: Unique identifier for the conversation
            max_turns: Maximum number of turns before ending
        """
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.max_turns = max_turns
        self.participants: List[str] = []
        self.messages: List[ConversationMessage] = []
        self.turns: List[ConversationTurn] = []
        self.active = True
        self.start_time = datetime.utcnow()
        self.end_time: Optional[datetime] = None
        # Conversation state
        self.current_speaker: Optional[str] = None
        self.turn_count = 0
        # agent_id -> goals
        self.conversation_goals: Dict[str, List[str]] = {}
        logger.info(f"Created conversation {self.conversation_id}")

    def add_participant(self, agent_id: str, goals: Optional[List[str]] = None) -> None:
        """Add an agent to the conversation with their goals"""
        if agent_id not in self.participants:
            self.participants.append(agent_id)
            self.conversation_goals[agent_id] = goals or []
            logger.debug(f"Added participant {agent_id} to conversation")

    def generate_message(
        self,
        speaker_id: str,
        speaker_state: Dict[str, Any],
        conversation_context: List[ConversationMessage],
        llm_client: Optional[Any] = None,
    ) -> ConversationMessage:
        """

        Generate a message based on speaker's goals and state.
        Args:
            speaker_id: ID of the speaking agent
            speaker_state: Agent's current state including beliefs and goals
            conversation_context: Recent messages for context
            llm_client: LLM client for natural language generation
        Returns:
            Generated message
        """
        # Determine intent based on speaker's goals
        intent = self._determine_intent(speaker_state)
        # Create message metadata
        metadata = {
            "speaker_goals": self.conversation_goals.get(speaker_id, []),
            "free_energy": speaker_state.get("free_energy", 0),
            "confidence": speaker_state.get("confidence", 0.5),
        }
        # Generate content based on intent
        if llm_client:
            content = self._generate_with_llm(
                speaker_id, speaker_state, intent, conversation_context, llm_client
            )
        else:
            content = self._generate_template_message(speaker_id, speaker_state, intent)
        # Create message
        message = ConversationMessage(
            id=str(uuid.uuid4()),
            sender_id=speaker_id,
            recipient_id=self._select_recipient(speaker_id, intent),
            content=content,
            intent=intent,
            metadata=metadata,
        )
        return message

    def _determine_intent(self, speaker_state: Dict[str, Any]) -> ConversationIntent:
        """Determine conversation intent based on agent state"""
        # Check for urgent needs
        if speaker_state.get("danger_detected", False):
            return ConversationIntent.WARN_DANGER
        # Check resources
        resources = speaker_state.get("resources", {})
        if any(amount < 20 for amount in resources.values()):
            return ConversationIntent.PROPOSE_TRADE
        # Check for discoveries
        recent_discoveries = speaker_state.get("recent_discoveries", [])
        if recent_discoveries:
            return ConversationIntent.SHARE_DISCOVERY
        # Check uncertainty
        uncertainty = speaker_state.get("uncertainty", 0)
        if uncertainty > 0.7:
            return ConversationIntent.SEEK_INFORMATION
        # Check for cooperation needs
        if speaker_state.get("seeking_allies", False):
            return ConversationIntent.FORM_ALLIANCE
        # Default to casual
        return ConversationIntent.CASUAL_GREETING

    def _select_recipient(self, speaker_id: str, intent: ConversationIntent) -> Optional[str]:
        """Select recipient based on intent"""
        other_participants = [p for p in self.participants if p != speaker_id]
        if not other_participants:
            return None
        # For warnings, broadcast to all
        if intent == ConversationIntent.WARN_DANGER:
            return None  # Broadcast
        # For now, select first other participant
        # In full implementation, would use more sophisticated selection
        return other_participants[0]

    def _generate_template_message(
        self, speaker_id: str, speaker_state: Dict[str, Any], intent: ConversationIntent
    ) -> str:
        """Generate message using templates"""
        templates = {
            ConversationIntent.SHARE_DISCOVERY: [
                "I've discovered something interesting: {discovery}",
                "You might want to know about {discovery}",
                "I found {discovery} at {location}",
            ],
            ConversationIntent.PROPOSE_TRADE: [
                "I need {need}. Would you trade for {offer}?",
                "Looking to exchange {offer} for {need}",
                "Anyone interested in trading? I have {offer}",
            ],
            ConversationIntent.FORM_ALLIANCE: [
                "We should work together",
                "Want to form an alliance?",
                "Together we could achieve more",
            ],
            ConversationIntent.SEEK_INFORMATION: [
                "Do you know anything about {topic}?",
                "I'm looking for information on {topic}",
                "Has anyone seen {topic}?",
            ],
            ConversationIntent.WARN_DANGER: [
                "Warning! {danger} detected at {location}!",
                "Everyone be careful - {danger}!",
                "Danger alert: {danger}",
            ],
            ConversationIntent.CASUAL_GREETING: [
                "Hello there!",
                "How's everyone doing?",
                "Nice to meet you all",
            ],
        }
        # Select template
        template = random.choice(templates.get(intent, ["Hello"]))
        # Fill in template
        if intent == ConversationIntent.SHARE_DISCOVERY:
            discoveries = speaker_state.get("recent_discoveries", ["something"])
            discovery = discoveries[0] if discoveries else "something"
            location = speaker_state.get("location", "nearby")
            return template.format(discovery=discovery, location=location)
        elif intent == ConversationIntent.PROPOSE_TRADE:
            resources = speaker_state.get("resources", {})
            # Find what's needed and what can be offered
            need = min(resources.items(), key=lambda x: x[1])[0] if resources else "resources"
            offer = max(resources.items(), key=lambda x: x[1])[0] if resources else "items"
            return template.format(need=need, offer=offer)
        elif intent == ConversationIntent.SEEK_INFORMATION:
            topic = speaker_state.get("uncertainty_topics", ["the area"])[0]
            return template.format(topic=topic)
        elif intent == ConversationIntent.WARN_DANGER:
            danger = speaker_state.get("danger_type", "threat")
            location = speaker_state.get("danger_location", "nearby")
            return template.format(danger=danger, location=location)
        else:
            return template

    def _generate_with_llm(
        self,
        speaker_id: str,
        speaker_state: Dict[str, Any],
        intent: ConversationIntent,
        conversation_context: List[ConversationMessage],
        llm_client: Any,
    ) -> str:
        """Generate natural language using LLM"""
        # Build prompt
        prompt = self._build_llm_prompt(speaker_id, speaker_state, intent, conversation_context)
        # Generate response
        try:
            response = llm_client.generate(prompt, max_tokens=100)
            return str(response).strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Fall back to template
            return self._generate_template_message(speaker_id, speaker_state, intent)

    def _build_llm_prompt(
        self,
        speaker_id: str,
        speaker_state: Dict[str, Any],
        intent: ConversationIntent,
        conversation_context: List[ConversationMessage],
    ) -> str:
        """Build prompt for LLM generation"""
        # Agent personality
        personality = speaker_state.get("personality", {})
        traits = ", ".join([f"{k}: {v}" for k, v in personality.items()])
        # Recent context
        context_str = ""
        for msg in conversation_context[-3:]:  # Last 3 messages
            context_str += f"{msg.sender_id}: {msg.content}\n"
        prompt = f"""You are agent {speaker_id} with personality traits: {traits}.
Your current intent is: {intent.value}
Your goals: {', '.join(self.conversation_goals.get(speaker_id, []))}
Recent conversation:
{context_str}
Generate a natural response that:
1. Matches your personality
2. Achieves your intent ({intent.value})
3. Is concise (1-2 sentences)
4. Continues the conversation naturally
Response:"""
        return prompt

    def process_turn(
        self, agent_id: str, agent_state: Dict[str, Any], llm_client: Optional[Any] = None
    ) -> ConversationTurn:
        """
        Process a conversation turn for an agent.
        Args:
            agent_id: ID of the agent taking turn
            agent_state: Agent's current state
            llm_client: Optional LLM client
        Returns:
            ConversationTurn with action taken
        """
        if not self.active:
            raise ValueError("Conversation has ended")
        # Determine action (speak, listen, think)
        action = self._determine_action(agent_id, agent_state)
        turn = ConversationTurn(
            agent_id=agent_id,
            action=action,
            internal_state={"free_energy": agent_state.get("free_energy", 0)},
        )
        if action == "speak":
            # Generate and add message
            message = self.generate_message(
                agent_id,
                agent_state,
                self.messages[-5:],  # Last 5 messages for context
                llm_client,
            )
            self.add_message(message)
            turn.message = message
            self.current_speaker = agent_id
        elif action == "listen":
            # Record listening state
            turn.internal_state["listening_to"] = self.current_speaker
        self.turns.append(turn)
        self.turn_count += 1
        # Check if conversation should end
        if self.turn_count >= self.max_turns:
            self.end_conversation()
        return turn

    def _determine_action(self, agent_id: str, agent_state: Dict[str, Any]) -> str:
        """Determine what action agent should take"""
        # Simple turn-taking logic
        # In full implementation, would use more sophisticated decision-making
        # If no one has spoken yet, speak
        if not self.messages:
            return "speak"
        # If agent just spoke, listen
        if self.messages and self.messages[-1].sender_id == agent_id:
            return "listen"
        # If agent has something urgent, speak
        if agent_state.get("urgent_message", False):
            return "speak"
        # Otherwise, probabilistically decide
        return random.choice(["speak", "listen", "think"])

    def add_message(self, message: ConversationMessage) -> None:
        """Add a message to the conversation"""
        self.messages.append(message)
        logger.debug(f"Added message from {message.sender_id}: {message.content[:50]}...")

    def update_beliefs_from_message(
        self,
        recipient_id: str,
        message: ConversationMessage,
        knowledge_graph: KnowledgeGraph,
    ) -> List[BeliefNode]:
        """
        Update recipient's beliefs based on received message.
        Args:
            recipient_id: ID of the recipient agent
            message: Received message
            knowledge_graph: Recipient's knowledge graph
        Returns:
            List of updated or new beliefs
        """
        updated_beliefs = []
        # Extract information based on intent
        if message.intent == ConversationIntent.SHARE_DISCOVERY:
            # Create belief about shared discovery
            belief = BeliefNode(
                id=str(uuid.uuid4()),
                statement=(f"Agent {message.sender_id} discovered: {message.content}"),
                confidence=0.7,  # Moderate confidence in shared info
                supporting_patterns=[],
                contradicting_patterns=[],
            )
            # Add to knowledge graph
            knowledge_graph.add_belief(
                belief.statement,
                belief.confidence,
                metadata={"type": "belief", "sender": message.sender_id},
            )
            updated_beliefs.append(belief)
        elif message.intent == ConversationIntent.WARN_DANGER:
            # Create high-confidence belief about danger
            belief = BeliefNode(
                id=str(uuid.uuid4()),
                statement=f"Danger warning: {message.content}",
                confidence=0.9,  # High confidence in warnings
                supporting_patterns=[],
                contradicting_patterns=[],
            )
            # Add to knowledge graph
            knowledge_graph.add_belief(
                belief.statement,
                belief.confidence,
                metadata={"type": "belief", "sender": message.sender_id},
            )
            updated_beliefs.append(belief)
        elif message.intent == ConversationIntent.PROPOSE_TRADE:
            # Create belief about trade opportunity
            belief = BeliefNode(
                id=str(uuid.uuid4()),
                statement=(f"Trade opportunity with {message.sender_id}: {message.content}"),
                confidence=0.6,
                supporting_patterns=[],
                contradicting_patterns=[],
            )
            # Add to knowledge graph
            knowledge_graph.add_belief(
                belief.statement,
                belief.confidence,
                metadata={"type": "belief", "sender": message.sender_id},
            )
            updated_beliefs.append(belief)
        # Update trust/relationship beliefs
        trust_belief = BeliefNode(
            id=str(uuid.uuid4()),
            statement=(
                f"Agent {message.sender_id} communicated with intent {message.intent.value}"
            ),
            confidence=0.8,
            supporting_patterns=[],
            contradicting_patterns=[],
        )
        # Add to knowledge graph
        knowledge_graph.add_belief(
            trust_belief.statement,
            trust_belief.confidence,
            metadata={
                "type": "belief",
                "sender": message.sender_id,
            },
        )
        updated_beliefs.append(trust_belief)
        return updated_beliefs

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of the conversation"""
        intent_counts: Dict[str, int] = {}
        for msg in self.messages:
            intent = msg.intent.value
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        return {
            "conversation_id": self.conversation_id,
            "participants": self.participants,
            "message_count": len(self.messages),
            "turn_count": self.turn_count,
            "duration": ((self.end_time or datetime.utcnow()) - self.start_time).total_seconds(),
            "active": self.active,
            "intent_distribution": intent_counts,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }

    def end_conversation(self):
        """End the conversation"""
        self.active = False
        self.end_time = datetime.utcnow()
        logger.info(f"Ended conversation {self.conversation_id}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation to dictionary"""

        return {
            "conversation_id": self.conversation_id,
            "participants": self.participants,
            "messages": [msg.to_dict() for msg in self.messages],
            "summary": self.get_conversation_summary(),
        }


class ConversationManager:
    """Manages multiple conversations between agents"""

    def __init__(self) -> None:
        self.conversations: Dict[str, AgentConversation] = {}
        self.agent_conversations: Dict[str, List[str]] = {}  # agent_id -> conv_ids

    def create_conversation(
        self, participants: List[str], goals: Optional[Dict[str, List[str]]] = None
    ) -> AgentConversation:
        """
        Create a new conversation.
        Args:
            participants: List of agent IDs
            goals: Optional mapping of agent_id -> goals
        Returns:
            Created conversation
        """
        conversation = AgentConversation()
        for agent_id in participants:
            agent_goals = goals.get(agent_id, []) if goals else []
            conversation.add_participant(agent_id, agent_goals)
            # Track agent's conversations
            if agent_id not in self.agent_conversations:
                self.agent_conversations[agent_id] = []
            self.agent_conversations[agent_id].append(conversation.conversation_id)
        self.conversations[conversation.conversation_id] = conversation
        return conversation

    def get_agent_communications(self, agent_id: str) -> List[AgentConversation]:
        """Get all conversations for an agent"""
        conv_ids = self.agent_conversations.get(agent_id, [])
        return [
            self.conversations[conv_id] for conv_id in conv_ids if conv_id in self.conversations
        ]

    def get_active_conversations(self) -> List[AgentConversation]:
        """Get all active conversations"""

        return [conv for conv in self.conversations.values() if conv.active]


class CommunicationCapability:
    """
    Communication capability for agents.
    This class provides agents with the ability to send and receive messages,
    participate in conversations, and manage communication settings.
    """

    def __init__(
        self,
        message_system,
        agent_id: str,
        communication_range: float = 5.0,
        bandwidth: int = 10,
        protocols: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize communication capability.
        Args:
            message_system: The message system instance for sending/receiving
            agent_id: ID of the agent this capability belongs to
            communication_range: Maximum distance for communication
            bandwidth: Maximum messages per cycle
            protocols: Supported communication protocols
        """

        self.message_system = message_system
        self.agent_id = agent_id
        self.communication_range = communication_range
        self.bandwidth = bandwidth
        self.protocols = protocols or ["direct", "broadcast"]
        # Communication state
        self.active_conversations: Dict[str, AgentConversation] = {}
        self.message_queue: List[ConversationMessage] = []
        self.sent_messages_count = 0
        self.received_messages_count = 0
        # Register with message system
        if hasattr(message_system, "register_agent"):
            message_system.register_agent(agent_id, self)

    def send_message(
        self,
        recipient_id: Optional[str],
        content: str,
        intent: ConversationIntent = ConversationIntent.CASUAL_GREETING,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Send a message to another agent or broadcast.
        Args:
            recipient_id: ID of recipient (None for broadcast)
            content: Message content
            intent: Communication intent
            metadata: Optional message metadata
        Returns:
            True if message was sent successfully
        """
        if self.sent_messages_count >= self.bandwidth:
            logger.warning(f"Agent {self.agent_id} exceeded bandwidth limit")
            return False
        message = ConversationMessage(
            id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            content=content,
            intent=intent,
            metadata=metadata or {},
        )
        # Send through message system
        if hasattr(self.message_system, "send_message"):
            success = self.message_system.send_message(message)
            if success:
                self.sent_messages_count += 1
            return bool(success)
        else:
            # Fallback: add to queue
            self.message_queue.append(message)
            self.sent_messages_count += 1
            return True

    def receive_message(self, message: ConversationMessage) -> bool:
        """
        Receive a message from another agent.
        Args:
            message: Received message
        Returns:
            True if message was processed successfully
        """

        # Check if within communication range (simplified)
        # In full implementation, would check actual positions
        self.message_queue.append(message)
        self.received_messages_count += 1
        # Find or create conversation
        conversation_id = self._get_conversation_id(message.sender_id)
        if conversation_id not in self.active_conversations:
            conversation = AgentConversation()
            conversation.add_participant(self.agent_id)
            conversation.add_participant(message.sender_id)
            self.active_conversations[conversation_id] = conversation
        # Add message to conversation
        self.active_conversations[conversation_id].add_message(message)
        return True

    def start_conversation(
        self, target_agent_id: str, goals: Optional[List[str]] = None
    ) -> Optional[AgentConversation]:
        """
        Start a new conversation with another agent.
        Args:
            target_agent_id: ID of the target agent
            goals: Optional conversation goals
        Returns:
            Created conversation or None if failed
        """

        conversation_id = self._get_conversation_id(target_agent_id)
        if conversation_id in self.active_conversations:
            return self.active_conversations[conversation_id]
        conversation = AgentConversation()
        conversation.add_participant(self.agent_id, goals or [])
        conversation.add_participant(target_agent_id)
        self.active_conversations[conversation_id] = conversation
        return conversation

    def get_pending_messages(self) -> List[ConversationMessage]:
        """Get all pending messages in the queue"""
        messages = self.message_queue.copy()
        self.message_queue.clear()
        return messages

    def get_active_conversations(self) -> List[AgentConversation]:
        """Get all active conversations"""
        return list(self.active_conversations.values())

    def end_conversation(self, conversation_id: str) -> bool:
        """
        End a conversation.
        Args:
            conversation_id: ID of conversation to end
        Returns:
            True if conversation was ended successfully
        """
        if conversation_id in self.active_conversations:
            self.active_conversations[conversation_id].end_conversation()
            del self.active_conversations[conversation_id]
            return True
        return False

    def reset_cycle(self) -> None:
        """Reset per-cycle counters (called at start of each simulation
        cycle)"""
        self.sent_messages_count = 0
        self.received_messages_count = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get communication statistics"""
        return {
            "agent_id": self.agent_id,
            "communication_range": self.communication_range,
            "bandwidth": self.bandwidth,
            "active_conversations": len(self.active_conversations),
            "messages_sent": self.sent_messages_count,
            "messages_received": self.received_messages_count,
            "pending_messages": len(self.message_queue),
        }

    def _get_conversation_id(self, other_agent_id: str) -> str:
        """Generate a consistent conversation ID for two agents"""
        agents = sorted([self.agent_id, other_agent_id])
        return f"conv_{agents[0]}_{agents[1]}"

    def __repr__(self) -> str:
        return (
            f"CommunicationCapability(agent={self.agent_id}, "
            f"range={self.communication_range}, "
            f"conversations={len(self.active_conversations)})"
        )


# Example usage
if __name__ == "__main__":
    # Create conversation manager
    manager = ConversationManager()
    # Create a conversation between two agents
    conversation = manager.create_conversation(
        participants=["agent_1", "agent_2"],
        goals={
            "agent_1": ["find_food", "share_knowledge"],
            "agent_2": ["find_water", "form_alliance"],
        },
    )
    # Simulate conversation turns
    agent1_state = {
        "resources": {"food": 80, "water": 20},
        "recent_discoveries": ["berry_bush_location"],
        "personality": {"openness": 0.8, "agreeableness": 0.7},
    }
    agent2_state = {
        "resources": {"food": 30, "water": 70},
        "seeking_allies": True,
        "personality": {"openness": 0.6, "agreeableness": 0.8},
    }
    # Process turns
    for i in range(6):
        if i % 2 == 0:
            turn = conversation.process_turn("agent_1", agent1_state)
        else:
            turn = conversation.process_turn("agent_2", agent2_state)
        if turn.message:
            print(f"{turn.agent_id}: {turn.message.content}")
    # Get summary
    summary = conversation.get_conversation_summary()
    print(f"\nConversation summary: {json.dumps(summary, indent=2)}")
