"""
Comprehensive tests for Agent Communication System.

Tests inter-agent communication with Active Inference goals,
ensuring proper PyMDP alignment and GNN notation support.
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest

from agents.base.communication import (
    AgentConversation,
    BeliefNode,
    ConversationIntent,
    ConversationMessage,
    ConversationTurn,
)


class TestBeliefNode:
    """Test BeliefNode dataclass functionality."""

    def test_belief_node_creation(self):
        """Test creating a belief node."""
        node = BeliefNode(
            id="belief_1",
            statement="The market is bullish",
            confidence=0.8,
            supporting_patterns=["pattern1", "pattern2"],
            contradicting_patterns=["pattern3"],
        )

        assert node.id == "belief_1"
        assert node.statement == "The market is bullish"
        assert node.confidence == 0.8
        assert node.supporting_patterns == ["pattern1", "pattern2"]
        assert node.contradicting_patterns == ["pattern3"]

    def test_belief_node_confidence_bounds(self):
        """Test belief node with various confidence levels."""
        # Test high confidence
        high_conf = BeliefNode("id1", "statement", 0.95, [], [])
        assert high_conf.confidence == 0.95

        # Test low confidence
        low_conf = BeliefNode("id2", "statement", 0.1, [], [])
        assert low_conf.confidence == 0.1

        # Test certainty
        certain = BeliefNode("id3", "statement", 1.0, [], [])
        assert certain.confidence == 1.0

    def test_belief_node_patterns(self):
        """Test belief node pattern handling."""
        node = BeliefNode(
            id="test",
            statement="Test statement",
            confidence=0.7,
            supporting_patterns=["pattern_a", "pattern_b", "pattern_c"],
            contradicting_patterns=["contra_1", "contra_2"],
        )

        assert len(node.supporting_patterns) == 3
        assert len(node.contradicting_patterns) == 2
        assert "pattern_a" in node.supporting_patterns
        assert "contra_1" in node.contradicting_patterns


class TestConversationIntent:
    """Test ConversationIntent enum functionality."""

    def test_all_intents_defined(self):
        """Test that all expected intents are defined."""
        expected_intents = [
            "SHARE_DISCOVERY",
            "PROPOSE_TRADE",
            "FORM_ALLIANCE",
            "SEEK_INFORMATION",
            "WARN_DANGER",
            "CASUAL_GREETING",
        ]

        for intent_name in expected_intents:
            assert hasattr(ConversationIntent, intent_name)

    def test_intent_values(self):
        """Test intent enum values."""
        assert ConversationIntent.SHARE_DISCOVERY.value == "share_discovery"
        assert ConversationIntent.PROPOSE_TRADE.value == "propose_trade"
        assert ConversationIntent.FORM_ALLIANCE.value == "form_alliance"
        assert ConversationIntent.SEEK_INFORMATION.value == "seek_information"
        assert ConversationIntent.WARN_DANGER.value == "warn_danger"
        assert ConversationIntent.CASUAL_GREETING.value == "casual_greeting"

    def test_intent_iteration(self):
        """Test iterating over all intents."""
        all_intents = list(ConversationIntent)
        assert len(all_intents) == 6

        intent_values = [intent.value for intent in all_intents]
        assert "share_discovery" in intent_values
        assert "propose_trade" in intent_values


class TestConversationMessage:
    """Test ConversationMessage dataclass functionality."""

    def test_message_creation_basic(self):
        """Test creating a basic conversation message."""
        message = ConversationMessage(
            id="msg_1",
            sender_id="agent_1",
            recipient_id="agent_2",
            content="Hello, how are you?",
            intent=ConversationIntent.CASUAL_GREETING,
        )

        assert message.id == "msg_1"
        assert message.sender_id == "agent_1"
        assert message.recipient_id == "agent_2"
        assert message.content == "Hello, how are you?"
        assert message.intent == ConversationIntent.CASUAL_GREETING
        assert message.metadata == {}
        assert isinstance(message.timestamp, datetime)

    def test_message_creation_broadcast(self):
        """Test creating a broadcast message (no specific recipient)."""
        message = ConversationMessage(
            id="broadcast_1",
            sender_id="agent_1",
            recipient_id=None,  # Broadcast
            content="Danger approaching from the north!",
            intent=ConversationIntent.WARN_DANGER,
        )

        assert message.recipient_id is None
        assert message.intent == ConversationIntent.WARN_DANGER

    def test_message_with_metadata(self):
        """Test message with custom metadata."""
        metadata = {"priority": "high", "location": {"x": 10, "y": 20}, "confidence": 0.95}

        message = ConversationMessage(
            id="msg_meta",
            sender_id="agent_1",
            recipient_id="agent_2",
            content="I found valuable resources",
            intent=ConversationIntent.SHARE_DISCOVERY,
            metadata=metadata,
        )

        assert message.metadata == metadata
        assert message.metadata["priority"] == "high"
        assert message.metadata["confidence"] == 0.95

    def test_message_to_dict(self):
        """Test converting message to dictionary."""
        # Use datetime.utcnow() to match the implementation
        timestamp = datetime.utcnow()

        message = ConversationMessage(
            id="dict_test",
            sender_id="sender",
            recipient_id="recipient",
            content="Test content",
            intent=ConversationIntent.PROPOSE_TRADE,
            metadata={"key": "value"},
            timestamp=timestamp,
        )

        result_dict = message.to_dict()

        expected = {
            "id": "dict_test",
            "sender_id": "sender",
            "recipient_id": "recipient",
            "content": "Test content",
            "intent": "propose_trade",
            "metadata": {"key": "value"},
            "timestamp": timestamp.isoformat(),
        }

        assert result_dict == expected

    def test_message_serialization(self):
        """Test that message can be JSON serialized."""
        message = ConversationMessage(
            id="serialize_test",
            sender_id="agent_1",
            recipient_id="agent_2",
            content="JSON test",
            intent=ConversationIntent.SEEK_INFORMATION,
        )

        message_dict = message.to_dict()
        json_str = json.dumps(message_dict)

        # Should not raise exception
        assert isinstance(json_str, str)
        assert "serialize_test" in json_str
        assert "seek_information" in json_str


class TestConversationTurn:
    """Test ConversationTurn dataclass functionality."""

    def test_turn_creation_speak(self):
        """Test creating a speaking turn."""
        message = ConversationMessage(
            id="turn_msg",
            sender_id="agent_1",
            recipient_id="agent_2",
            content="Hello",
            intent=ConversationIntent.CASUAL_GREETING,
        )

        turn = ConversationTurn(agent_id="agent_1", action="speak", message=message)

        assert turn.agent_id == "agent_1"
        assert turn.action == "speak"
        assert turn.message == message
        assert turn.internal_state == {}

    def test_turn_creation_listen(self):
        """Test creating a listening turn."""
        turn = ConversationTurn(agent_id="agent_2", action="listen", message=None)

        assert turn.agent_id == "agent_2"
        assert turn.action == "listen"
        assert turn.message is None
        assert turn.internal_state == {}

    def test_turn_creation_think(self):
        """Test creating a thinking turn with internal state."""
        internal_state = {"current_belief": 0.8, "uncertainty": 0.2, "expected_free_energy": -1.5}

        turn = ConversationTurn(
            agent_id="agent_3", action="think", message=None, internal_state=internal_state
        )

        assert turn.action == "think"
        assert turn.internal_state == internal_state
        assert turn.internal_state["expected_free_energy"] == -1.5

    def test_turn_actions(self):
        """Test various turn actions."""
        actions = ["speak", "listen", "think", "process", "analyze"]

        for action in actions:
            turn = ConversationTurn(agent_id="test_agent", action=action)
            assert turn.action == action


class TestAgentConversation:
    """Test AgentConversation class functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.conversation = AgentConversation()

    def test_conversation_initialization_default(self):
        """Test conversation initialization with defaults."""
        conv = AgentConversation()

        assert conv.conversation_id is not None
        assert conv.max_turns == 10
        assert conv.participants == []
        assert conv.messages == []
        assert conv.turns == []
        assert isinstance(conv.conversation_id, str)

    def test_conversation_initialization_custom(self):
        """Test conversation initialization with custom values."""
        conv_id = "custom_conversation_123"
        conv = AgentConversation(conversation_id=conv_id, max_turns=20)

        assert conv.conversation_id == conv_id
        assert conv.max_turns == 20
        assert conv.participants == []
        assert conv.messages == []
        assert conv.turns == []


class TestConversationWorkflows:
    """Test complete conversation workflows."""

    def test_simple_two_agent_conversation(self):
        """Test a simple two-agent conversation workflow."""
        conv = AgentConversation(conversation_id="simple_chat")

        # Add participants
        conv.participants.extend(["alice", "bob"])

        # Alice greets Bob
        greeting = ConversationMessage(
            id="msg_1",
            sender_id="alice",
            recipient_id="bob",
            content="Hello Bob!",
            intent=ConversationIntent.CASUAL_GREETING,
        )
        conv.messages.append(greeting)

        alice_turn1 = ConversationTurn(agent_id="alice", action="speak", message=greeting)
        bob_turn1 = ConversationTurn(agent_id="bob", action="listen")
        conv.turns.extend([alice_turn1, bob_turn1])

        # Bob responds
        response = ConversationMessage(
            id="msg_2",
            sender_id="bob",
            recipient_id="alice",
            content="Hi Alice! How are you?",
            intent=ConversationIntent.CASUAL_GREETING,
        )
        conv.messages.append(response)

        bob_turn2 = ConversationTurn(agent_id="bob", action="speak", message=response)
        alice_turn2 = ConversationTurn(agent_id="alice", action="listen")
        conv.turns.extend([bob_turn2, alice_turn2])

        # Verify conversation state
        assert len(conv.participants) == 2
        assert len(conv.messages) == 2
        assert len(conv.turns) == 4
        assert conv.messages[0].sender_id == "alice"
        assert conv.messages[1].sender_id == "bob"

    def test_information_seeking_conversation(self):
        """Test conversation focused on information seeking."""
        conv = AgentConversation(conversation_id="info_seek")
        conv.participants.extend(["explorer", "merchant"])

        # Explorer seeks information about trade routes
        info_request = ConversationMessage(
            id="info_req",
            sender_id="explorer",
            recipient_id="merchant",
            content="Do you know any good trade routes to the eastern markets?",
            intent=ConversationIntent.SEEK_INFORMATION,
            metadata={"topic": "trade_routes", "region": "eastern"},
        )
        conv.messages.append(info_request)

        # Merchant shares discovery
        info_response = ConversationMessage(
            id="info_resp",
            sender_id="merchant",
            recipient_id="explorer",
            content="Yes, I discovered a safe route through the mountains.",
            intent=ConversationIntent.SHARE_DISCOVERY,
            metadata={"route_safety": 0.9, "estimated_time": "3_days"},
        )
        conv.messages.append(info_response)

        assert len(conv.messages) == 2
        assert conv.messages[0].intent == ConversationIntent.SEEK_INFORMATION
        assert conv.messages[1].intent == ConversationIntent.SHARE_DISCOVERY
        assert conv.messages[0].metadata["topic"] == "trade_routes"
        assert conv.messages[1].metadata["route_safety"] == 0.9


class TestActiveInferenceIntegration:
    """Test integration with Active Inference concepts."""

    def test_belief_driven_communication(self):
        """Test communication driven by belief states."""
        conv = AgentConversation(conversation_id="belief_comm")

        # Agent communicates based on high confidence belief
        belief_message = ConversationMessage(
            id="belief_msg",
            sender_id="scholar",
            recipient_id="explorer",
            content="I'm highly confident there are valuable minerals in the northern region.",
            intent=ConversationIntent.SHARE_DISCOVERY,
            metadata={
                "belief_confidence": 0.95,
                "supporting_evidence": ["geological_survey", "historical_records"],
                "expected_free_energy_reduction": -2.3,
            },
        )
        conv.messages.append(belief_message)

        # Corresponding turn with PyMDP-aligned internal state
        belief_turn = ConversationTurn(
            agent_id="scholar",
            action="speak",
            message=belief_message,
            internal_state={
                "current_beliefs": [0.05, 0.1, 0.8, 0.05],  # Belief distribution
                "policy_preferences": [0.2, 0.6, 0.2],  # Policy probabilities
                "free_energy": -1.8,  # Current free energy
                "epistemic_value": 0.7,  # Information gain expectation
            },
        )
        conv.turns.append(belief_turn)

        assert belief_message.metadata["belief_confidence"] == 0.95
        assert belief_turn.internal_state["free_energy"] == -1.8
        assert sum(belief_turn.internal_state["current_beliefs"]) == 1.0


class TestGNNNotationSupport:
    """Test GNN (Generalized Notation Notation) support in communication."""

    def test_gnn_formalized_communication(self):
        """Test communication with GNN notation metadata."""
        conv = AgentConversation(conversation_id="gnn_comm")

        # Message with GNN notation for belief updates
        gnn_message = ConversationMessage(
            id="gnn_msg",
            sender_id="notation_expert",
            recipient_id="agent_collective",
            content="Updating belief dynamics according to GNN formalism.",
            intent=ConversationIntent.SHARE_DISCOVERY,
            metadata={
                "notation_system": "GNN",
                "formalism_version": "1.0",
                "belief_update_notation": "q(s|o) ∝ P(o|s)q(s)",
                "free_energy_notation": "F = D_KL[q(s)||P(s)] - E_q[ln P(o|s)]",
                "policy_notation": "π* = argmin_π G(π)",
                "mathematical_validation": True,
            },
        )
        conv.messages.append(gnn_message)

        assert gnn_message.metadata["notation_system"] == "GNN"
        assert "q(s|o)" in gnn_message.metadata["belief_update_notation"]
        assert gnn_message.metadata["mathematical_validation"] is True


class TestAgentConversationMethods:
    """Test AgentConversation methods comprehensively."""

    def test_add_participant(self):
        """Test adding participants to conversation."""
        conversation = AgentConversation()

        conversation.add_participant("agent_1", ["goal1", "goal2"])
        conversation.add_participant("agent_2")

        assert "agent_1" in conversation.participants
        assert "agent_2" in conversation.participants
        assert conversation.conversation_goals["agent_1"] == ["goal1", "goal2"]
        assert conversation.conversation_goals["agent_2"] == []

    def test_add_participant_duplicate(self):
        """Test adding the same participant twice."""
        conversation = AgentConversation()

        conversation.add_participant("agent_1", ["goal1"])
        conversation.add_participant("agent_1", ["goal2"])  # Should not duplicate

        assert conversation.participants.count("agent_1") == 1
        assert conversation.conversation_goals["agent_1"] == ["goal1"]  # Original goals kept

    def test_determine_intent_danger(self):
        """Test intent determination for danger situations."""
        conversation = AgentConversation()
        speaker_state = {"danger_detected": True}

        intent = conversation._determine_intent(speaker_state)

        assert intent == ConversationIntent.WARN_DANGER

    def test_determine_intent_trade(self):
        """Test intent determination for trade situations."""
        conversation = AgentConversation()
        speaker_state = {"resources": {"water": 15, "food": 25}}

        intent = conversation._determine_intent(speaker_state)

        assert intent == ConversationIntent.PROPOSE_TRADE

    def test_determine_intent_discovery(self):
        """Test intent determination for discovery sharing."""
        conversation = AgentConversation()
        speaker_state = {"recent_discoveries": ["water_source"]}

        intent = conversation._determine_intent(speaker_state)

        assert intent == ConversationIntent.SHARE_DISCOVERY

    def test_determine_intent_information_seeking(self):
        """Test intent determination for information seeking."""
        conversation = AgentConversation()
        speaker_state = {"uncertainty": 0.8}

        intent = conversation._determine_intent(speaker_state)

        assert intent == ConversationIntent.SEEK_INFORMATION

    def test_determine_intent_alliance(self):
        """Test intent determination for alliance formation."""
        conversation = AgentConversation()
        speaker_state = {"seeking_allies": True}

        intent = conversation._determine_intent(speaker_state)

        assert intent == ConversationIntent.FORM_ALLIANCE

    def test_determine_intent_casual(self):
        """Test intent determination for casual greeting."""
        conversation = AgentConversation()
        speaker_state = {}

        intent = conversation._determine_intent(speaker_state)

        assert intent == ConversationIntent.CASUAL_GREETING

    def test_select_recipient_warning(self):
        """Test recipient selection for warnings (broadcast)."""
        conversation = AgentConversation()
        conversation.add_participant("agent_1")
        conversation.add_participant("agent_2")

        recipient = conversation._select_recipient("agent_1", ConversationIntent.WARN_DANGER)

        assert recipient is None  # Broadcast

    def test_select_recipient_normal(self):
        """Test recipient selection for normal conversation."""
        conversation = AgentConversation()
        conversation.add_participant("agent_1")
        conversation.add_participant("agent_2")

        recipient = conversation._select_recipient("agent_1", ConversationIntent.CASUAL_GREETING)

        assert recipient == "agent_2"

    def test_select_recipient_no_others(self):
        """Test recipient selection when no other participants."""
        conversation = AgentConversation()
        conversation.add_participant("agent_1")

        recipient = conversation._select_recipient("agent_1", ConversationIntent.CASUAL_GREETING)

        assert recipient is None

    def test_generate_template_message_discovery(self):
        """Test template message generation for discovery sharing."""
        conversation = AgentConversation()
        speaker_state = {"recent_discoveries": ["water_source"], "location": "hex_123"}

        message = conversation._generate_template_message(
            "agent_1", speaker_state, ConversationIntent.SHARE_DISCOVERY
        )

        assert "water_source" in message
        # Location may or may not be included depending on random template choice
        # Just verify the message is not empty and contains the discovery

    def test_generate_template_message_trade(self):
        """Test template message generation for trade proposals."""
        conversation = AgentConversation()
        speaker_state = {"resources": {"water": 10, "food": 50}}

        message = conversation._generate_template_message(
            "agent_1", speaker_state, ConversationIntent.PROPOSE_TRADE
        )

        assert "water" in message or "food" in message

    def test_generate_template_message_information(self):
        """Test template message generation for information seeking."""
        conversation = AgentConversation()
        speaker_state = {"uncertainty_topics": ["forest_area"]}

        message = conversation._generate_template_message(
            "agent_1", speaker_state, ConversationIntent.SEEK_INFORMATION
        )

        assert "forest_area" in message

    def test_generate_template_message_danger(self):
        """Test template message generation for danger warnings."""
        conversation = AgentConversation()
        speaker_state = {"danger_type": "predator", "danger_location": "hex_456"}

        message = conversation._generate_template_message(
            "agent_1", speaker_state, ConversationIntent.WARN_DANGER
        )

        assert "predator" in message
        # Location may or may not be included depending on random template choice
        # Just verify the message is not empty and contains the danger type

    def test_generate_template_message_casual(self):
        """Test template message generation for casual greetings."""
        conversation = AgentConversation()
        speaker_state = {}

        message = conversation._generate_template_message(
            "agent_1", speaker_state, ConversationIntent.CASUAL_GREETING
        )

        assert len(message) > 0
        assert message in ["Hello there!", "How's everyone doing?", "Nice to meet you all"]

    def test_build_llm_prompt(self):
        """Test building LLM prompt for message generation."""
        conversation = AgentConversation()
        conversation.add_participant("agent_1", ["find_water", "explore"])

        speaker_state = {"personality": {"openness": 0.8, "friendliness": 0.7}}

        context_messages = [
            ConversationMessage(
                id="msg_1",
                sender_id="agent_2",
                recipient_id="agent_1",
                content="Hello!",
                intent=ConversationIntent.CASUAL_GREETING,
            )
        ]

        prompt = conversation._build_llm_prompt(
            "agent_1", speaker_state, ConversationIntent.SHARE_DISCOVERY, context_messages
        )

        assert "agent_1" in prompt
        assert "openness: 0.8" in prompt
        assert "friendliness: 0.7" in prompt
        assert "share_discovery" in prompt
        assert "find_water, explore" in prompt
        assert "agent_2: Hello!" in prompt

    @patch("agents.base.communication.logger")
    def test_generate_with_llm_success(self, mock_logger):
        """Test successful LLM message generation."""
        conversation = AgentConversation()

        mock_llm = Mock()
        mock_llm.generate.return_value = "I found a water source nearby!"

        result = conversation._generate_with_llm(
            "agent_1", {}, ConversationIntent.SHARE_DISCOVERY, [], mock_llm
        )

        assert result == "I found a water source nearby!"
        mock_llm.generate.assert_called_once()

    @patch("agents.base.communication.logger")
    def test_generate_with_llm_failure(self, mock_logger):
        """Test LLM message generation with fallback."""
        conversation = AgentConversation()

        mock_llm = Mock()
        mock_llm.generate.side_effect = Exception("LLM error")

        speaker_state = {"recent_discoveries": ["water"]}

        result = conversation._generate_with_llm(
            "agent_1", speaker_state, ConversationIntent.SHARE_DISCOVERY, [], mock_llm
        )

        # Should fall back to template message
        assert "water" in result
        mock_logger.error.assert_called_once()

    def test_generate_message_with_llm(self):
        """Test generating message with LLM client."""
        conversation = AgentConversation()
        conversation.add_participant("agent_1", ["explore"])
        conversation.add_participant("agent_2")

        mock_llm = Mock()
        mock_llm.generate.return_value = "Hello there, friend!"

        speaker_state = {"free_energy": 0.3, "confidence": 0.8}

        message = conversation.generate_message("agent_1", speaker_state, [], mock_llm)

        assert message.sender_id == "agent_1"
        assert message.content == "Hello there, friend!"
        assert message.metadata["free_energy"] == 0.3
        assert message.metadata["confidence"] == 0.8
        assert message.metadata["speaker_goals"] == ["explore"]

    def test_generate_message_without_llm(self):
        """Test generating message without LLM client."""
        conversation = AgentConversation()
        conversation.add_participant("agent_1")
        conversation.add_participant("agent_2")

        speaker_state = {"recent_discoveries": ["berries"]}

        message = conversation.generate_message("agent_1", speaker_state, [])

        assert message.sender_id == "agent_1"
        assert message.recipient_id == "agent_2"
        assert message.intent == ConversationIntent.SHARE_DISCOVERY
        assert "berries" in message.content

    def test_determine_action_first_message(self):
        """Test action determination when no messages exist."""
        conversation = AgentConversation()

        action = conversation._determine_action("agent_1", {})

        assert action == "speak"

    def test_determine_action_just_spoke(self):
        """Test action determination after agent just spoke."""
        conversation = AgentConversation()
        conversation.add_message(
            ConversationMessage(
                id="msg_1",
                sender_id="agent_1",
                recipient_id="agent_2",
                content="Hello",
                intent=ConversationIntent.CASUAL_GREETING,
            )
        )

        action = conversation._determine_action("agent_1", {})

        assert action == "listen"

    def test_determine_action_urgent(self):
        """Test action determination with urgent message."""
        conversation = AgentConversation()
        conversation.add_message(
            ConversationMessage(
                id="msg_1",
                sender_id="agent_2",
                recipient_id="agent_1",
                content="Hello",
                intent=ConversationIntent.CASUAL_GREETING,
            )
        )

        action = conversation._determine_action("agent_1", {"urgent_message": True})

        assert action == "speak"

    @patch("random.choice")
    def test_determine_action_random(self, mock_choice):
        """Test random action determination."""
        mock_choice.return_value = "think"

        conversation = AgentConversation()
        conversation.add_message(
            ConversationMessage(
                id="msg_1",
                sender_id="agent_2",
                recipient_id="agent_1",
                content="Hello",
                intent=ConversationIntent.CASUAL_GREETING,
            )
        )

        action = conversation._determine_action("agent_1", {})

        assert action == "think"

    def test_process_turn_speak(self):
        """Test processing a speaking turn."""
        conversation = AgentConversation()
        conversation.add_participant("agent_1")
        conversation.add_participant("agent_2")

        agent_state = {"free_energy": 0.4}

        turn = conversation.process_turn("agent_1", agent_state)

        assert turn.agent_id == "agent_1"
        assert turn.action == "speak"
        assert turn.message is not None
        assert turn.internal_state["free_energy"] == 0.4
        assert conversation.current_speaker == "agent_1"
        assert conversation.turn_count == 1
        assert len(conversation.messages) == 1
        assert len(conversation.turns) == 1

    def test_process_turn_listen(self):
        """Test processing a listening turn."""
        conversation = AgentConversation()
        conversation.add_participant("agent_1")
        conversation.add_participant("agent_2")

        # First agent speaks
        conversation.process_turn("agent_1", {})

        # Second agent listens
        turn = conversation.process_turn("agent_1", {"free_energy": 0.6})

        assert turn.agent_id == "agent_1"
        assert turn.action == "listen"
        assert turn.message is None
        assert turn.internal_state["listening_to"] == "agent_1"
        assert conversation.turn_count == 2

    def test_process_turn_max_turns(self):
        """Test conversation ending after max turns."""
        conversation = AgentConversation(max_turns=2)
        conversation.add_participant("agent_1")
        conversation.add_participant("agent_2")

        # First turn
        conversation.process_turn("agent_1", {})
        assert conversation.active is True

        # Second turn - should end conversation
        conversation.process_turn("agent_2", {})
        assert conversation.active is False
        assert conversation.end_time is not None

    def test_process_turn_ended_conversation(self):
        """Test processing turn on ended conversation."""
        conversation = AgentConversation()
        conversation.end_conversation()

        with pytest.raises(ValueError, match="Conversation has ended"):
            conversation.process_turn("agent_1", {})

    def test_add_message(self):
        """Test adding messages to conversation."""
        conversation = AgentConversation()

        message = ConversationMessage(
            id="msg_1",
            sender_id="agent_1",
            recipient_id="agent_2",
            content="Test message",
            intent=ConversationIntent.CASUAL_GREETING,
        )

        conversation.add_message(message)

        assert len(conversation.messages) == 1
        assert conversation.messages[0] == message

    def test_update_beliefs_from_discovery_message(self):
        """Test belief updating from discovery message."""
        conversation = AgentConversation()

        mock_kg = Mock()
        mock_kg.add_belief = Mock()

        message = ConversationMessage(
            id="msg_1",
            sender_id="agent_1",
            recipient_id="agent_2",
            content="I discovered a water source",
            intent=ConversationIntent.SHARE_DISCOVERY,
        )

        beliefs = conversation.update_beliefs_from_message("agent_2", message, mock_kg)

        assert len(beliefs) == 2  # Discovery belief + trust belief

        discovery_belief = beliefs[0]
        assert "Agent agent_1 discovered" in discovery_belief.statement
        assert discovery_belief.confidence == 0.7

        trust_belief = beliefs[1]
        assert "Agent agent_1 communicated with intent share_discovery" in trust_belief.statement
        assert trust_belief.confidence == 0.8

        # Verify knowledge graph calls
        assert mock_kg.add_belief.call_count == 2

    def test_update_beliefs_from_danger_message(self):
        """Test belief updating from danger warning."""
        conversation = AgentConversation()

        mock_kg = Mock()
        mock_kg.add_belief = Mock()

        message = ConversationMessage(
            id="msg_1",
            sender_id="agent_1",
            recipient_id="agent_2",
            content="Warning! Predator ahead!",
            intent=ConversationIntent.WARN_DANGER,
        )

        beliefs = conversation.update_beliefs_from_message("agent_2", message, mock_kg)

        assert len(beliefs) == 2

        danger_belief = beliefs[0]
        assert "Danger warning" in danger_belief.statement
        assert danger_belief.confidence == 0.9  # High confidence for warnings

    def test_update_beliefs_from_trade_message(self):
        """Test belief updating from trade proposal."""
        conversation = AgentConversation()

        mock_kg = Mock()
        mock_kg.add_belief = Mock()

        message = ConversationMessage(
            id="msg_1",
            sender_id="agent_1",
            recipient_id="agent_2",
            content="Want to trade water for food?",
            intent=ConversationIntent.PROPOSE_TRADE,
        )

        beliefs = conversation.update_beliefs_from_message("agent_2", message, mock_kg)

        assert len(beliefs) == 2

        trade_belief = beliefs[0]
        assert "Trade opportunity with agent_1" in trade_belief.statement
        assert trade_belief.confidence == 0.6

    def test_get_conversation_summary(self):
        """Test getting conversation summary."""
        conversation = AgentConversation()
        conversation.add_participant("agent_1")
        conversation.add_participant("agent_2")

        # Add some messages with different intents
        conversation.add_message(
            ConversationMessage(
                id="msg_1",
                sender_id="agent_1",
                recipient_id="agent_2",
                content="Hello",
                intent=ConversationIntent.CASUAL_GREETING,
            )
        )
        conversation.add_message(
            ConversationMessage(
                id="msg_2",
                sender_id="agent_2",
                recipient_id="agent_1",
                content="I found water",
                intent=ConversationIntent.SHARE_DISCOVERY,
            )
        )

        conversation.turn_count = 3

        summary = conversation.get_conversation_summary()

        assert summary["conversation_id"] == conversation.conversation_id
        assert summary["participants"] == ["agent_1", "agent_2"]
        assert summary["message_count"] == 2
        assert summary["turn_count"] == 3
        assert summary["active"] is True
        assert summary["intent_distribution"]["casual_greeting"] == 1
        assert summary["intent_distribution"]["share_discovery"] == 1
        assert "start_time" in summary
        assert summary["end_time"] is None

    def test_end_conversation(self):
        """Test ending a conversation."""
        conversation = AgentConversation()

        assert conversation.active is True
        assert conversation.end_time is None

        conversation.end_conversation()

        assert conversation.active is False
        assert conversation.end_time is not None

    def test_to_dict(self):
        """Test converting conversation to dictionary."""
        conversation = AgentConversation()
        conversation.add_participant("agent_1")
        conversation.add_participant("agent_2")

        conversation.add_message(
            ConversationMessage(
                id="msg_1",
                sender_id="agent_1",
                recipient_id="agent_2",
                content="Hello",
                intent=ConversationIntent.CASUAL_GREETING,
            )
        )

        conv_dict = conversation.to_dict()

        assert conv_dict["conversation_id"] == conversation.conversation_id
        assert conv_dict["participants"] == ["agent_1", "agent_2"]
        assert len(conv_dict["messages"]) == 1
        assert conv_dict["messages"][0]["content"] == "Hello"
        assert "summary" in conv_dict


class TestConversationManager:
    """Test ConversationManager class functionality."""

    def test_conversation_manager_creation(self):
        """Test creating a conversation manager."""
        from agents.base.communication import ConversationManager

        manager = ConversationManager()

        assert manager.conversations == {}
        assert manager.agent_conversations == {}

    def test_create_conversation(self):
        """Test creating a conversation through manager."""
        from agents.base.communication import ConversationManager

        manager = ConversationManager()

        participants = ["agent_1", "agent_2"]
        goals = {"agent_1": ["explore"], "agent_2": ["trade"]}

        conversation = manager.create_conversation(participants, goals)

        assert conversation.conversation_id in manager.conversations
        assert "agent_1" in conversation.participants
        assert "agent_2" in conversation.participants
        assert conversation.conversation_goals["agent_1"] == ["explore"]
        assert conversation.conversation_goals["agent_2"] == ["trade"]

        # Check agent conversation tracking
        assert conversation.conversation_id in manager.agent_conversations["agent_1"]
        assert conversation.conversation_id in manager.agent_conversations["agent_2"]

    def test_create_conversation_no_goals(self):
        """Test creating conversation without goals."""
        from agents.base.communication import ConversationManager

        manager = ConversationManager()

        conversation = manager.create_conversation(["agent_1", "agent_2"])

        assert conversation.conversation_goals["agent_1"] == []
        assert conversation.conversation_goals["agent_2"] == []

    def test_get_agent_communications(self):
        """Test getting conversations for an agent."""
        from agents.base.communication import ConversationManager

        manager = ConversationManager()

        # Create multiple conversations
        conv1 = manager.create_conversation(["agent_1", "agent_2"])
        conv2 = manager.create_conversation(["agent_1", "agent_3"])
        conv3 = manager.create_conversation(["agent_2", "agent_3"])

        agent1_convs = manager.get_agent_communications("agent_1")

        assert len(agent1_convs) == 2
        assert conv1 in agent1_convs
        assert conv2 in agent1_convs
        assert conv3 not in agent1_convs

    def test_get_agent_communications_no_conversations(self):
        """Test getting conversations for agent with no conversations."""
        from agents.base.communication import ConversationManager

        manager = ConversationManager()

        convs = manager.get_agent_communications("agent_1")

        assert convs == []

    def test_get_active_conversations(self):
        """Test getting all active conversations."""
        from agents.base.communication import ConversationManager

        manager = ConversationManager()

        conv1 = manager.create_conversation(["agent_1", "agent_2"])
        conv2 = manager.create_conversation(["agent_3", "agent_4"])

        # End one conversation
        conv2.end_conversation()

        active_convs = manager.get_active_conversations()

        assert len(active_convs) == 1
        assert conv1 in active_convs
        assert conv2 not in active_convs


class TestCommunicationCapability:
    """Test CommunicationCapability class functionality."""

    def test_communication_capability_creation(self):
        """Test creating a communication capability."""
        from agents.base.communication import CommunicationCapability

        mock_message_system = Mock()

        capability = CommunicationCapability(
            message_system=mock_message_system,
            agent_id="agent_1",
            communication_range=10.0,
            bandwidth=15,
            protocols=["direct", "broadcast", "relay"],
        )

        assert capability.message_system == mock_message_system
        assert capability.agent_id == "agent_1"
        assert capability.communication_range == 10.0
        assert capability.bandwidth == 15
        assert capability.protocols == ["direct", "broadcast", "relay"]
        assert capability.active_conversations == {}
        assert capability.message_queue == []
        assert capability.sent_messages_count == 0
        assert capability.received_messages_count == 0

    def test_communication_capability_defaults(self):
        """Test communication capability with default values."""
        from agents.base.communication import CommunicationCapability

        mock_message_system = Mock()

        capability = CommunicationCapability(message_system=mock_message_system, agent_id="agent_1")

        assert capability.communication_range == 5.0
        assert capability.bandwidth == 10
        assert capability.protocols == ["direct", "broadcast"]

    def test_communication_capability_registration(self):
        """Test registration with message system."""
        from agents.base.communication import CommunicationCapability

        mock_message_system = Mock()
        mock_message_system.register_agent = Mock()

        capability = CommunicationCapability(message_system=mock_message_system, agent_id="agent_1")

        mock_message_system.register_agent.assert_called_once_with("agent_1", capability)

    def test_send_message_success(self):
        """Test successful message sending."""
        from agents.base.communication import CommunicationCapability

        mock_message_system = Mock()
        mock_message_system.send_message = Mock(return_value=True)

        capability = CommunicationCapability(message_system=mock_message_system, agent_id="agent_1")

        result = capability.send_message(
            recipient_id="agent_2",
            content="Hello!",
            intent=ConversationIntent.CASUAL_GREETING,
            metadata={"priority": "normal"},
        )

        assert result is True
        assert capability.sent_messages_count == 1
        mock_message_system.send_message.assert_called_once()

        # Check message structure
        sent_message = mock_message_system.send_message.call_args[0][0]
        assert sent_message.sender_id == "agent_1"
        assert sent_message.recipient_id == "agent_2"
        assert sent_message.content == "Hello!"
        assert sent_message.intent == ConversationIntent.CASUAL_GREETING
        assert sent_message.metadata == {"priority": "normal"}

    def test_send_message_bandwidth_exceeded(self):
        """Test message sending when bandwidth is exceeded."""
        from agents.base.communication import CommunicationCapability

        mock_message_system = Mock()

        capability = CommunicationCapability(
            message_system=mock_message_system, agent_id="agent_1", bandwidth=2
        )

        # Send messages up to bandwidth limit
        capability.send_message("agent_2", "Message 1")
        capability.send_message("agent_2", "Message 2")

        # This should fail due to bandwidth
        result = capability.send_message("agent_2", "Message 3")

        assert result is False
        assert capability.sent_messages_count == 2

    def test_send_message_fallback_queue(self):
        """Test message sending with fallback to queue."""
        from agents.base.communication import CommunicationCapability

        mock_message_system = Mock()
        # Message system without send_message method
        delattr(mock_message_system, "send_message")

        capability = CommunicationCapability(message_system=mock_message_system, agent_id="agent_1")

        result = capability.send_message("agent_2", "Hello!")

        assert result is True
        assert capability.sent_messages_count == 1
        assert len(capability.message_queue) == 1
        assert capability.message_queue[0].content == "Hello!"

    def test_send_broadcast_message(self):
        """Test sending broadcast message."""
        from agents.base.communication import CommunicationCapability

        mock_message_system = Mock()
        mock_message_system.send_message = Mock(return_value=True)

        capability = CommunicationCapability(message_system=mock_message_system, agent_id="agent_1")

        result = capability.send_message(
            recipient_id=None,  # Broadcast
            content="Warning! Danger ahead!",
            intent=ConversationIntent.WARN_DANGER,
        )

        assert result is True

        sent_message = mock_message_system.send_message.call_args[0][0]
        assert sent_message.recipient_id is None
        assert sent_message.intent == ConversationIntent.WARN_DANGER

    def test_receive_message(self):
        """Test receiving a message."""
        from agents.base.communication import CommunicationCapability

        mock_message_system = Mock()

        capability = CommunicationCapability(message_system=mock_message_system, agent_id="agent_1")

        message = ConversationMessage(
            id="msg_1",
            sender_id="agent_2",
            recipient_id="agent_1",
            content="Hello!",
            intent=ConversationIntent.CASUAL_GREETING,
        )

        result = capability.receive_message(message)

        assert result is True
        assert capability.received_messages_count == 1
        assert len(capability.message_queue) == 1
        assert capability.message_queue[0] == message

        # Check conversation creation
        conv_id = capability._get_conversation_id("agent_2")
        assert conv_id in capability.active_conversations

        conversation = capability.active_conversations[conv_id]
        assert "agent_1" in conversation.participants
        assert "agent_2" in conversation.participants
        assert len(conversation.messages) == 1

    def test_start_conversation(self):
        """Test starting a new conversation."""
        from agents.base.communication import CommunicationCapability

        mock_message_system = Mock()

        capability = CommunicationCapability(message_system=mock_message_system, agent_id="agent_1")

        conversation = capability.start_conversation("agent_2", ["explore", "trade"])

        assert conversation is not None
        assert "agent_1" in conversation.participants
        assert "agent_2" in conversation.participants
        assert conversation.conversation_goals["agent_1"] == ["explore", "trade"]

        conv_id = capability._get_conversation_id("agent_2")
        assert capability.active_conversations[conv_id] == conversation

    def test_start_conversation_existing(self):
        """Test starting conversation that already exists."""
        from agents.base.communication import CommunicationCapability

        mock_message_system = Mock()

        capability = CommunicationCapability(message_system=mock_message_system, agent_id="agent_1")

        # Start first conversation
        conv1 = capability.start_conversation("agent_2")

        # Try to start same conversation again
        conv2 = capability.start_conversation("agent_2")

        assert conv1 == conv2

    def test_get_pending_messages(self):
        """Test getting pending messages."""
        from agents.base.communication import CommunicationCapability

        mock_message_system = Mock()

        capability = CommunicationCapability(message_system=mock_message_system, agent_id="agent_1")

        # Add some messages to queue
        msg1 = ConversationMessage(
            id="msg_1",
            sender_id="agent_2",
            recipient_id="agent_1",
            content="Hello",
            intent=ConversationIntent.CASUAL_GREETING,
        )
        msg2 = ConversationMessage(
            id="msg_2",
            sender_id="agent_3",
            recipient_id="agent_1",
            content="Trade?",
            intent=ConversationIntent.PROPOSE_TRADE,
        )

        capability.receive_message(msg1)
        capability.receive_message(msg2)

        pending = capability.get_pending_messages()

        assert len(pending) == 2
        assert msg1 in pending
        assert msg2 in pending
        assert len(capability.message_queue) == 0  # Queue should be cleared

    def test_get_active_conversations_capability(self):
        """Test getting active conversations from capability."""
        from agents.base.communication import CommunicationCapability

        mock_message_system = Mock()

        capability = CommunicationCapability(message_system=mock_message_system, agent_id="agent_1")

        conv1 = capability.start_conversation("agent_2")
        conv2 = capability.start_conversation("agent_3")

        active_convs = capability.get_active_conversations()

        assert len(active_convs) == 2
        assert conv1 in active_convs
        assert conv2 in active_convs

    def test_end_conversation(self):
        """Test ending a conversation."""
        from agents.base.communication import CommunicationCapability

        mock_message_system = Mock()

        capability = CommunicationCapability(message_system=mock_message_system, agent_id="agent_1")

        conversation = capability.start_conversation("agent_2")
        conv_id = capability._get_conversation_id("agent_2")

        result = capability.end_conversation(conv_id)

        assert result is True
        assert conversation.active is False
        assert conv_id not in capability.active_conversations

    def test_end_conversation_not_found(self):
        """Test ending a non-existent conversation."""
        from agents.base.communication import CommunicationCapability

        mock_message_system = Mock()

        capability = CommunicationCapability(message_system=mock_message_system, agent_id="agent_1")

        result = capability.end_conversation("nonexistent_conv")

        assert result is False

    def test_reset_cycle(self):
        """Test resetting per-cycle counters."""
        from agents.base.communication import CommunicationCapability

        mock_message_system = Mock()
        mock_message_system.send_message = Mock(return_value=True)

        capability = CommunicationCapability(message_system=mock_message_system, agent_id="agent_1")

        # Send and receive some messages
        capability.send_message("agent_2", "Hello")
        capability.receive_message(
            ConversationMessage(
                id="msg_1",
                sender_id="agent_2",
                recipient_id="agent_1",
                content="Hi",
                intent=ConversationIntent.CASUAL_GREETING,
            )
        )

        assert capability.sent_messages_count == 1
        assert capability.received_messages_count == 1

        capability.reset_cycle()

        assert capability.sent_messages_count == 0
        assert capability.received_messages_count == 0

    def test_get_stats(self):
        """Test getting communication statistics."""
        from agents.base.communication import CommunicationCapability

        mock_message_system = Mock()
        mock_message_system.send_message = Mock(return_value=True)

        capability = CommunicationCapability(
            message_system=mock_message_system,
            agent_id="agent_1",
            communication_range=15.0,
            bandwidth=20,
        )

        # Add some activity
        capability.send_message("agent_2", "Hello")
        capability.start_conversation("agent_3")
        capability.receive_message(
            ConversationMessage(
                id="msg_1",
                sender_id="agent_2",
                recipient_id="agent_1",
                content="Hi",
                intent=ConversationIntent.CASUAL_GREETING,
            )
        )

        stats = capability.get_stats()

        assert stats["agent_id"] == "agent_1"
        assert stats["communication_range"] == 15.0
        assert stats["bandwidth"] == 20
        assert stats["active_conversations"] == 2  # agent_2 and agent_3
        assert stats["messages_sent"] == 1
        assert stats["messages_received"] == 1
        assert stats["pending_messages"] == 1

    def test_get_conversation_id(self):
        """Test conversation ID generation."""
        from agents.base.communication import CommunicationCapability

        mock_message_system = Mock()

        capability = CommunicationCapability(message_system=mock_message_system, agent_id="agent_1")

        conv_id1 = capability._get_conversation_id("agent_2")
        conv_id2 = capability._get_conversation_id("agent_2")
        conv_id3 = capability._get_conversation_id("agent_3")

        # Same agents should generate same ID
        assert conv_id1 == conv_id2

        # Different agents should generate different ID
        assert conv_id1 != conv_id3

        # Should be consistent regardless of order
        capability2 = CommunicationCapability(
            message_system=mock_message_system, agent_id="agent_2"
        )
        conv_id4 = capability2._get_conversation_id("agent_1")
        assert conv_id1 == conv_id4

    def test_repr(self):
        """Test string representation."""
        from agents.base.communication import CommunicationCapability

        mock_message_system = Mock()

        capability = CommunicationCapability(
            message_system=mock_message_system, agent_id="agent_1", communication_range=12.5
        )

        capability.start_conversation("agent_2")

        repr_str = repr(capability)

        assert "CommunicationCapability" in repr_str
        assert "agent=agent_1" in repr_str
        assert "range=12.5" in repr_str
        assert "conversations=1" in repr_str


class TestCommunicationIntegration:
    """Test integration between communication components."""

    def test_full_conversation_flow(self):
        """Test complete conversation flow from start to finish."""
        from agents.base.communication import ConversationManager

        # Create conversation manager
        manager = ConversationManager()

        # Create conversation
        conversation = manager.create_conversation(
            participants=["agent_1", "agent_2"],
            goals={"agent_1": ["find_water"], "agent_2": ["form_alliance"]},
        )

        # Agent 1 state (has discovery to share)
        agent1_state = {
            "recent_discoveries": ["water_source"],
            "location": "hex_123",
            "free_energy": 0.3,
            "confidence": 0.8,
        }

        # Agent 2 state (seeking allies)
        agent2_state = {"seeking_allies": True, "free_energy": 0.5, "confidence": 0.7}

        # Process several turns
        turn1 = conversation.process_turn("agent_1", agent1_state)
        assert turn1.action == "speak"
        assert turn1.message.intent == ConversationIntent.SHARE_DISCOVERY
        assert "water_source" in turn1.message.content

        turn2 = conversation.process_turn("agent_2", agent2_state)
        # Action is determined randomly when agent didn't just speak and no urgent message
        # Could be speak, listen, or think
        assert turn2.action in ["speak", "listen", "think"]
        if turn2.action == "speak":
            assert turn2.message.intent == ConversationIntent.FORM_ALLIANCE

        # Check conversation state
        expected_messages = 2 if turn2.action == "speak" else 1
        assert len(conversation.messages) == expected_messages
        assert len(conversation.turns) == 2
        assert conversation.turn_count == 2
        assert conversation.active is True

        # Get summary
        summary = conversation.get_conversation_summary()
        assert summary["message_count"] == expected_messages
        assert summary["intent_distribution"]["share_discovery"] == 1
        if turn2.action == "speak":
            assert summary["intent_distribution"]["form_alliance"] == 1

    def test_communication_capability_with_conversation(self):
        """Test communication capability integration with conversations."""
        from agents.base.communication import CommunicationCapability

        mock_message_system = Mock()
        mock_message_system.send_message = Mock(return_value=True)

        # Create two communication capabilities
        cap1 = CommunicationCapability(mock_message_system, "agent_1")
        cap2 = CommunicationCapability(mock_message_system, "agent_2")

        # Agent 1 starts conversation and sends message
        conversation = cap1.start_conversation("agent_2", ["explore"])

        result = cap1.send_message(
            recipient_id="agent_2",
            content="Want to explore together?",
            intent=ConversationIntent.FORM_ALLIANCE,
        )

        assert result is True

        # Simulate message delivery to agent 2
        sent_message = mock_message_system.send_message.call_args[0][0]
        cap2.receive_message(sent_message)

        # Check both agents have the conversation
        cap1_convs = cap1.get_active_conversations()
        cap2_convs = cap2.get_active_conversations()

        assert len(cap1_convs) == 1
        assert len(cap2_convs) == 1

        # Check message was added to cap2's conversation
        cap2_conv_id = cap2._get_conversation_id("agent_1")
        cap2_conversation = cap2.active_conversations[cap2_conv_id]
        assert len(cap2_conversation.messages) == 1
        assert cap2_conversation.messages[0].content == "Want to explore together?"

    def test_belief_updating_integration(self):
        """Test belief updating with knowledge graph integration."""
        conversation = AgentConversation()

        # Mock knowledge graph
        mock_kg = Mock()
        mock_kg.add_belief = Mock()

        # Create discovery message
        message = ConversationMessage(
            id="msg_1",
            sender_id="agent_1",
            recipient_id="agent_2",
            content="I found a rich mineral deposit at hex_456",
            intent=ConversationIntent.SHARE_DISCOVERY,
        )

        # Update beliefs
        beliefs = conversation.update_beliefs_from_message("agent_2", message, mock_kg)

        # Verify beliefs were created and added to knowledge graph
        assert len(beliefs) == 2
        assert mock_kg.add_belief.call_count == 2

        # Check belief content
        discovery_belief = beliefs[0]
        assert "Agent agent_1 discovered" in discovery_belief.statement
        assert discovery_belief.confidence == 0.7

        # Check knowledge graph calls
        first_call = mock_kg.add_belief.call_args_list[0]
        assert discovery_belief.statement in first_call[0]
        assert first_call[0][1] == 0.7  # confidence
        assert first_call[1]["metadata"]["type"] == "belief"
        assert first_call[1]["metadata"]["sender"] == "agent_1"


class TestCommunicationEdgeCases:
    """Test edge cases and error conditions in communication system."""

    def test_conversation_manager_missing_conversation(self):
        """Test getting communications for missing conversation IDs."""
        from agents.base.communication import ConversationManager
        
        manager = ConversationManager()
        conv = manager.create_conversation(["agent_1", "agent_2"])
        
        # Manually remove conversation but keep agent mapping
        del manager.conversations[conv.conversation_id]
        
        # Should handle missing conversation gracefully
        convs = manager.get_agent_communications("agent_1")
        assert convs == []

    def test_communication_capability_no_register_method(self):
        """Test capability with message system that has no register_agent method."""
        from agents.base.communication import CommunicationCapability
        
        mock_system = Mock()
        # Remove register_agent method
        if hasattr(mock_system, 'register_agent'):
            delattr(mock_system, 'register_agent')
        
        # Should not raise exception
        cap = CommunicationCapability(mock_system, "agent_1")
        assert cap.agent_id == "agent_1"

    def test_communication_capability_system_send_failure(self):
        """Test message sending when system returns False."""
        from agents.base.communication import CommunicationCapability
        
        mock_system = Mock()
        mock_system.send_message = Mock(return_value=False)
        
        cap = CommunicationCapability(mock_system, "agent_1")
        
        result = cap.send_message("agent_2", "Hello")
        
        assert result is False
        assert cap.sent_messages_count == 0  # Should not increment on failure

    def test_agent_conversation_empty_resources_trade(self):
        """Test trade intent with empty resources."""
        conversation = AgentConversation()
        
        # Empty resources should still trigger trade intent check
        state = {"resources": {}}
        intent = conversation._determine_intent(state)
        
        # Should default to casual greeting
        assert intent == ConversationIntent.CASUAL_GREETING

    def test_agent_conversation_empty_discoveries(self):
        """Test discovery intent with empty discoveries list."""
        conversation = AgentConversation()
        
        state = {"recent_discoveries": []}
        intent = conversation._determine_intent(state)
        
        assert intent == ConversationIntent.CASUAL_GREETING

    def test_generate_template_empty_discoveries(self):
        """Test template generation with empty discoveries."""
        conversation = AgentConversation()
        
        state = {"recent_discoveries": [], "location": "test"}
        message = conversation._generate_template_message(
            "agent_1", state, ConversationIntent.SHARE_DISCOVERY
        )
        
        # Should use "something" as fallback
        assert "something" in message

    def test_generate_template_empty_resources(self):
        """Test template generation with empty resources for trade."""
        conversation = AgentConversation()
        
        state = {"resources": {}}
        message = conversation._generate_template_message(
            "agent_1", state, ConversationIntent.PROPOSE_TRADE
        )
        
        # Should use fallback values
        assert "resources" in message or "items" in message

    def test_generate_template_no_uncertainty_topics(self):
        """Test template generation without uncertainty topics."""
        conversation = AgentConversation()
        
        state = {}  # No uncertainty_topics
        message = conversation._generate_template_message(
            "agent_1", state, ConversationIntent.SEEK_INFORMATION
        )
        
        # Should use default "the area"
        assert "the area" in message

    def test_generate_template_missing_danger_info(self):
        """Test template generation without danger info."""
        conversation = AgentConversation()
        
        state = {}  # No danger_type or danger_location
        message = conversation._generate_template_message(
            "agent_1", state, ConversationIntent.WARN_DANGER
        )
        
        # Should use fallback values
        assert "threat" in message or "nearby" in message

    def test_conversation_summary_with_end_time(self):
        """Test conversation summary when conversation is ended."""
        conversation = AgentConversation()
        conversation.end_conversation()
        
        summary = conversation.get_conversation_summary()
        
        assert summary["active"] is False
        assert summary["end_time"] is not None
        assert isinstance(summary["duration"], float)

    def test_build_llm_prompt_no_personality(self):
        """Test LLM prompt building without personality."""
        conversation = AgentConversation()
        conversation.add_participant("agent_1", ["goal1"])
        
        state = {}  # No personality
        context = []
        
        prompt = conversation._build_llm_prompt(
            "agent_1", state, ConversationIntent.CASUAL_GREETING, context
        )
        
        assert "agent_1" in prompt
        assert "casual_greeting" in prompt

    def test_build_llm_prompt_many_context_messages(self):
        """Test LLM prompt with more than 3 context messages."""
        conversation = AgentConversation()
        conversation.add_participant("agent_1", ["goal1"])
        
        # Create 5 context messages
        context = []
        for i in range(5):
            context.append(ConversationMessage(
                id=f"msg_{i}",
                sender_id="agent_2",
                recipient_id="agent_1",
                content=f"Message {i}",
                intent=ConversationIntent.CASUAL_GREETING,
            ))
        
        prompt = conversation._build_llm_prompt(
            "agent_1", {}, ConversationIntent.CASUAL_GREETING, context
        )
        
        # Should only include last 3 messages
        assert "Message 2" in prompt
        assert "Message 3" in prompt 
        assert "Message 4" in prompt
        assert "Message 0" not in prompt
        assert "Message 1" not in prompt

    def test_update_beliefs_alliance_intent(self):
        """Test belief updates for alliance formation messages."""
        conversation = AgentConversation()
        
        mock_kg = Mock()
        mock_kg.add_belief = Mock()
        
        message = ConversationMessage(
            id="msg_1",
            sender_id="agent_1",
            recipient_id="agent_2",
            content="Let's work together!",
            intent=ConversationIntent.FORM_ALLIANCE,
        )
        
        beliefs = conversation.update_beliefs_from_message("agent_2", message, mock_kg)
        
        # Should only create trust belief for non-covered intents
        assert len(beliefs) == 1
        assert "Agent agent_1 communicated with intent form_alliance" in beliefs[0].statement

    def test_update_beliefs_seek_information_intent(self):
        """Test belief updates for information seeking messages."""
        conversation = AgentConversation()
        
        mock_kg = Mock()
        mock_kg.add_belief = Mock()
        
        message = ConversationMessage(
            id="msg_1",
            sender_id="agent_1",
            recipient_id="agent_2",
            content="Do you know about the forest?",
            intent=ConversationIntent.SEEK_INFORMATION,
        )
        
        beliefs = conversation.update_beliefs_from_message("agent_2", message, mock_kg)
        
        # Should only create trust belief
        assert len(beliefs) == 1
        assert beliefs[0].confidence == 0.8

    def test_communication_capability_conversation_id_sorted(self):
        """Test conversation ID generation maintains sorting."""
        from agents.base.communication import CommunicationCapability
        
        mock_system = Mock()
        
        cap1 = CommunicationCapability(mock_system, "zebra")
        cap2 = CommunicationCapability(mock_system, "alpha")
        
        id1 = cap1._get_conversation_id("alpha")
        id2 = cap2._get_conversation_id("zebra")
        
        # Should be identical due to sorting
        assert id1 == id2
        assert "conv_alpha_zebra" == id1

    def test_conversation_with_context_limit(self):
        """Test conversation processing with context limit."""
        conversation = AgentConversation()
        conversation.add_participant("agent_1")
        conversation.add_participant("agent_2")
        
        # Add many messages to test context limiting
        for i in range(10):
            msg = ConversationMessage(
                id=f"msg_{i}",
                sender_id="agent_2",
                recipient_id="agent_1", 
                content=f"Message {i}",
                intent=ConversationIntent.CASUAL_GREETING,
            )
            conversation.add_message(msg)
        
        # Generate message should only use last 5 for context
        state = {"recent_discoveries": ["test"]}
        message = conversation.generate_message("agent_1", state, [])
        
        # Should complete without error
        assert message.sender_id == "agent_1"

    def test_communication_stats_with_activity(self):
        """Test communication statistics with various activities."""
        from agents.base.communication import CommunicationCapability
        
        mock_system = Mock()
        
        cap = CommunicationCapability(mock_system, "agent_1", bandwidth=3)
        
        # Add queue messages directly (fallback path)
        cap.message_queue.append(ConversationMessage(
            id="queued_msg",
            sender_id="agent_2",
            recipient_id="agent_1",
            content="Queued message",
            intent=ConversationIntent.CASUAL_GREETING,
        ))
        
        stats = cap.get_stats()
        
        assert stats["pending_messages"] == 1

    def test_llm_generate_with_whitespace(self):
        """Test LLM generation with whitespace handling."""
        conversation = AgentConversation()
        
        mock_llm = Mock()
        mock_llm.generate.return_value = "  \n  Response with whitespace  \n  "
        
        result = conversation._generate_with_llm(
            "agent_1", {}, ConversationIntent.CASUAL_GREETING, [], mock_llm
        )
        
        # Should strip whitespace
        assert result == "Response with whitespace"

    def test_conversation_max_turns_boundary(self):
        """Test conversation behavior at max turns boundary."""
        conversation = AgentConversation(max_turns=1)
        conversation.add_participant("agent_1")
        
        # First turn should end the conversation
        turn = conversation.process_turn("agent_1", {})
        
        assert turn.action == "speak"
        assert conversation.active is False
        assert conversation.turn_count == 1

    def test_conversation_manager_multiple_agent_tracking(self):
        """Test conversation manager tracks agents across multiple conversations."""
        from agents.base.communication import ConversationManager
        
        manager = ConversationManager()
        
        conv1 = manager.create_conversation(["agent_1", "agent_2"])
        conv2 = manager.create_conversation(["agent_1", "agent_3"])
        conv3 = manager.create_conversation(["agent_1", "agent_4"])
        
        agent1_convs = manager.get_agent_communications("agent_1")
        
        assert len(agent1_convs) == 3
        assert all(conv in agent1_convs for conv in [conv1, conv2, conv3])


class TestExampleUsageSimulation:
    """Test the example usage from the main block."""

    def test_example_conversation_simulation(self):
        """Test simulation similar to the example in main block."""
        from agents.base.communication import ConversationManager
        
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
        
        # Agent states from example
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
        
        # Process several turns
        messages_generated = []
        for i in range(6):
            if i % 2 == 0:
                turn = conversation.process_turn("agent_1", agent1_state)
            else:
                turn = conversation.process_turn("agent_2", agent2_state)
            
            if turn.message:
                messages_generated.append(turn.message.content)
        
        # Should have generated some messages
        assert len(messages_generated) > 0
        
        # Get summary like in example
        summary = conversation.get_conversation_summary()
        
        assert summary["participants"] == ["agent_1", "agent_2"]
        assert "intent_distribution" in summary
        assert summary["turn_count"] == 6

    def test_json_serialization_roundtrip(self):
        """Test full JSON serialization roundtrip."""
        conversation = AgentConversation()
        conversation.add_participant("agent_1", ["goal1"])
        
        # Add a message
        msg = ConversationMessage(
            id="test_msg",
            sender_id="agent_1",
            recipient_id="agent_2",
            content="Test message",
            intent=ConversationIntent.SHARE_DISCOVERY,
            metadata={"test": "value"}
        )
        conversation.add_message(msg)
        
        # Convert to dict and serialize
        conv_dict = conversation.to_dict()
        json_str = json.dumps(conv_dict)
        
        # Should be able to deserialize
        parsed = json.loads(json_str)
        
        assert parsed["conversation_id"] == conversation.conversation_id
        assert len(parsed["messages"]) == 1
        assert parsed["messages"][0]["content"] == "Test message"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
