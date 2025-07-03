"""
Module for FreeAgentics Active Inference implementation.
"""

import threading
import time
from datetime import datetime, timedelta

import pytest

from agents.base.data_model import Agent, AgentResources, Position
from agents.base.interaction import (
    CommunicationProtocol,
    ConflictResolver,
    InteractionRequest,
    InteractionSystem,
    InteractionType,
    Message,
    MessageType,
    ResourceExchange,
    ResourceManager,
    ResourceType,
)


class TestMessage:
    """Test Message class"""

    def test_message_creation(self) -> None:
        msg = Message(
            sender_id="agent1",
            receiver_id="agent2",
            message_type=MessageType.REQUEST,
            content={"request": "help"},
        )
        assert msg.sender_id == "agent1"
        assert msg.receiver_id == "agent2"
        assert msg.message_type == MessageType.REQUEST
        assert msg.content == {"request": "help"}
        assert not msg.is_broadcast()

    def test_broadcast_message(self) -> None:
        msg = Message(
            sender_id="agent1",
            receiver_id=None,
            message_type=MessageType.INFORM,
            content={"info": "global update"},
        )
        assert msg.is_broadcast()

    def test_message_with_response_deadline(self) -> None:
        deadline = datetime.now() + timedelta(seconds=10)
        msg = Message(
            sender_id="agent1",
            receiver_id="agent2",
            requires_response=True,
            response_deadline=deadline,
        )
        assert msg.requires_response
        assert msg.response_deadline == deadline


class TestInteractionRequest:
    """Test InteractionRequest class"""

    def test_request_creation(self) -> None:
        req = InteractionRequest(
            initiator_id="agent1",
            target_id="agent2",
            interaction_type=InteractionType.COMMUNICATION,
            parameters={"message": "hello"},
        )
        assert req.initiator_id == "agent1"
        assert req.target_id == "agent2"
        assert req.interaction_type == InteractionType.COMMUNICATION
        assert req.parameters == {"message": "hello"}
        assert not req.is_expired()

    def test_request_expiration(self) -> None:
        req = InteractionRequest(initiator_id="agent1", target_id="agent2", timeout=0.1)
        assert not req.is_expired()
        time.sleep(0.2)
        assert req.is_expired()


class TestResourceExchange:
    """Test ResourceExchange class"""

    def test_exchange_creation(self) -> None:
        exchange = ResourceExchange(
            from_agent_id="agent1",
            to_agent_id="agent2",
            resource_type=ResourceType.ENERGY,
            amount=10.0,
        )
        assert exchange.from_agent_id == "agent1"
        assert exchange.to_agent_id == "agent2"
        assert exchange.resource_type == ResourceType.ENERGY
        assert exchange.amount == 10.0
        assert not exchange.completed

    def test_exchange_execution_success(self) -> None:
        agent1 = Agent(
            agent_id="agent1",
            position=Position(0, 0),
            resources=AgentResources(energy=50.0),
        )
        agent2 = Agent(
            agent_id="agent2",
            position=Position(1, 1),
            resources=AgentResources(energy=30.0),
        )
        exchange = ResourceExchange(
            from_agent_id=agent1.agent_id,
            to_agent_id=agent2.agent_id,
            resource_type=ResourceType.ENERGY,
            amount=20.0,
        )
        success = exchange.execute(agent1, agent2)
        assert success
        assert exchange.completed
        assert agent1.resources.energy == 30.0
        assert agent2.resources.energy == 50.0

    def test_exchange_execution_failure(self) -> None:
        agent1 = Agent(
            agent_id="agent1",
            position=Position(0, 0),
            resources=AgentResources(energy=10.0),
        )
        agent2 = Agent(
            agent_id="agent2",
            position=Position(1, 1),
            resources=AgentResources(energy=30.0),
        )
        exchange = ResourceExchange(
            from_agent_id=agent1.agent_id,
            to_agent_id=agent2.agent_id,
            resource_type=ResourceType.ENERGY,
            amount=20.0,
        )
        success = exchange.execute(agent1, agent2)
        assert not success
        assert not exchange.completed
        assert agent1.resources.energy == 10.0
        assert agent2.resources.energy == 30.0


class TestCommunicationProtocol:
    """Test CommunicationProtocol class"""

    def test_send_and_receive_message(self) -> None:
        protocol = CommunicationProtocol()
        msg = Message(sender_id="agent1", receiver_id="agent2", content={"text": "hello"})
        success = protocol.send_message(msg)
        assert success
        messages = protocol.receive_messages("agent2")
        assert len(messages) == 1
        assert messages[0].content == {"text": "hello"}
        messages = protocol.receive_messages("agent3")
        assert len(messages) == 0

    def test_broadcast_message(self) -> None:
        protocol = CommunicationProtocol()
        broadcast = Message(
            sender_id="agent1", receiver_id=None, content={"announcement": "global"}
        )
        protocol.send_message(broadcast)
        messages1 = protocol.receive_messages("agent2")
        messages2 = protocol.receive_messages("agent3")
        assert len(messages1) == 1
        assert len(messages2) == 1
        assert messages1[0].content == {"announcement": "global"}

    def test_conversation_history(self) -> None:
        protocol = CommunicationProtocol()
        msg1 = Message(sender_id="agent1", receiver_id="agent2", content={"text": "hi"})
        msg2 = Message(sender_id="agent2", receiver_id="agent1", content={"text": "hello"})
        protocol.send_message(msg1)
        protocol.send_message(msg2)
        history = protocol.get_conversation_history("agent1", "agent2")
        assert len(history) == 2
        assert history[0].content == {"text": "hi"}
        assert history[1].content == {"text": "hello"}

    def test_pending_responses(self) -> None:
        protocol = CommunicationProtocol()
        deadline = datetime.now() + timedelta(seconds=0.1)
        msg = Message(
            sender_id="agent1",
            receiver_id="agent2",
            requires_response=True,
            response_deadline=deadline,
        )
        protocol.send_message(msg)
        timed_out = protocol.check_pending_responses()
        assert len(timed_out) == 0
        time.sleep(0.2)
        timed_out = protocol.check_pending_responses()
        assert len(timed_out) == 1
        assert timed_out[0].id == msg.id


class TestResourceManager:
    """Test ResourceManager class"""

    def test_propose_exchange(self) -> None:
        manager = ResourceManager()
        exchange = ResourceExchange(
            from_agent_id="agent1",
            to_agent_id="agent2",
            resource_type=ResourceType.ENERGY,
            amount=10.0,
        )
        exchange_id = manager.propose_exchange(exchange)
        assert exchange_id == exchange.id
        assert exchange_id in manager.pending_exchanges

    def test_execute_exchange(self) -> None:
        manager = ResourceManager()
        agent1 = Agent(
            agent_id="agent1",
            position=Position(0, 0),
            resources=AgentResources(energy=50.0),
        )
        agent2 = Agent(
            agent_id="agent2",
            position=Position(1, 1),
            resources=AgentResources(energy=30.0),
        )
        exchange = ResourceExchange(
            from_agent_id=agent1.agent_id,
            to_agent_id=agent2.agent_id,
            resource_type=ResourceType.ENERGY,
            amount=20.0,
        )
        exchange_id = manager.propose_exchange(exchange)
        success = manager.execute_exchange(exchange_id, agent1, agent2)
        assert success
        assert exchange_id not in manager.pending_exchanges
        assert len(manager.exchange_history) == 1
        assert agent1.resources.energy == 30.0
        assert agent2.resources.energy == 50.0

    def test_cancel_exchange(self) -> None:
        manager = ResourceManager()
        exchange = ResourceExchange(from_agent_id="agent1", to_agent_id="agent2", amount=10.0)
        exchange_id = manager.propose_exchange(exchange)
        success = manager.cancel_exchange(exchange_id)
        assert success
        assert exchange_id not in manager.pending_exchanges
        assert len(manager.exchange_history) == 0

    def test_exchange_history(self) -> None:
        manager = ResourceManager()
        agent1 = Agent(
            agent_id="agent1",
            position=Position(0, 0),
            resources=AgentResources(energy=100.0),
        )
        agent2 = Agent(
            agent_id="agent2",
            position=Position(1, 1),
            resources=AgentResources(energy=50.0),
        )
        agent3 = Agent(
            agent_id="agent3",
            position=Position(2, 2),
            resources=AgentResources(energy=50.0),
        )
        exchange1 = ResourceExchange(
            from_agent_id=agent1.agent_id, to_agent_id=agent2.agent_id, amount=10.0
        )
        id1 = manager.propose_exchange(exchange1)
        manager.execute_exchange(id1, agent1, agent2)
        exchange2 = ResourceExchange(
            from_agent_id=agent2.agent_id, to_agent_id=agent3.agent_id, amount=5.0
        )
        id2 = manager.propose_exchange(exchange2)
        manager.execute_exchange(id2, agent2, agent3)
        history1 = manager.get_exchange_history("agent1")
        history2 = manager.get_exchange_history("agent2")
        history3 = manager.get_exchange_history("agent3")
        assert len(history1) == 1
        assert len(history2) == 2
        assert len(history3) == 1


class TestConflictResolver:
    """Test ConflictResolver class"""

    def test_resource_conflict_resolution(self) -> None:
        resolver = ConflictResolver()
        agent1 = Agent(
            agent_id="agent1",
            position=Position(0, 0),
            resources=AgentResources(energy=30.0),
        )
        agent2 = Agent(
            agent_id="agent2",
            position=Position(1, 1),
            resources=AgentResources(energy=50.0),
        )
        resolution = resolver.resolve_resource_conflict(agent1, agent2, ResourceType.ENERGY, 100.0)
        assert agent1.agent_id in resolution
        assert agent2.agent_id in resolution
        assert resolution[agent1.agent_id] + resolution[agent2.agent_id] == pytest.approx(100.0)
        assert len(resolver.conflict_history) == 1
        assert resolver.conflict_history[0]["type"] == "resource_conflict"

    def test_spatial_conflict_resolution(self) -> None:
        resolver = ConflictResolver()
        disputed_pos = Position(5, 5)
        agent1 = Agent(
            agent_id="agent1",
            position=Position(3, 3),
            resources=AgentResources(energy=50.0),
        )
        agent2 = Agent(
            agent_id="agent2",
            position=Position(8, 8),
            resources=AgentResources(energy=30.0),
        )
        winner = resolver.resolve_spatial_conflict(agent1, agent2, disputed_pos)
        assert winner == agent1.agent_id
        assert len(resolver.conflict_history) == 1
        assert resolver.conflict_history[0]["type"] == "spatial_conflict"
        assert resolver.conflict_history[0]["winner"] == winner


class TestInteractionSystem:
    """Test InteractionSystem class"""

    def test_agent_registration(self) -> None:
        system = InteractionSystem()
        agent = Agent(agent_id="agent1", position=Position(0, 0))
        system.register_agent(agent)
        assert "agent1" in system.registered_agents
        system.unregister_agent("agent1")
        assert "agent1" not in system.registered_agents

    def test_communication_interaction(self) -> None:
        system = InteractionSystem()
        request = InteractionRequest(
            initiator_id="agent1",
            target_id="agent2",
            interaction_type=InteractionType.COMMUNICATION,
            parameters={
                "message_type": MessageType.REQUEST,
                "content": {"text": "help needed"},
                "requires_response": True,
            },
        )
        interaction_id = system.initiate_interaction(request)
        result = system.process_interaction(interaction_id)
        assert result.success
        assert "message_id" in result.outcome
        messages = system.communication.receive_messages("agent2")
        assert len(messages) == 1
        assert messages[0].content == {"text": "help needed"}

    def test_resource_exchange_interaction(self) -> None:
        system = InteractionSystem()
        agent1 = Agent(
            agent_id="agent1",
            position=Position(0, 0),
            resources=AgentResources(energy=100.0),
        )
        agent2 = Agent(
            agent_id="agent2",
            position=Position(1, 1),
            resources=AgentResources(energy=50.0),
        )
        system.register_agent(agent1)
        system.register_agent(agent2)
        request = InteractionRequest(
            initiator_id="agent1",
            target_id="agent2",
            interaction_type=InteractionType.RESOURCE_EXCHANGE,
            parameters={"resource_type": ResourceType.ENERGY, "amount": 25.0},
        )
        interaction_id = system.initiate_interaction(request)
        result = system.process_interaction(interaction_id)
        assert result.success
        assert "exchange_id" in result.outcome
        exchange_id = result.outcome["exchange_id"]
        success = system.resource_manager.execute_exchange(exchange_id, agent1, agent2)
        assert success
        assert agent1.resources.energy == 75.0
        assert agent2.resources.energy == 75.0

    def test_conflict_interaction(self) -> None:
        system = InteractionSystem()
        agent1 = Agent(agent_id="agent1", position=Position(0, 0))
        agent2 = Agent(agent_id="agent2", position=Position(1, 1))
        system.register_agent(agent1)
        system.register_agent(agent2)
        request = InteractionRequest(
            initiator_id="agent1",
            target_id="agent2",
            interaction_type=InteractionType.CONFLICT,
            parameters={
                "conflict_type": "resource",
                "resource_type": ResourceType.ENERGY,
                "disputed_amount": 50.0,
            },
        )
        interaction_id = system.initiate_interaction(request)
        result = system.process_interaction(interaction_id)
        assert result.success
        assert agent1.agent_id in result.outcome
        assert agent2.agent_id in result.outcome

    def test_interaction_timeout(self) -> None:
        system = InteractionSystem()
        request = InteractionRequest(
            initiator_id="agent1",
            target_id="agent2",
            interaction_type=InteractionType.COMMUNICATION,
            timeout=0.1,
        )
        interaction_id = system.initiate_interaction(request)
        time.sleep(0.2)
        result = system.process_interaction(interaction_id)
        assert not result.success
        assert result.error_message == "Interaction timed out"

    def test_cleanup_expired_interactions(self) -> None:
        system = InteractionSystem()
        for i in range(3):
            request = InteractionRequest(initiator_id=f"agent{i}", target_id="target", timeout=0.1)
            system.initiate_interaction(request)
        assert len(system.active_interactions) == 3
        time.sleep(0.2)
        cleaned = system.cleanup_expired_interactions()
        assert cleaned == 3
        assert len(system.active_interactions) == 0
        assert len(system.interaction_history) == 3

    def test_interaction_callbacks(self) -> None:
        system = InteractionSystem()
        callback_called = []

        def test_callback(request) -> None:
            callback_called.append(request.id)

        system.register_interaction_callback(InteractionType.COMMUNICATION, test_callback)
        request = InteractionRequest(
            initiator_id="agent1",
            target_id="agent2",
            interaction_type=InteractionType.COMMUNICATION,
        )
        system.initiate_interaction(request)
        assert len(callback_called) == 1
        assert callback_called[0] == request.id

    def test_concurrent_interactions(self) -> None:
        system = InteractionSystem()
        results = []

        def run_interaction(agent_id):
            request = InteractionRequest(
                initiator_id=agent_id,
                target_id="target",
                interaction_type=InteractionType.COMMUNICATION,
                parameters={"content": {"from": agent_id}},
            )
            interaction_id = system.initiate_interaction(request)
            result = system.process_interaction(interaction_id)
            results.append(result)

        threads = []
        for i in range(10):
            thread = threading.Thread(target=run_interaction, args=(f"agent{i}",))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        assert len(results) == 10
        assert all(r.success for r in results)
