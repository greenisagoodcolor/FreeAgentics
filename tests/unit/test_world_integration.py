"""
Module for FreeAgentics Active Inference implementation.
"""

import importlib.util
import os
import sys
import unittest
from datetime import datetime
from unittest.mock import Mock

"""
Unit tests for Agent World Integration System
Tests the comprehensive world integration system including AgentWorldManager,
WorldEventSystem, action execution, and agent-world coordination.
"""
# Add the project root to Python path to import modules directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
# Import the world integration module directly
spec = importlib.util.spec_from_file_location(
    "world_integration",
    os.path.join(os.path.dirname(__file__), "..", "..", "agents", "base", "world_integration.py"),
)
world_integration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(world_integration)
# Extract the classes we need
AgentWorldManager = world_integration.AgentWorldManager
WorldEventSystem = world_integration.WorldEventSystem
ActionType = world_integration.ActionType
EventType = world_integration.EventType
WorldEvent = world_integration.WorldEvent
Perception = world_integration.Perception
ActionResult = world_integration.ActionResult
IWorldPerceptionInterface = world_integration.IWorldPerceptionInterface
IWorldActionInterface = world_integration.IWorldActionInterface
IWorldEventSystem = world_integration.IWorldEventSystem


class MockH3World:
    """Mock H3World for testing"""

    def __init__(self) -> None:
        self.resolution = 7
        self.cells = {}

    def get_cell(self, hex_id: str):
        """Get a mock cell"""
        if hex_id not in self.cells:
            self.cells[hex_id] = MockHexCell(hex_id)
        return self.cells[hex_id]

    def get_neighbors(self, hex_id: str):
        """Get mock neighbors"""
        return [MockHexCell(f"{hex_id}_neighbor_{i}") for i in range(3)]

    def get_visible_cells(self, hex_id: str):
        """Get mock visible cells"""
        return [MockHexCell(f"{hex_id}_visible_{i}") for i in range(5)]


class MockHexCell:
    """Mock HexCell for testing"""

    def __init__(self, hex_id: str) -> None:
        self.hex_id = hex_id
        self.resources = {"water": 50.0, "food": 30.0, "materials": 20.0}
        self.temperature = 20.0
        self.moisture = 0.6
        self.elevation = 100.0
        self.biome = Mock(value="grassland")
        self.terrain = Mock(value="ground")
        self.movement_cost = 1.0


class TestWorldEventSystem(unittest.TestCase):
    """Test the world event system"""

    def set_up(self) -> None:
        """Set up test fixtures"""
        self.event_system = WorldEventSystem()
        self.test_events = []

    def test_subscribe_to_events(self) -> None:
        """Test event subscription"""

        def callback(event):
            self.test_events.append(event)

        self.event_system.subscribe_to_events("agent1", [EventType.AGENT_MOVED], callback)
        # Check that agent is subscribed
        self.assertIn("agent1", self.event_system.subscribers[EventType.AGENT_MOVED])

    def test_publish_event(self) -> None:
        """Test event publishing"""

        def callback(event):
            self.test_events.append(event)

        self.event_system.subscribe_to_events("agent1", [EventType.AGENT_MOVED], callback)
        event = WorldEvent(
            event_type=EventType.AGENT_MOVED,
            location="test_hex",
            agent_id="agent2",
            data={"from": "old_hex", "to": "test_hex"},
        )
        self.event_system.publish_event(event)
        # Check that event was received
        self.assertEqual(len(self.test_events), 1)
        self.assertEqual(self.test_events[0].event_type, EventType.AGENT_MOVED)

    def test_unsubscribe_from_events(self) -> None:
        """Test event unsubscription"""

        def callback(event):
            self.test_events.append(event)

        self.event_system.subscribe_to_events("agent1", [EventType.AGENT_MOVED], callback)
        self.event_system.unsubscribe_from_events("agent1", [EventType.AGENT_MOVED])
        # Check that agent is unsubscribed
        self.assertNotIn("agent1", self.event_system.subscribers[EventType.AGENT_MOVED])

    def test_get_recent_events(self) -> None:
        """Test getting recent events"""
        event = WorldEvent(
            event_type=EventType.RESOURCE_DEPLETED, location="test_hex", agent_id="agent1"
        )
        self.event_system.publish_event(event)
        recent_events = self.event_system.get_recent_events("test_hex", 10)
        self.assertEqual(len(recent_events), 1)
        self.assertEqual(recent_events[0].event_type, EventType.RESOURCE_DEPLETED)


class TestAgentWorldManager(unittest.TestCase):
    """Test the main agent world manager"""

    def set_up(self) -> None:
        """Set up test fixtures"""
        self.world = MockH3World()
        self.manager = AgentWorldManager(self.world)

    def test_place_agent(self) -> None:
        """Test placing an agent in the world"""
        success = self.manager.place_agent("agent1", "hex_123")
        self.assertTrue(success)
        self.assertEqual(self.manager.agent_locations["agent1"], "hex_123")
        self.assertEqual(self.manager.agent_energy["agent1"], 100.0)

    def test_place_agent_invalid_location(self) -> None:
        """Test placing agent at invalid location"""
        # Mock world to return None for invalid hex
        self.world.get_cell = Mock(return_value=None)
        success = self.manager.place_agent("agent1", "invalid_hex")
        self.assertFalse(success)
        self.assertNotIn("agent1", self.manager.agent_locations)

    def test_remove_agent(self) -> None:
        """Test removing an agent from the world"""
        self.manager.place_agent("agent1", "hex_123")
        self.manager.remove_agent("agent1")
        self.assertNotIn("agent1", self.manager.agent_locations)
        self.assertNotIn("agent1", self.manager.agent_resources)
        self.assertNotIn("agent1", self.manager.agent_energy)

    def test_perceive_environment(self) -> None:
        """Test agent environment perception"""
        self.manager.place_agent("agent1", "hex_123")
        self.manager.place_agent("agent2", "hex_123_visible_1")  # Nearby agent
        perception = self.manager.perceive_environment("agent1")
        self.assertIsNotNone(perception)
        self.assertEqual(perception.current_location, "hex_123")
        self.assertIn("agent2", perception.nearby_agents)
        self.assertIn("water", perception.available_resources)
        self.assertGreater(len(perception.movement_options), 0)

    def test_perceive_environment_invalid_agent(self) -> None:
        """Test perception for non-existent agent"""
        perception = self.manager.perceive_environment("invalid_agent")
        self.assertIsNone(perception)

    def test_get_available_actions(self) -> None:
        """Test getting available actions for agent"""
        self.manager.place_agent("agent1", "hex_123")
        actions = self.manager.get_available_actions("agent1")
        self.assertIn(ActionType.OBSERVE, actions)
        self.assertIn(ActionType.COMMUNICATE, actions)
        self.assertIn(ActionType.MOVE, actions)  # Should have neighbors
        self.assertIn(ActionType.HARVEST_RESOURCE, actions)  # Cell has resources

    def test_perform_move_action(self) -> None:
        """Test movement action"""
        self.manager.place_agent("agent1", "hex_123")
        result = self.manager.perform_action(
            "agent1", ActionType.MOVE, {"target_hex": "hex_123_neighbor_0"}
        )
        self.assertTrue(result.success)
        self.assertEqual(result.action_type, ActionType.MOVE)
        self.assertEqual(self.manager.agent_locations["agent1"], "hex_123_neighbor_0")
        self.assertLess(self.manager.agent_energy["agent1"], 100.0)  # Energy consumed

    def test_perform_move_action_invalid_target(self) -> None:
        """Test movement to invalid target"""
        self.manager.place_agent("agent1", "hex_123")
        result = self.manager.perform_action(
            "agent1", ActionType.MOVE, {"target_hex": "invalid_far_hex"}
        )
        self.assertFalse(result.success)
        self.assertIn("not adjacent", result.message)

    def test_perform_harvest_action(self) -> None:
        """Test resource harvesting"""
        self.manager.place_agent("agent1", "hex_123")
        result = self.manager.perform_action(
            "agent1", ActionType.HARVEST_RESOURCE, {"resource_type": "water", "amount": 10.0}
        )
        self.assertTrue(result.success)
        self.assertEqual(result.action_type, ActionType.HARVEST_RESOURCE)
        self.assertEqual(self.manager.agent_resources["agent1"]["water"], 10.0)
        # Check world state updated
        cell = self.world.get_cell("hex_123")
        self.assertEqual(cell.resources["water"], 40.0)  # 50 - 10

    def test_perform_harvest_nonexistent_resource(self) -> None:
        """Test harvesting nonexistent resource"""
        self.manager.place_agent("agent1", "hex_123")
        result = self.manager.perform_action(
            "agent1", ActionType.HARVEST_RESOURCE, {"resource_type": "nonexistent", "amount": 10.0}
        )
        self.assertFalse(result.success)
        self.assertIn("not available", result.message)

    def test_perform_deposit_action(self) -> None:
        """Test resource deposit"""
        self.manager.place_agent("agent1", "hex_123")
        self.manager.agent_resources["agent1"]["water"] = 15.0
        result = self.manager.perform_action(
            "agent1", ActionType.DEPOSIT_RESOURCE, {"resource_type": "water", "amount": 5.0}
        )
        self.assertTrue(result.success)
        self.assertEqual(self.manager.agent_resources["agent1"]["water"], 10.0)
        # Check world state updated
        cell = self.world.get_cell("hex_123")
        self.assertEqual(cell.resources["water"], 55.0)  # 50 + 5

    def test_perform_observe_action(self) -> None:
        """Test observation action"""
        self.manager.place_agent("agent1", "hex_123")
        result = self.manager.perform_action("agent1", ActionType.OBSERVE, {})
        self.assertTrue(result.success)
        self.assertEqual(result.action_type, ActionType.OBSERVE)
        self.assertIn("perception", result.effects)

    def test_perform_communicate_action(self) -> None:
        """Test communication action"""
        self.manager.place_agent("agent1", "hex_123")
        result = self.manager.perform_action(
            "agent1", ActionType.COMMUNICATE, {"target_agent": "agent2", "message": "Hello"}
        )
        self.assertTrue(result.success)
        self.assertEqual(result.action_type, ActionType.COMMUNICATE)
        self.assertEqual(len(result.generated_events), 1)
        self.assertEqual(result.generated_events[0].event_type, EventType.AGENT_INTERACTION)

    def test_perform_build_action(self) -> None:
        """Test building action"""
        self.manager.place_agent("agent1", "hex_123")
        result = self.manager.perform_action(
            "agent1", ActionType.BUILD_STRUCTURE, {"structure_type": "shelter"}
        )
        self.assertTrue(result.success)
        self.assertEqual(result.action_type, ActionType.BUILD_STRUCTURE)
        self.assertIn("shelter", self.manager.structures["hex_123"])

    def test_insufficient_energy(self) -> None:
        """Test action with insufficient energy"""
        self.manager.place_agent("agent1", "hex_123")
        self.manager.agent_energy["agent1"] = 5.0  # Very low energy
        result = self.manager.perform_action(
            "agent1", ActionType.MOVE, {"target_hex": "hex_123_neighbor_0"}
        )
        self.assertFalse(result.success)
        self.assertIn("Insufficient energy", result.message)

    def test_get_agent_energy(self) -> None:
        """Test getting agent energy"""
        self.manager.place_agent("agent1", "hex_123")
        energy = self.manager.get_agent_energy("agent1")
        self.assertEqual(energy, 100.0)

    def test_restore_agent_energy(self) -> None:
        """Test restoring agent energy"""
        self.manager.place_agent("agent1", "hex_123")
        self.manager.agent_energy["agent1"] = 50.0
        self.manager.restore_agent_energy("agent1", 30.0)
        self.assertEqual(self.manager.agent_energy["agent1"], 80.0)

    def test_restore_agent_energy_cap(self) -> None:
        """Test energy restoration with cap"""
        self.manager.place_agent("agent1", "hex_123")
        self.manager.restore_agent_energy("agent1", 50.0)  # Over 100
        self.assertEqual(self.manager.agent_energy["agent1"], 100.0)  # Capped

    def test_get_agent_resources(self) -> None:
        """Test getting agent resources"""
        self.manager.place_agent("agent1", "hex_123")
        self.manager.agent_resources["agent1"]["water"] = 25.0
        resources = self.manager.get_agent_resources("agent1")
        self.assertEqual(resources["water"], 25.0)
        self.assertIsInstance(resources, dict)  # Should be a copy

    def test_get_world_state_summary(self) -> None:
        """Test getting world state summary"""
        self.manager.place_agent("agent1", "hex_123")
        self.manager.place_agent("agent2", "hex_456")
        summary = self.manager.get_world_state_summary()
        self.assertEqual(summary["num_agents"], 2)
        self.assertEqual(summary["total_energy"], 200.0)
        self.assertIn("num_structures", summary)
        self.assertIn("recent_events", summary)


class TestWorldIntegrationInterfaces(unittest.TestCase):
    """Test the world integration interfaces"""

    def test_perception_interface_compliance(self) -> None:
        """Test that AgentWorldManager implements perception interface"""
        world = MockH3World()
        manager = AgentWorldManager(world)
        # Test interface compliance
        self.assertTrue(hasattr(manager, "perceive_environment"))
        self.assertTrue(hasattr(manager, "get_available_actions"))
        # Test basic functionality
        manager.place_agent("agent1", "hex_123")
        perception = manager.perceive_environment("agent1")
        self.assertIsNotNone(perception)
        actions = manager.get_available_actions("agent1")
        self.assertIsInstance(actions, list)

    def test_action_interface_compliance(self) -> None:
        """Test that AgentWorldManager implements action interface"""
        world = MockH3World()
        manager = AgentWorldManager(world)
        # Test interface compliance
        self.assertTrue(hasattr(manager, "perform_action"))
        # Test basic functionality
        manager.place_agent("agent1", "hex_123")
        result = manager.perform_action("agent1", ActionType.OBSERVE, {})
        self.assertIsInstance(result, ActionResult)

    def test_event_interface_compliance(self) -> None:
        """Test that WorldEventSystem implements event interface"""
        event_system = WorldEventSystem()
        # Test interface compliance
        self.assertTrue(hasattr(event_system, "subscribe_to_events"))
        self.assertTrue(hasattr(event_system, "unsubscribe_from_events"))
        self.assertTrue(hasattr(event_system, "publish_event"))
        self.assertTrue(hasattr(event_system, "get_recent_events"))


class TestWorldIntegrationDataStructures(unittest.TestCase):
    """Test world integration data structures"""

    def test_world_event_creation(self) -> None:
        """Test WorldEvent creation and attributes"""
        event = WorldEvent(
            event_type=EventType.AGENT_MOVED,
            location="hex_123",
            agent_id="agent1",
            data={"from": "hex_456", "to": "hex_123"},
        )
        self.assertEqual(event.event_type, EventType.AGENT_MOVED)
        self.assertEqual(event.location, "hex_123")
        self.assertEqual(event.agent_id, "agent1")
        self.assertEqual(event.data["from"], "hex_456")
        self.assertIsInstance(event.timestamp, datetime)

    def test_perception_creation(self) -> None:
        """Test Perception creation and attributes"""
        perception = Perception(
            current_location="hex_123",
            visible_cells=[],
            nearby_agents={"agent2": "hex_456"},
            available_resources={"water": 50.0},
            movement_options=["hex_789"],
            environmental_conditions={"temperature": 20.0},
            recent_events=[],
        )
        self.assertEqual(perception.current_location, "hex_123")
        self.assertEqual(perception.nearby_agents["agent2"], "hex_456")
        self.assertEqual(perception.available_resources["water"], 50.0)
        self.assertIsInstance(perception.timestamp, datetime)

    def test_action_result_creation(self) -> None:
        """Test ActionResult creation and attributes"""
        result = ActionResult(
            success=True,
            action_type=ActionType.MOVE,
            cost=10.0,
            effects={"new_location": "hex_123"},
            message="Move successful",
        )
        self.assertTrue(result.success)
        self.assertEqual(result.action_type, ActionType.MOVE)
        self.assertEqual(result.cost, 10.0)
        self.assertEqual(result.effects["new_location"], "hex_123")
        self.assertEqual(result.message, "Move successful")


if __name__ == "__main__":
    unittest.main()
