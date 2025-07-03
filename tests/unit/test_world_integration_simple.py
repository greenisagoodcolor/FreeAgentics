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
Simplified Unit tests for Agent World Integration System
Tests the world integration system by importing the module directly to avoid
dependency issues with other components.
"""
# Add the project root to Python path
project_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, project_root)
# Import the world integration module directly
spec = importlib.util.spec_from_file_location(
    "world_integration", os.path.join(project_root, "agents", "base", "world_integration.py")
)
world_integration = importlib.util.module_from_spec(spec)
# Mock the world imports to avoid dependency issues
world_integration.H3World = Mock
world_integration.HexCell = Mock
world_integration.Biome = Mock
world_integration.TerrainType = Mock
world_integration.SpatialAPI = Mock
world_integration.SpatialCoordinate = Mock
world_integration.ResourceType = Mock
# Execute the module
spec.loader.exec_module(world_integration)
# Extract the classes we need
AgentWorldManager = world_integration.AgentWorldManager
WorldEventSystem = world_integration.WorldEventSystem
ActionType = world_integration.ActionType
EventType = world_integration.EventType
WorldEvent = world_integration.WorldEvent
Perception = world_integration.Perception
ActionResult = world_integration.ActionResult


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

    def setUp(self) -> None:
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

    def setUp(self) -> None:
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
