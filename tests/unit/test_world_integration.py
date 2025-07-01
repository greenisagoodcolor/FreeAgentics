"""
Comprehensive test suite for agents.base.world_integration module.
Covers all classes and functions with edge cases and integration scenarios.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from agents.base.world_integration import (
    ActionType,
    EventType,
    WorldEvent,
    Perception,
    ActionResult,
    IWorldPerceptionInterface,
    IWorldActionInterface,
    IWorldEventSystem,
    WorldEventSystem,
    AgentWorldManager,
)


class TestActionType:
    """Test ActionType enum"""

    def test_action_type_values(self):
        """Test all action type enum values"""
        assert ActionType.MOVE.value == "move"
        assert ActionType.OBSERVE.value == "observe"
        assert ActionType.HARVEST_RESOURCE.value == "harvest_resource"
        assert ActionType.DEPOSIT_RESOURCE.value == "deposit_resource"
        assert ActionType.MODIFY_TERRAIN.value == "modify_terrain"
        assert ActionType.COMMUNICATE.value == "communicate"
        assert ActionType.BUILD_STRUCTURE.value == "build_structure"
        assert ActionType.TRADE.value == "trade"

    def test_action_type_completeness(self):
        """Test that all expected action types are defined"""
        expected_actions = {
            "move", "observe", "harvest_resource", "deposit_resource",
            "modify_terrain", "communicate", "build_structure", "trade"
        }
        actual_actions = {action.value for action in ActionType}
        assert actual_actions == expected_actions


class TestEventType:
    """Test EventType enum"""

    def test_event_type_values(self):
        """Test all event type enum values"""
        assert EventType.AGENT_MOVED.value == "agent_moved"
        assert EventType.RESOURCE_DEPLETED.value == "resource_depleted"
        assert EventType.RESOURCE_DISCOVERED.value == "resource_discovered"
        assert EventType.WEATHER_CHANGED.value == "weather_changed"
        assert EventType.STRUCTURE_BUILT.value == "structure_built"
        assert EventType.AGENT_INTERACTION.value == "agent_interaction"
        assert EventType.TERRITORY_CLAIMED.value == "territory_claimed"

    def test_event_type_completeness(self):
        """Test that all expected event types are defined"""
        expected_events = {
            "agent_moved", "resource_depleted", "resource_discovered",
            "weather_changed", "structure_built", "agent_interaction", "territory_claimed"
        }
        actual_events = {event.value for event in EventType}
        assert actual_events == expected_events


class TestWorldEvent:
    """Test WorldEvent dataclass"""

    def test_world_event_creation(self):
        """Test basic world event creation"""
        event = WorldEvent(
            event_type=EventType.AGENT_MOVED,
            location="hex_123",
            agent_id="agent_1",
            data={"from": "hex_122", "to": "hex_123"}
        )
        assert event.event_type == EventType.AGENT_MOVED
        assert event.location == "hex_123"
        assert event.agent_id == "agent_1"
        assert event.data == {"from": "hex_122", "to": "hex_123"}
        assert isinstance(event.timestamp, datetime)
        assert event.affected_agents == set()

    def test_world_event_with_affected_agents(self):
        """Test world event with affected agents"""
        affected = {"agent_2", "agent_3"}
        event = WorldEvent(
            event_type=EventType.RESOURCE_DISCOVERED,
            location="hex_456",
            affected_agents=affected
        )
        assert event.affected_agents == affected

    def test_world_event_defaults(self):
        """Test world event with default values"""
        event = WorldEvent(
            event_type=EventType.WEATHER_CHANGED,
            location="hex_789"
        )
        assert event.agent_id is None
        assert event.data == {}
        assert event.affected_agents == set()
        assert event.timestamp.tzinfo == timezone.utc

    def test_world_event_timestamp_timezone(self):
        """Test that timestamp is in UTC"""
        event = WorldEvent(
            event_type=EventType.STRUCTURE_BUILT,
            location="hex_000"
        )
        assert event.timestamp.tzinfo == timezone.utc


class TestPerception:
    """Test Perception dataclass"""

    def test_perception_creation(self):
        """Test basic perception creation"""
        visible_cells = [Mock(hex_id="hex_1"), Mock(hex_id="hex_2")]
        nearby_agents = {"agent_2": "hex_2", "agent_3": "hex_3"}
        resources = {"wood": 50.0, "stone": 25.0}
        movement_options = ["hex_north", "hex_south", "hex_east"]
        env_conditions = {"temperature": 20, "humidity": 60}
        recent_events = [WorldEvent(EventType.AGENT_MOVED, "hex_1")]

        perception = Perception(
            current_location="hex_1",
            visible_cells=visible_cells,
            nearby_agents=nearby_agents,
            available_resources=resources,
            movement_options=movement_options,
            environmental_conditions=env_conditions,
            recent_events=recent_events
        )

        assert perception.current_location == "hex_1"
        assert perception.visible_cells == visible_cells
        assert perception.nearby_agents == nearby_agents
        assert perception.available_resources == resources
        assert perception.movement_options == movement_options
        assert perception.environmental_conditions == env_conditions
        assert perception.recent_events == recent_events
        assert isinstance(perception.timestamp, datetime)
        assert perception.timestamp.tzinfo == timezone.utc

    def test_perception_empty_collections(self):
        """Test perception with empty collections"""
        perception = Perception(
            current_location="hex_empty",
            visible_cells=[],
            nearby_agents={},
            available_resources={},
            movement_options=[],
            environmental_conditions={},
            recent_events=[]
        )
        assert len(perception.visible_cells) == 0
        assert len(perception.nearby_agents) == 0
        assert len(perception.available_resources) == 0
        assert len(perception.movement_options) == 0
        assert len(perception.environmental_conditions) == 0
        assert len(perception.recent_events) == 0


class TestActionResult:
    """Test ActionResult dataclass"""

    def test_action_result_success(self):
        """Test successful action result"""
        events = [WorldEvent(EventType.AGENT_MOVED, "hex_1")]
        effects = {"new_location": "hex_2"}
        
        result = ActionResult(
            success=True,
            action_type=ActionType.MOVE,
            cost=10.0,
            effects=effects,
            generated_events=events,
            message="Successfully moved"
        )
        
        assert result.success is True
        assert result.action_type == ActionType.MOVE
        assert result.cost == 10.0
        assert result.effects == effects
        assert result.generated_events == events
        assert result.message == "Successfully moved"

    def test_action_result_failure(self):
        """Test failed action result"""
        result = ActionResult(
            success=False,
            action_type=ActionType.HARVEST_RESOURCE,
            message="No resources available"
        )
        
        assert result.success is False
        assert result.action_type == ActionType.HARVEST_RESOURCE
        assert result.cost == 0.0
        assert result.effects == {}
        assert result.generated_events == []
        assert result.message == "No resources available"

    def test_action_result_defaults(self):
        """Test action result with default values"""
        result = ActionResult(
            success=True,
            action_type=ActionType.OBSERVE
        )
        
        assert result.cost == 0.0
        assert result.effects == {}
        assert result.generated_events == []
        assert result.message == ""


class TestWorldEventSystem:
    """Test WorldEventSystem implementation"""

    def setup_method(self):
        """Set up test fixtures"""
        self.event_system = WorldEventSystem()

    def test_initialization(self):
        """Test event system initialization"""
        assert len(self.event_system.subscribers) == 0
        assert len(self.event_system.event_history) == 0
        assert self.event_system.max_history_size == 10000

    def test_subscribe_to_events(self):
        """Test subscribing agents to events"""
        callback = Mock()
        event_types = [EventType.AGENT_MOVED, EventType.RESOURCE_DEPLETED]
        
        self.event_system.subscribe_to_events("agent_1", event_types, callback)
        
        assert "agent_1" in self.event_system.subscribers[EventType.AGENT_MOVED]
        assert "agent_1" in self.event_system.subscribers[EventType.RESOURCE_DEPLETED]
        assert self.event_system.subscribers[EventType.AGENT_MOVED]["agent_1"] == callback

    def test_unsubscribe_from_events(self):
        """Test unsubscribing agents from events"""
        callback = Mock()
        event_types = [EventType.AGENT_MOVED, EventType.RESOURCE_DEPLETED]
        
        # Subscribe first
        self.event_system.subscribe_to_events("agent_1", event_types, callback)
        
        # Then unsubscribe
        self.event_system.unsubscribe_from_events("agent_1", [EventType.AGENT_MOVED])
        
        assert "agent_1" not in self.event_system.subscribers[EventType.AGENT_MOVED]
        assert "agent_1" in self.event_system.subscribers[EventType.RESOURCE_DEPLETED]

    def test_unsubscribe_nonexistent_agent(self):
        """Test unsubscribing agent that wasn't subscribed"""
        # Should not raise exception
        self.event_system.unsubscribe_from_events("nonexistent", [EventType.AGENT_MOVED])

    def test_publish_event_no_subscribers(self):
        """Test publishing event with no subscribers"""
        event = WorldEvent(
            event_type=EventType.WEATHER_CHANGED,
            location="hex_1"
        )
        
        # Should not raise exception
        self.event_system.publish_event(event)
        
        assert len(self.event_system.event_history) == 1
        assert self.event_system.event_history[0] == event

    def test_publish_event_with_subscribers(self):
        """Test publishing event with subscribers"""
        callback1 = Mock()
        callback2 = Mock()
        
        self.event_system.subscribe_to_events("agent_1", [EventType.AGENT_MOVED], callback1)
        self.event_system.subscribe_to_events("agent_2", [EventType.AGENT_MOVED], callback2)
        
        event = WorldEvent(
            event_type=EventType.AGENT_MOVED,
            location="hex_1",
            agent_id="agent_3"
        )
        
        self.event_system.publish_event(event)
        
        callback1.assert_called_once_with(event)
        callback2.assert_called_once_with(event)

    def test_publish_event_with_affected_agents(self):
        """Test publishing event with specific affected agents"""
        callback1 = Mock()
        callback2 = Mock()
        callback3 = Mock()
        
        self.event_system.subscribe_to_events("agent_1", [EventType.RESOURCE_DISCOVERED], callback1)
        self.event_system.subscribe_to_events("agent_2", [EventType.RESOURCE_DISCOVERED], callback2)
        self.event_system.subscribe_to_events("agent_3", [EventType.RESOURCE_DISCOVERED], callback3)
        
        event = WorldEvent(
            event_type=EventType.RESOURCE_DISCOVERED,
            location="hex_1",
            agent_id="agent_1",
            affected_agents={"agent_2"}  # Only agent_2 should be notified
        )
        
        self.event_system.publish_event(event)
        
        callback1.assert_called_once_with(event)  # Called because agent_1 is event.agent_id
        callback2.assert_called_once_with(event)  # Called because agent_2 is in affected_agents
        callback3.assert_not_called()  # Should not be called

    def test_publish_event_callback_exception(self):
        """Test publishing event when callback raises exception"""
        callback_good = Mock()
        callback_bad = Mock(side_effect=Exception("Callback error"))
        
        self.event_system.subscribe_to_events("agent_1", [EventType.AGENT_MOVED], callback_good)
        self.event_system.subscribe_to_events("agent_2", [EventType.AGENT_MOVED], callback_bad)
        
        event = WorldEvent(
            event_type=EventType.AGENT_MOVED,
            location="hex_1"
        )
        
        # Should not raise exception, should continue with other callbacks
        self.event_system.publish_event(event)
        
        callback_good.assert_called_once_with(event)
        callback_bad.assert_called_once_with(event)

    def test_event_history_management(self):
        """Test event history size management"""
        # Set a small max history size for testing
        self.event_system.max_history_size = 3
        
        # Add events beyond the limit
        for i in range(5):
            event = WorldEvent(
                event_type=EventType.AGENT_MOVED,
                location=f"hex_{i}"
            )
            self.event_system.publish_event(event)
        
        # Should only keep the last 3 events
        assert len(self.event_system.event_history) == 3
        assert self.event_system.event_history[0].location == "hex_2"
        assert self.event_system.event_history[-1].location == "hex_4"

    def test_get_recent_events(self):
        """Test getting recent events near a location"""
        # Create events at different times
        old_event = WorldEvent(
            event_type=EventType.AGENT_MOVED,
            location="hex_1"
        )
        # Manually set old timestamp
        old_event.timestamp = datetime.now(timezone.utc).replace(hour=0, minute=0)
        
        recent_event = WorldEvent(
            event_type=EventType.RESOURCE_DISCOVERED,
            location="hex_1"
        )
        
        self.event_system.event_history = [old_event, recent_event]
        
        # Get events from the last 10 minutes
        recent_events = self.event_system.get_recent_events("hex_1", time_window_minutes=10)
        
        # Should only get the recent event
        assert len(recent_events) == 1
        assert recent_events[0] == recent_event

    def test_get_recent_events_different_location(self):
        """Test getting recent events filters by location"""
        event1 = WorldEvent(
            event_type=EventType.AGENT_MOVED,
            location="hex_1"
        )
        event2 = WorldEvent(
            event_type=EventType.AGENT_MOVED,
            location="hex_2"
        )
        
        self.event_system.event_history = [event1, event2]
        
        recent_events = self.event_system.get_recent_events("hex_1")
        
        assert len(recent_events) == 1
        assert recent_events[0] == event1

    def test_get_recent_events_empty_history(self):
        """Test getting recent events with empty history"""
        recent_events = self.event_system.get_recent_events("hex_1")
        assert len(recent_events) == 0


class TestAgentWorldManager:
    """Test AgentWorldManager implementation"""

    def setup_method(self):
        """Set up test fixtures"""
        # Mock world and spatial API
        self.mock_world = Mock()
        self.mock_spatial_api = Mock()
        
        # Create manager with mocks
        with patch('agents.base.world_integration.SpatialAPI') as mock_spatial_class:
            mock_spatial_class.return_value = self.mock_spatial_api
            self.manager = AgentWorldManager(self.mock_world, self.mock_spatial_api)

    def test_initialization(self):
        """Test manager initialization"""
        assert self.manager.world == self.mock_world
        assert self.manager.spatial_api == self.mock_spatial_api
        assert isinstance(self.manager.event_system, WorldEventSystem)
        assert len(self.manager.agent_locations) == 0
        assert len(self.manager.agent_resources) == 0
        assert len(self.manager.agent_energy) == 0

    def test_place_agent_success(self):
        """Test successfully placing an agent"""
        mock_cell = Mock()
        self.mock_world.get_cell.return_value = mock_cell
        
        result = self.manager.place_agent("agent_1", "hex_123")
        
        assert result is True
        assert self.manager.agent_locations["agent_1"] == "hex_123"
        assert self.manager.agent_energy["agent_1"] == 100.0
        self.mock_world.get_cell.assert_called_once_with("hex_123")

    def test_place_agent_invalid_location(self):
        """Test placing agent at invalid location"""
        self.mock_world.get_cell.return_value = None
        
        result = self.manager.place_agent("agent_1", "invalid_hex")
        
        assert result is False
        assert "agent_1" not in self.manager.agent_locations

    def test_place_agent_move_existing(self):
        """Test moving an existing agent to new location"""
        mock_cell = Mock()
        self.mock_world.get_cell.return_value = mock_cell
        
        # Place agent initially
        self.manager.place_agent("agent_1", "hex_1")
        
        # Move to new location
        result = self.manager.place_agent("agent_1", "hex_2")
        
        assert result is True
        assert self.manager.agent_locations["agent_1"] == "hex_2"

    def test_remove_agent(self):
        """Test removing an agent"""
        # Setup agent first
        self.manager.agent_locations["agent_1"] = "hex_1"
        self.manager.agent_resources["agent_1"] = {"wood": 10.0}
        self.manager.agent_energy["agent_1"] = 50.0
        
        self.manager.remove_agent("agent_1")
        
        assert "agent_1" not in self.manager.agent_locations
        assert "agent_1" not in self.manager.agent_resources
        assert "agent_1" not in self.manager.agent_energy

    def test_remove_nonexistent_agent(self):
        """Test removing agent that doesn't exist"""
        # Should not raise exception
        self.manager.remove_agent("nonexistent")

    def test_perceive_environment_success(self):
        """Test successful environment perception"""
        # Setup agent
        self.manager.agent_locations["agent_1"] = "hex_1"
        
        # Mock world responses
        mock_cell = Mock()
        mock_cell.resources = {"wood": 50.0}
        mock_cell.temperature = 20
        mock_cell.moisture = 60
        mock_cell.elevation = 100
        mock_cell.biome = Mock(value="forest")
        mock_cell.terrain = Mock(value="grassland")
        
        mock_neighbor = Mock()
        mock_neighbor.hex_id = "hex_2"
        mock_neighbor.movement_cost = 1.0
        
        visible_cells = [mock_cell, mock_neighbor]
        
        self.mock_world.get_cell.return_value = mock_cell
        self.mock_world.get_visible_cells.return_value = visible_cells
        self.mock_world.get_neighbors.return_value = [mock_neighbor]
        
        # Add another agent in visible area
        self.manager.agent_locations["agent_2"] = "hex_2"
        
        perception = self.manager.perceive_environment("agent_1")
        
        assert perception is not None
        assert perception.current_location == "hex_1"
        assert perception.visible_cells == visible_cells
        assert "agent_2" in perception.nearby_agents
        assert perception.available_resources == {"wood": 50.0}
        assert "hex_2" in perception.movement_options
        assert perception.environmental_conditions["temperature"] == 20
        assert perception.environmental_conditions["biome"] == "forest"

    def test_perceive_environment_agent_not_found(self):
        """Test perception for nonexistent agent"""
        perception = self.manager.perceive_environment("nonexistent")
        assert perception is None

    def test_perceive_environment_invalid_cell(self):
        """Test perception when agent's cell is invalid"""
        self.manager.agent_locations["agent_1"] = "hex_1"
        self.mock_world.get_cell.return_value = None
        
        perception = self.manager.perceive_environment("agent_1")
        assert perception is None

    def test_get_available_actions_basic(self):
        """Test getting available actions for agent"""
        self.manager.agent_locations["agent_1"] = "hex_1"
        
        mock_cell = Mock()
        mock_cell.resources = {"wood": 10.0}
        
        mock_neighbor = Mock()
        
        self.mock_world.get_neighbors.return_value = [mock_neighbor]
        self.mock_world.get_cell.return_value = mock_cell
        
        actions = self.manager.get_available_actions("agent_1")
        
        assert ActionType.OBSERVE in actions
        assert ActionType.COMMUNICATE in actions
        assert ActionType.MOVE in actions
        assert ActionType.HARVEST_RESOURCE in actions
        assert ActionType.BUILD_STRUCTURE in actions
        assert ActionType.TRADE in actions

    def test_get_available_actions_with_resources(self):
        """Test available actions when agent has resources"""
        self.manager.agent_locations["agent_1"] = "hex_1"
        self.manager.agent_resources["agent_1"] = {"stone": 5.0}
        
        mock_cell = Mock()
        mock_cell.resources = {}
        
        self.mock_world.get_neighbors.return_value = []
        self.mock_world.get_cell.return_value = mock_cell
        
        actions = self.manager.get_available_actions("agent_1")
        
        assert ActionType.DEPOSIT_RESOURCE in actions

    def test_get_available_actions_no_agent(self):
        """Test available actions for nonexistent agent"""
        actions = self.manager.get_available_actions("nonexistent")
        assert actions == []

    def test_perform_action_agent_not_found(self):
        """Test performing action for nonexistent agent"""
        result = self.manager.perform_action("nonexistent", ActionType.MOVE, {})
        
        assert result.success is False
        assert "Agent not found" in result.message

    def test_perform_action_insufficient_energy(self):
        """Test performing action with insufficient energy"""
        self.manager.agent_locations["agent_1"] = "hex_1"
        self.manager.agent_energy["agent_1"] = 5.0  # Less than move cost
        
        # Mock the target cell with proper movement_cost
        mock_target_cell = Mock()
        mock_target_cell.movement_cost = 1.0  # Fix: set as float, not Mock
        self.mock_world.get_cell.return_value = mock_target_cell
        
        result = self.manager.perform_action("agent_1", ActionType.MOVE, {"target_hex": "hex_2"})
        
        assert result.success is False
        assert "Insufficient energy" in result.message

    def test_perform_move_action_success(self):
        """Test successful move action"""
        self.manager.agent_locations["agent_1"] = "hex_1"
        self.manager.agent_energy["agent_1"] = 100.0
        
        mock_neighbor = Mock()
        mock_neighbor.hex_id = "hex_2"
        
        self.mock_world.get_neighbors.return_value = [mock_neighbor]
        
        # Mock get_cell for cost calculation
        mock_cell = Mock()
        mock_cell.movement_cost = 1.0
        self.mock_world.get_cell.return_value = mock_cell
        
        result = self.manager.perform_action("agent_1", ActionType.MOVE, {"target_hex": "hex_2"})
        
        assert result.success is True
        assert self.manager.agent_locations["agent_1"] == "hex_2"
        assert self.manager.agent_energy["agent_1"] < 100.0  # Energy consumed
        assert len(result.generated_events) == 1
        assert result.generated_events[0].event_type == EventType.AGENT_MOVED

    def test_perform_move_action_no_target(self):
        """Test move action without target hex"""
        self.manager.agent_locations["agent_1"] = "hex_1"
        self.manager.agent_energy["agent_1"] = 100.0
        
        result = self.manager.perform_action("agent_1", ActionType.MOVE, {})
        
        assert result.success is False
        assert "No target_hex specified" in result.message

    def test_perform_move_action_invalid_target(self):
        """Test move action to invalid target"""
        self.manager.agent_locations["agent_1"] = "hex_1"
        self.manager.agent_energy["agent_1"] = 100.0
        
        # Mock the target cell with proper movement_cost for cost calculation
        mock_target_cell = Mock()
        mock_target_cell.movement_cost = 1.0  # Fix: set as float, not Mock
        self.mock_world.get_cell.return_value = mock_target_cell
        
        # No neighbors available
        self.mock_world.get_neighbors.return_value = []
        
        result = self.manager.perform_action("agent_1", ActionType.MOVE, {"target_hex": "hex_2"})
        
        assert result.success is False
        assert "not adjacent or invalid" in result.message

    def test_perform_harvest_action_success(self):
        """Test successful harvest action"""
        self.manager.agent_locations["agent_1"] = "hex_1"
        self.manager.agent_energy["agent_1"] = 100.0
        
        mock_cell = Mock()
        mock_cell.resources = {"wood": 50.0}
        self.mock_world.get_cell.return_value = mock_cell
        
        result = self.manager.perform_action(
            "agent_1", 
            ActionType.HARVEST_RESOURCE, 
            {"resource_type": "wood", "amount": 10.0}
        )
        
        assert result.success is True
        assert self.manager.agent_resources["agent_1"]["wood"] == 10.0
        assert mock_cell.resources["wood"] == 40.0
        assert result.effects["harvested"] == 10.0

    def test_perform_harvest_action_no_resource_type(self):
        """Test harvest action without resource type"""
        self.manager.agent_locations["agent_1"] = "hex_1"
        self.manager.agent_energy["agent_1"] = 100.0
        
        result = self.manager.perform_action("agent_1", ActionType.HARVEST_RESOURCE, {})
        
        assert result.success is False
        assert "No resource_type specified" in result.message

    def test_perform_harvest_action_resource_not_available(self):
        """Test harvest action for unavailable resource"""
        self.manager.agent_locations["agent_1"] = "hex_1"
        self.manager.agent_energy["agent_1"] = 100.0
        
        mock_cell = Mock()
        mock_cell.resources = {}
        self.mock_world.get_cell.return_value = mock_cell
        
        result = self.manager.perform_action(
            "agent_1", 
            ActionType.HARVEST_RESOURCE, 
            {"resource_type": "gold"}
        )
        
        assert result.success is False
        assert "not available at location" in result.message

    def test_perform_harvest_action_resource_depleted(self):
        """Test harvest action that depletes resource"""
        self.manager.agent_locations["agent_1"] = "hex_1"
        self.manager.agent_energy["agent_1"] = 100.0
        
        mock_cell = Mock()
        mock_cell.resources = {"wood": 10.0}
        self.mock_world.get_cell.return_value = mock_cell
        
        result = self.manager.perform_action(
            "agent_1", 
            ActionType.HARVEST_RESOURCE, 
            {"resource_type": "wood", "amount": 10.0}
        )
        
        assert result.success is True
        assert len(result.generated_events) == 1
        assert result.generated_events[0].event_type == EventType.RESOURCE_DEPLETED

    def test_perform_observe_action(self):
        """Test observe action"""
        self.manager.agent_locations["agent_1"] = "hex_1"
        self.manager.agent_energy["agent_1"] = 100.0
        
        # Mock perception
        mock_cell = Mock()
        mock_cell.resources = {}
        mock_cell.temperature = 20
        mock_cell.moisture = 60
        mock_cell.elevation = 100
        mock_cell.biome = Mock(value="forest")
        mock_cell.terrain = Mock(value="grassland")
        
        self.mock_world.get_cell.return_value = mock_cell
        self.mock_world.get_visible_cells.return_value = [mock_cell]
        self.mock_world.get_neighbors.return_value = []
        
        result = self.manager.perform_action("agent_1", ActionType.OBSERVE, {})
        
        assert result.success is True
        assert "perception" in result.effects
        assert self.manager.agent_energy["agent_1"] < 100.0

    def test_perform_communicate_action(self):
        """Test communicate action"""
        self.manager.agent_locations["agent_1"] = "hex_1"
        self.manager.agent_energy["agent_1"] = 100.0
        
        result = self.manager.perform_action(
            "agent_1", 
            ActionType.COMMUNICATE, 
            {"target_agent": "agent_2", "message": "Hello"}
        )
        
        assert result.success is True
        assert len(result.generated_events) == 1
        assert result.generated_events[0].event_type == EventType.AGENT_INTERACTION
        assert "agent_2" in result.generated_events[0].affected_agents

    def test_perform_build_action_success(self):
        """Test successful build action"""
        self.manager.agent_locations["agent_1"] = "hex_1"
        self.manager.agent_energy["agent_1"] = 100.0
        
        result = self.manager.perform_action(
            "agent_1", 
            ActionType.BUILD_STRUCTURE, 
            {"structure_type": "shelter"}
        )
        
        assert result.success is True
        assert "shelter" in self.manager.structures["hex_1"]
        assert len(result.generated_events) == 1
        assert result.generated_events[0].event_type == EventType.STRUCTURE_BUILT

    def test_perform_build_action_structure_exists(self):
        """Test build action when structure already exists"""
        self.manager.agent_locations["agent_1"] = "hex_1"
        self.manager.agent_energy["agent_1"] = 100.0
        self.manager.structures["hex_1"]["shelter"] = {"built_by": "agent_1"}
        
        result = self.manager.perform_action(
            "agent_1", 
            ActionType.BUILD_STRUCTURE, 
            {"structure_type": "shelter"}
        )
        
        assert result.success is False
        assert "already exists" in result.message

    def test_perform_unsupported_action(self):
        """Test performing unsupported action type"""
        self.manager.agent_locations["agent_1"] = "hex_1"
        self.manager.agent_energy["agent_1"] = 100.0
        
        result = self.manager.perform_action("agent_1", ActionType.TRADE, {})
        
        assert result.success is False
        assert "not implemented" in result.message

    def test_calculate_action_cost_basic(self):
        """Test basic action cost calculation"""
        self.manager.agent_locations["agent_1"] = "hex_1"
        
        cost = self.manager._calculate_action_cost("agent_1", ActionType.MOVE, {})
        assert cost == 10.0

    def test_calculate_action_cost_with_terrain(self):
        """Test action cost calculation with terrain modifier"""
        self.manager.agent_locations["agent_1"] = "hex_1"
        
        mock_current_cell = Mock()
        mock_target_cell = Mock()
        mock_target_cell.movement_cost = 2.0
        
        self.mock_world.get_cell.side_effect = [mock_current_cell, mock_target_cell]
        
        cost = self.manager._calculate_action_cost(
            "agent_1", 
            ActionType.MOVE, 
            {"target_hex": "hex_2"}
        )
        assert cost == 20.0  # 10.0 * 2.0

    def test_get_agent_energy(self):
        """Test getting agent energy"""
        energy = self.manager.get_agent_energy("agent_1")
        assert energy == 100.0  # Default value
        
        self.manager.agent_energy["agent_1"] = 75.0
        energy = self.manager.get_agent_energy("agent_1")
        assert energy == 75.0

    def test_restore_agent_energy(self):
        """Test restoring agent energy"""
        self.manager.agent_energy["agent_1"] = 50.0
        
        self.manager.restore_agent_energy("agent_1", 30.0)
        assert self.manager.agent_energy["agent_1"] == 80.0
        
        # Test max cap
        self.manager.restore_agent_energy("agent_1", 50.0)
        assert self.manager.agent_energy["agent_1"] == 100.0

    def test_get_agent_resources(self):
        """Test getting agent resources"""
        resources = self.manager.get_agent_resources("agent_1")
        assert resources == {}
        
        self.manager.agent_resources["agent_1"] = {"wood": 10.0, "stone": 5.0}
        resources = self.manager.get_agent_resources("agent_1")
        assert resources == {"wood": 10.0, "stone": 5.0}
        
        # Test that returned dict is a copy
        resources["wood"] = 20.0
        assert self.manager.agent_resources["agent_1"]["wood"] == 10.0

    def test_get_world_state_summary(self):
        """Test getting world state summary"""
        # Setup some state
        self.manager.agent_locations["agent_1"] = "hex_1"
        self.manager.agent_locations["agent_2"] = "hex_2"
        self.manager.agent_energy["agent_1"] = 80.0
        self.manager.agent_energy["agent_2"] = 90.0
        self.manager.structures["hex_1"]["shelter"] = {}
        self.manager.structures["hex_2"]["farm"] = {}
        self.manager.structures["hex_2"]["well"] = {}
        self.manager.event_system.event_history = [
            WorldEvent(EventType.AGENT_MOVED, "hex_1"),
            WorldEvent(EventType.STRUCTURE_BUILT, "hex_2")
        ]
        
        summary = self.manager.get_world_state_summary()
        
        assert summary["num_agents"] == 2
        assert summary["total_energy"] == 170.0
        assert summary["num_structures"] == 3
        assert summary["modified_cells"] == 0
        assert summary["recent_events"] == 2

    def test_perform_deposit_action_success(self):
        """Test successful deposit action"""
        self.manager.agent_locations["agent_1"] = "hex_1"
        self.manager.agent_energy["agent_1"] = 100.0
        self.manager.agent_resources["agent_1"] = {"wood": 20.0}
        
        mock_cell = Mock()
        mock_cell.resources = {"wood": 5.0}
        self.mock_world.get_cell.return_value = mock_cell
        
        result = self.manager.perform_action(
            "agent_1", 
            ActionType.DEPOSIT_RESOURCE, 
            {"resource_type": "wood", "amount": 10.0}
        )
        
        assert result.success is True
        assert self.manager.agent_resources["agent_1"]["wood"] == 10.0
        assert mock_cell.resources["wood"] == 15.0
        assert result.effects["deposited"] == 10.0

    def test_perform_deposit_action_insufficient_resources(self):
        """Test deposit action with insufficient resources"""
        self.manager.agent_locations["agent_1"] = "hex_1"
        self.manager.agent_energy["agent_1"] = 100.0
        self.manager.agent_resources["agent_1"] = {"wood": 5.0}
        
        result = self.manager.perform_action(
            "agent_1", 
            ActionType.DEPOSIT_RESOURCE, 
            {"resource_type": "wood", "amount": 10.0}
        )
        
        assert result.success is False
        assert "Insufficient" in result.message