"""Characterization tests for Agent Manager following Michael Feathers' principles.

Documents the current behavior of:
- Agent lifecycle management
- World creation and integration
- Agent positioning and coordination
- Event queue processing
- Error handling patterns
- PyMDP integration behavior
"""

import logging
import threading
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from typing import Dict, Any

import pytest
import numpy as np

from agents.agent_manager import AgentManager
from agents.base_agent import ActiveInferenceAgent, BasicExplorerAgent
from agents.agent_adapter import ActiveInferenceGridAdapter
from world.grid_world import (
    GridWorld, 
    GridWorldConfig, 
    Position, 
    CellType,
    Cell,
)


class TestAgentManagerCharacterization:
    """Characterize AgentManager behavior."""
    
    def test_initialization_state(self):
        """Document initial state of AgentManager."""
        # When
        manager = AgentManager()
        
        # Then - Document initial state
        assert isinstance(manager.agents, dict)
        assert len(manager.agents) == 0
        assert manager.world is None
        assert isinstance(manager.adapter, ActiveInferenceGridAdapter)
        assert manager.running is False
        assert manager._agent_counter == 0
        assert hasattr(manager, '_executor')
        assert hasattr(manager, '_event_queue')
        assert isinstance(manager._event_queue, list)
        assert hasattr(manager, '_event_lock')
        assert isinstance(manager._event_lock, type(threading.Lock()))
        
    def test_world_creation_behavior(self):
        """Document world creation behavior."""
        # Given
        manager = AgentManager()
        
        # When - Create default world
        world = manager.create_world()
        
        # Then
        assert isinstance(world, GridWorld)
        assert world.width == 10
        assert world.height == 10
        assert manager.world is world
        
        # When - Create custom size world
        world2 = manager.create_world(size=20)
        
        # Then
        assert world2.width == 20
        assert world2.height == 20
        assert manager.world is world2  # Replaces previous world
        
    def test_agent_id_generation_pattern(self):
        """Document agent ID generation patterns."""
        # Given
        manager = AgentManager()
        manager.create_world()
        
        # When - Create active inference agent
        id1 = manager.create_agent("active_inference", "Agent 1")
        
        # Then
        assert id1 == "ai_agent_1"
        assert manager._agent_counter == 1
        
        # When - Create explorer agent
        id2 = manager.create_agent("explorer", "Agent 2")
        
        # Then 
        assert id2 == "test_agent_2"
        assert manager._agent_counter == 2
        
        # When - Create basic agent
        id3 = manager.create_agent("basic", "Agent 3")
        
        # Then
        assert id3 == "test_agent_3"
        assert manager._agent_counter == 3
        
    def test_agent_creation_with_kwargs(self):
        """Document how kwargs are passed to agents."""
        # Given
        manager = AgentManager()
        manager.create_world()
        
        # When - Create agent with PyMDP parameters
        agent_id = manager.create_agent(
            "active_inference",
            "Test Agent",
            num_states=[10, 10],
            num_obs=[5, 5],
            num_controls=4,
            num_actions=4
        )
        
        # Then
        agent = manager.agents[agent_id]
        assert hasattr(agent, 'num_states')
        assert agent.num_states == [10, 10]
        assert hasattr(agent, 'num_obs')
        assert agent.num_obs == [5, 5]
        assert hasattr(agent, 'num_controls')
        assert agent.num_controls == 4
        assert hasattr(agent, 'num_actions')
        assert agent.num_actions == 4
        
    def test_agent_positioning_behavior(self):
        """Document agent positioning logic."""
        # Given
        manager = AgentManager()
        world = manager.create_world(size=3)  # Small world for testing
        
        # When - Create agent without position
        agent_id1 = manager.create_agent("explorer", "Agent 1")
        
        # Then - Placed at first free position
        grid_agent1 = world.agents.get(agent_id1)
        assert grid_agent1 is not None
        assert grid_agent1.position == Position(0, 0)  # First free position
        
        # When - Create agent with specific position
        agent_id2 = manager.create_agent(
            "explorer", 
            "Agent 2",
            position=(2, 1)
        )
        
        # Then - Placed at requested position
        grid_agent2 = world.agents.get(agent_id2)
        assert grid_agent2 is not None
        assert grid_agent2.position == Position(2, 1)
        
    def test_find_free_position_algorithm(self):
        """Document free position finding algorithm."""
        # Given
        manager = AgentManager()
        world = manager.create_world(size=2)  # 2x2 world
        
        # Mock world state
        world.agents = {
            "agent1": Mock(position=Position(0, 0)),
            "agent2": Mock(position=Position(1, 0)),
        }
        
        # Mock get_cell to return walkable cells
        def mock_get_cell(pos):
            cell = Mock()
            cell.type = CellType.EMPTY
            return cell
        world.get_cell = mock_get_cell
        
        # When
        free_pos = manager._find_free_position()
        
        # Then - Finds first unoccupied position
        assert free_pos == Position(0, 1)  # (1,1) is occupied, so (0,1)
        
    def test_find_free_position_fallback(self):
        """Document fallback behavior when no free positions."""
        # Given
        manager = AgentManager()
        world = manager.create_world(size=1)  # 1x1 world
        
        # Fill the single position
        world.agents = {"agent1": Mock(position=Position(0, 0))}
        
        # When
        free_pos = manager._find_free_position()
        
        # Then - Falls back to (0,0)
        assert free_pos == Position(0, 0)
        
    def test_unknown_agent_type_handling(self):
        """Document behavior for unknown agent types."""
        # Given
        manager = AgentManager()
        manager.create_world()
        
        # When
        agent_id = manager.create_agent("unknown_type", "Test Agent")
        
        # Then
        assert agent_id is None
        assert len(manager.agents) == 0
        
    def test_agent_creation_failure_handling(self):
        """Document error handling during agent creation."""
        # Given
        manager = AgentManager()
        manager.create_world()
        
        # Mock BasicExplorerAgent to raise exception
        with patch('agents.agent_manager.BasicExplorerAgent') as mock_agent_class:
            mock_agent_class.side_effect = Exception("Creation failed")
            
            # When
            agent_id = manager.create_agent("explorer", "Test Agent")
            
            # Then
            assert agent_id is None
            assert len(manager.agents) == 0
            
    def test_event_queue_mechanism(self):
        """Document event queue behavior."""
        # Given
        manager = AgentManager()
        manager.create_world()
        
        # Clear any initial events
        manager._event_queue.clear()
        
        # When - Create agent (triggers event)
        agent_id = manager.create_agent("explorer", "Test Agent")
        
        # Then - Event is queued
        with manager._event_lock:
            assert len(manager._event_queue) >= 1
            event = manager._event_queue[-1]
            
        assert event['agent_id'] == agent_id
        assert event['event_type'] == 'created'
        assert 'agent_type' in event['data']
        assert event['data']['agent_type'] == 'explorer'
        assert event['data']['name'] == 'Test Agent'
        assert 'timestamp' in event['data']
        
    def test_adapter_integration(self):
        """Document adapter integration behavior."""
        # Given
        manager = AgentManager()
        manager.create_world()
        
        # Mock adapter
        mock_grid_agent = Mock()
        manager.adapter.register_agent = Mock(return_value=mock_grid_agent)
        
        # When
        agent_id = manager.create_agent("explorer", "Test Agent")
        
        # Then - Adapter is used
        manager.adapter.register_agent.assert_called_once()
        call_args = manager.adapter.register_agent.call_args
        
        # Verify agent and position passed to adapter
        assert isinstance(call_args[0][0], BasicExplorerAgent)
        assert isinstance(call_args[0][1], Position)
        
        # Verify grid agent added to world
        assert agent_id in manager.world.agents
        assert manager.world.agents[agent_id] == mock_grid_agent


class TestAgentLifecycleCharacterization:
    """Characterize agent lifecycle behavior."""
    
    def test_agent_removal_behavior(self):
        """Document agent removal behavior."""
        # Given
        manager = AgentManager()
        manager.create_world()
        agent_id = manager.create_agent("explorer", "Test Agent")
        
        # When
        manager.remove_agent(agent_id)
        
        # Then
        assert agent_id not in manager.agents
        assert agent_id not in manager.world.agents
        
        # Verify event queued
        with manager._event_lock:
            events = [e for e in manager._event_queue if e['event_type'] == 'removed']
        assert len(events) >= 1
        assert events[-1]['agent_id'] == agent_id
        
    def test_agent_action_execution(self):
        """Document agent action execution flow."""
        # Given
        manager = AgentManager()
        manager.create_world()
        agent_id = manager.create_agent("explorer", "Test Agent")
        
        # Mock agent action
        agent = manager.agents[agent_id]
        agent.act = Mock(return_value=1)  # Return action 1
        
        # When
        result = manager.execute_agent_action(agent_id, observation=0)
        
        # Then
        assert result['action'] == 1
        assert result['agent_id'] == agent_id
        agent.act.assert_called_once_with(0)
        
    def test_multi_agent_step_execution(self):
        """Document multi-agent step execution."""
        # Given
        manager = AgentManager()
        manager.create_world()
        
        # Create multiple agents
        agent1_id = manager.create_agent("explorer", "Agent 1")
        agent2_id = manager.create_agent("explorer", "Agent 2")
        
        # Mock agent actions
        manager.agents[agent1_id].act = Mock(return_value=0)
        manager.agents[agent2_id].act = Mock(return_value=1)
        
        # Mock world step
        manager.world.step = Mock(return_value={
            agent1_id: {'observation': 1, 'reward': 0.5},
            agent2_id: {'observation': 2, 'reward': 0.7}
        })
        
        # When
        results = manager.step()
        
        # Then
        assert len(results) == 2
        assert agent1_id in results
        assert agent2_id in results
        
        # Verify world.step was called with actions
        manager.world.step.assert_called_once()
        actions_dict = manager.world.step.call_args[0][0]
        assert actions_dict[agent1_id] == 0
        assert actions_dict[agent2_id] == 1


class TestActiveInferenceIntegrationCharacterization:
    """Characterize Active Inference specific behavior."""
    
    @pytest.fixture
    def mock_pymdp(self):
        """Mock PyMDP for testing."""
        with patch('agents.base_agent.PYMDP_AVAILABLE', True):
            with patch('agents.base_agent.PyMDPAgent') as mock_agent:
                # Mock PyMDP agent behavior
                instance = Mock()
                instance.infer_states = Mock(return_value={'qs': np.array([0.5, 0.5])})
                instance.infer_policies = Mock(return_value={'q_pi': np.array([0.6, 0.4])})
                instance.sample_action = Mock(return_value=np.array(1))
                instance.F = 10.5  # Free energy
                mock_agent.return_value = instance
                yield mock_agent
                
    def test_pymdp_agent_creation(self, mock_pymdp):
        """Document PyMDP agent creation behavior."""
        # Given
        manager = AgentManager()
        manager.create_world()
        
        # When
        agent_id = manager.create_agent(
            "active_inference",
            "PyMDP Agent",
            num_states=[10, 10],
            num_obs=[5, 5],
            num_controls=4
        )
        
        # Then
        agent = manager.agents[agent_id]
        assert isinstance(agent, BasicExplorerAgent)
        assert hasattr(agent, 'num_states')
        assert agent.num_states == [10, 10]
        
    def test_belief_update_flow(self, mock_pymdp):
        """Document belief update flow."""
        # Given
        manager = AgentManager()
        manager.create_world()
        agent_id = manager.create_agent("active_inference", "Test Agent")
        agent = manager.agents[agent_id]
        
        # Setup PyMDP agent mock
        agent.pymdp_agent = mock_pymdp.return_value
        
        # When
        agent.update_beliefs(observation=2)
        
        # Then - Verify PyMDP inference called
        agent.pymdp_agent.infer_states.assert_called_once_with(
            2,
            past_actions=None,
            empirical_prior=None,
            qs_seq_pi=None
        )
        
    def test_free_energy_tracking(self, mock_pymdp):
        """Document free energy tracking behavior."""
        # Given
        manager = AgentManager()
        manager.create_world()
        agent_id = manager.create_agent("active_inference", "Test Agent")
        agent = manager.agents[agent_id]
        
        # Setup PyMDP agent
        agent.pymdp_agent = mock_pymdp.return_value
        
        # When - Update beliefs (which calculates free energy)
        agent.update_beliefs(observation=1)
        
        # Then - Free energy is tracked
        assert hasattr(agent, 'free_energy_history')
        assert len(agent.free_energy_history) > 0
        assert agent.free_energy_history[-1] == 10.5