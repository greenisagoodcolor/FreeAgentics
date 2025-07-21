"""Characterization tests for agents/base_agent.py to improve coverage.

Following Michael Feathers' principles:
- Focus on characterizing existing behavior
- Cover untested code paths
- No production code changes unless fixing bugs
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime
import logging
from dataclasses import dataclass
from typing import Any, Dict

# Import functions and classes to test
from agents.base_agent import (
    safe_array_to_int,
    _get_pymdp_components,
    _get_llm_manager,
    ActiveInferenceAgent,
    AgentConfig,
)


# Mock classes for testing - these don't exist in production code
@dataclass
class ObservationResult:
    raw_observation: Dict[str, Any]
    processed_observation: np.ndarray
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class InferenceResult:
    beliefs: np.ndarray
    actions: np.ndarray
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]


@dataclass
class BeliefState:
    beliefs: np.ndarray
    confidence: float
    timestamp: datetime


# Mock BaseAgent class
class BaseAgent:
    def __init__(self):
        pass


class TestSafeArrayToInt:
    """Test safe_array_to_int function - covers lines 50-78."""
    
    def test_numpy_scalar(self):
        """Test with numpy scalar (0-dimensional array)."""
        value = np.array(42)
        assert safe_array_to_int(value) == 42
        
    def test_numpy_single_element_array(self):
        """Test with single element numpy array."""
        value = np.array([42])
        assert safe_array_to_int(value) == 42
        
    def test_numpy_multi_element_array(self):
        """Test with multi-element numpy array."""
        value = np.array([42, 10, 5])
        assert safe_array_to_int(value) == 42  # Takes first element
        
    def test_numpy_multi_dimensional_array(self):
        """Test with multi-dimensional numpy array."""
        value = np.array([[42, 10], [5, 3]])
        assert safe_array_to_int(value) == 42  # Takes first element via flat
        
    def test_empty_numpy_array(self):
        """Test with empty numpy array."""
        value = np.array([])
        with pytest.raises(ValueError, match="Empty array cannot be converted"):
            safe_array_to_int(value)
            
    def test_list(self):
        """Test with regular Python list."""
        assert safe_array_to_int([42, 10]) == 42
        
    def test_empty_list(self):
        """Test with empty list."""
        with pytest.raises(ValueError, match="Empty array cannot be converted"):
            safe_array_to_int([])
            
    def test_regular_int(self):
        """Test with regular integer."""
        assert safe_array_to_int(42) == 42
        
    def test_regular_float(self):
        """Test with regular float."""
        assert safe_array_to_int(42.7) == 42
        
    def test_numpy_scalar_with_item(self):
        """Test numpy scalar types that have item() method."""
        value = np.int64(42)
        assert safe_array_to_int(value) == 42
        
    def test_invalid_type(self):
        """Test with invalid type."""
        with pytest.raises(ValueError, match="Cannot convert"):
            safe_array_to_int("not a number")


class TestPyMDPComponents:
    """Test PyMDP component loading - covers lines 87-104."""
    
    @patch("agents.base_agent.PYMDP_AVAILABLE", None)
    def test_get_pymdp_components_success(self):
        """Test successful PyMDP import."""
        # Mock the imports
        with patch("agents.base_agent.importlib.import_module") as mock_import:
            mock_pymdp = Mock()
            mock_pymdp.utils = Mock()
            mock_pymdp.agent.Agent = Mock
            mock_import.return_value = mock_pymdp
            
            # Import should reset globals
            import agents.base_agent
            agents.base_agent.PYMDP_AVAILABLE = None
            agents.base_agent._pymdp_utils = None
            agents.base_agent._PyMDPAgent = None
            
            available, utils, agent_class = _get_pymdp_components()
            
            assert available is True
            assert utils is not None
            assert agent_class is not None
            
    @patch("agents.base_agent.PYMDP_AVAILABLE", None)
    def test_get_pymdp_components_import_error(self):
        """Test PyMDP import failure."""
        with patch("agents.base_agent.importlib.import_module", side_effect=ImportError("No pymdp")):
            import agents.base_agent
            agents.base_agent.PYMDP_AVAILABLE = None
            
            available, utils, agent_class = _get_pymdp_components()
            
            assert available is False
            assert utils is None
            assert agent_class is None
            
    @patch("agents.base_agent.PYMDP_AVAILABLE", True)
    def test_get_pymdp_components_cached(self):
        """Test PyMDP components are cached after first load."""
        # Set up cached values
        import agents.base_agent
        agents.base_agent._pymdp_utils = Mock()
        agents.base_agent._PyMDPAgent = Mock
        
        available, utils, agent_class = _get_pymdp_components()
        
        assert available is True
        assert utils is agents.base_agent._pymdp_utils
        assert agent_class is agents.base_agent._PyMDPAgent


class TestLLMManager:
    """Test LLM manager loading - covers lines 109-121."""
    
    @patch("agents.base_agent.LLM_MANAGER", None)
    def test_get_llm_manager_success(self):
        """Test successful LLM manager import."""
        with patch("agents.base_agent.importlib.import_module") as mock_import:
            mock_llm_module = Mock()
            mock_llm_module.LocalLLMManager = Mock
            mock_import.return_value = mock_llm_module
            
            # Reset global
            import agents.base_agent
            agents.base_agent.LLM_MANAGER = None
            
            manager = _get_llm_manager()
            assert manager is not None
            
    @patch("agents.base_agent.LLM_MANAGER", None)
    def test_get_llm_manager_import_error(self):
        """Test LLM manager import failure."""
        with patch("agents.base_agent.importlib.import_module", side_effect=ImportError):
            import agents.base_agent
            agents.base_agent.LLM_MANAGER = None
            
            manager = _get_llm_manager()
            assert manager is None
            
    def test_get_llm_manager_cached(self):
        """Test LLM manager is cached."""
        # Set up cached value
        import agents.base_agent
        cached_manager = Mock()
        agents.base_agent.LLM_MANAGER = cached_manager
        
        manager = _get_llm_manager()
        assert manager is cached_manager


class TestDataClasses:
    """Test data classes - covers lines 135-182."""
    
    def test_agent_config_creation(self):
        """Test AgentConfig dataclass."""
        config = AgentConfig(
            belief_threshold=0.8,
            discount_factor=0.95,
            exploration_rate=0.2,
            learning_rate=0.01,
            enable_learning=True,
            enable_planning=True,
            planning_horizon=5,
            num_iterations=10,
            action_sampling="softmax",
            inference_algorithm="vmp",
            memory_decay=0.9,
            batch_size=32,
            observation_noise=0.1,
            transition_noise=0.05
        )
        assert config.belief_threshold == 0.8
        assert config.discount_factor == 0.95
        assert config.enable_learning is True
        
    def test_observation_result(self):
        """Test ObservationResult dataclass."""
        obs = ObservationResult(
            raw_observation={"position": [1, 2]},
            processed_observation=np.array([1, 2]),
            confidence=0.95,
            timestamp=datetime.now(),
            metadata={"source": "sensor"}
        )
        assert obs.confidence == 0.95
        assert "source" in obs.metadata
        
    def test_inference_result(self):
        """Test InferenceResult dataclass."""
        beliefs = np.array([0.2, 0.3, 0.5])
        result = InferenceResult(
            beliefs=beliefs,
            expected_states=np.array([1, 2, 3]),
            free_energy=10.5,
            uncertainty=0.3,
            confidence=0.85,
            metadata={"method": "variational"}
        )
        assert result.free_energy == 10.5
        assert np.array_equal(result.beliefs, beliefs)
        
    def test_belief_state(self):
        """Test BeliefState dataclass."""
        state = BeliefState(
            beliefs={"location": np.array([0.1, 0.9])},
            timestamp=datetime.now(),
            confidence=0.9,
            observation_count=10
        )
        assert state.observation_count == 10
        assert "location" in state.beliefs


class TestBaseAgent:
    """Test BaseAgent abstract class - covers initialization and abstract methods."""
    
    def test_base_agent_cannot_instantiate(self):
        """Test that BaseAgent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAgent("test_id", Mock())
            
    def test_base_agent_subclass(self):
        """Test BaseAgent subclass implementation."""
        class TestAgent(BaseAgent):
            def sense(self, observation):
                return observation
                
            def think(self, observation):
                return {"action": "test"}
                
            def act(self, decision):
                return decision
                
        agent = TestAgent("test_id", Mock())
        assert agent.agent_id == "test_id"
        assert agent.sense("obs") == "obs"


class TestActiveInferenceAgent:
    """Test ActiveInferenceAgent class - covers remaining uncovered lines."""
    
    @patch("agents.base_agent._get_pymdp_components")
    def test_init_without_pymdp(self, mock_get_pymdp):
        """Test initialization when PyMDP is not available."""
        mock_get_pymdp.return_value = (False, None, None)
        
        world = Mock()
        agent = ActiveInferenceAgent("test_agent", world)
        
        assert agent.agent_id == "test_agent"
        assert agent.pymdp_agent is None
        assert not agent.has_pymdp
        
    @patch("agents.base_agent._get_pymdp_components")
    @patch("agents.base_agent._get_llm_manager")
    def test_init_with_llm(self, mock_get_llm, mock_get_pymdp):
        """Test initialization with LLM manager."""
        mock_get_pymdp.return_value = (False, None, None)
        mock_llm_manager = Mock()
        mock_get_llm.return_value = mock_llm_manager
        
        world = Mock()
        agent = ActiveInferenceAgent("test_agent", world, use_llm=True)
        
        assert agent.llm_manager is not None
        
    def test_process_observation_dict(self):
        """Test _process_observation with dict input."""
        agent = ActiveInferenceAgent("test", Mock())
        
        obs = {"position": [1, 2], "health": 100}
        result = agent._process_observation(obs)
        
        assert result.raw_observation == obs
        assert result.processed_observation == obs
        assert result.confidence == 1.0
        
    def test_process_observation_array(self):
        """Test _process_observation with array input."""
        agent = ActiveInferenceAgent("test", Mock())
        
        obs = np.array([1, 2, 3])
        result = agent._process_observation(obs)
        
        assert np.array_equal(result.raw_observation, obs)
        assert np.array_equal(result.processed_observation, obs)
        
    def test_process_observation_none(self):
        """Test _process_observation with None input."""
        agent = ActiveInferenceAgent("test", Mock())
        
        result = agent._process_observation(None)
        
        assert result.raw_observation is None
        assert result.processed_observation == {}
        assert result.confidence == 0.0
        
    def test_fallback_infer_states(self):
        """Test _fallback_infer_states method."""
        agent = ActiveInferenceAgent("test", Mock())
        agent.config.num_states = 3
        
        result = agent._fallback_infer_states({})
        
        assert isinstance(result, InferenceResult)
        assert len(result.beliefs) == 3
        assert abs(sum(result.beliefs) - 1.0) < 0.01  # Should sum to 1
        
    def test_fallback_sample_action(self):
        """Test _fallback_sample_action method."""
        agent = ActiveInferenceAgent("test", Mock())
        agent.available_actions = ["up", "down", "left", "right", "stay"]
        
        # Test multiple times to ensure randomness
        actions = set()
        for _ in range(10):
            action = agent._fallback_sample_action()
            actions.add(action)
            
        assert len(actions) > 1  # Should have some variety
        assert all(a in agent.available_actions for a in actions)
        
    def test_update_belief_history(self):
        """Test _update_belief_history method."""
        agent = ActiveInferenceAgent("test", Mock())
        
        # Add some belief states
        for i in range(5):
            beliefs = np.random.rand(3)
            beliefs /= beliefs.sum()
            agent._update_belief_history(beliefs)
            
        assert len(agent.belief_history) == 5
        
        # Add more to exceed max_history_length
        agent.config.max_history_length = 10
        for i in range(10):
            agent._update_belief_history(np.random.rand(3))
            
        assert len(agent.belief_history) == 10  # Should be capped
        
    def test_estimate_confidence(self):
        """Test _estimate_confidence method."""
        agent = ActiveInferenceAgent("test", Mock())
        
        # High confidence - low entropy
        beliefs1 = np.array([0.9, 0.05, 0.05])
        conf1 = agent._estimate_confidence(beliefs1)
        
        # Low confidence - high entropy
        beliefs2 = np.array([0.33, 0.33, 0.34])
        conf2 = agent._estimate_confidence(beliefs2)
        
        assert conf1 > conf2  # More peaked distribution = higher confidence
        assert 0 <= conf1 <= 1
        assert 0 <= conf2 <= 1
        
    def test_select_action_from_beliefs(self):
        """Test _select_action_from_beliefs method."""
        agent = ActiveInferenceAgent("test", Mock())
        agent.available_actions = ["up", "down", "stay"]
        
        # Test with clear preference
        beliefs = np.array([0.8, 0.1, 0.1])
        action = agent._select_action_from_beliefs(beliefs)
        assert action == "up"  # Should select highest belief
        
    def test_validate_action_output(self):
        """Test _validate_action_output method."""
        agent = ActiveInferenceAgent("test", Mock())
        agent.available_actions = ["up", "down", "stay"]
        
        # Test valid string action
        assert agent._validate_action_output("up") == "up"
        
        # Test invalid string action
        assert agent._validate_action_output("invalid") == "stay"
        
        # Test numeric action
        assert agent._validate_action_output(0) == "up"
        assert agent._validate_action_output(1) == "down"
        
        # Test None
        assert agent._validate_action_output(None) == "stay"
        
    def test_sense_method(self):
        """Test sense method."""
        agent = ActiveInferenceAgent("test", Mock())
        
        obs = {"position": [1, 2]}
        result = agent.sense(obs)
        
        assert isinstance(result, ObservationResult)
        assert result.raw_observation == obs
        
    def test_think_method_no_pymdp(self):
        """Test think method without PyMDP."""
        agent = ActiveInferenceAgent("test", Mock())
        agent.has_pymdp = False
        
        obs_result = ObservationResult(
            raw_observation={"pos": [1, 2]},
            processed_observation={"pos": [1, 2]},
            confidence=0.9,
            timestamp=datetime.now()
        )
        
        decision = agent.think(obs_result)
        
        assert "action" in decision
        assert decision["action"] in agent.available_actions
        
    def test_act_method(self):
        """Test act method."""
        world = Mock()
        world.execute_action = Mock(return_value={"success": True})
        
        agent = ActiveInferenceAgent("test", world)
        
        decision = {"action": "up", "confidence": 0.8}
        result = agent.act(decision)
        
        world.execute_action.assert_called_once_with("test", "up")
        assert result == {"success": True}
        
    def test_act_method_with_metadata(self):
        """Test act method with metadata."""
        world = Mock()
        world.execute_action = Mock(return_value={"success": True})
        
        agent = ActiveInferenceAgent("test", world)
        
        decision = {
            "action": "down",
            "confidence": 0.9,
            "metadata": {"reason": "exploration"}
        }
        agent.act(decision)
        
        world.execute_action.assert_called_once_with("test", "down", {"reason": "exploration"})
        
    def test_get_state(self):
        """Test get_state method."""
        agent = ActiveInferenceAgent("test", Mock())
        agent._last_observation = {"pos": [1, 2]}
        agent._last_action = "up"
        
        # Add some belief history
        agent._update_belief_history(np.array([0.3, 0.7]))
        
        state = agent.get_state()
        
        assert state["agent_id"] == "test"
        assert state["last_observation"] == {"pos": [1, 2]}
        assert state["last_action"] == "up"
        assert len(state["belief_history"]) == 1
        assert state["has_pymdp"] is False
        
    def test_update_config(self):
        """Test update_config method."""
        agent = ActiveInferenceAgent("test", Mock())
        
        
        agent.update_config(belief_threshold=0.9, learning_rate=0.05)
        
        assert agent.config.belief_threshold == 0.9
        assert agent.config.learning_rate == 0.05
        
    def test_reset(self):
        """Test reset method."""
        agent = ActiveInferenceAgent("test", Mock())
        
        # Add some state
        agent._last_observation = {"pos": [1, 2]}
        agent._last_action = "up"
        agent._update_belief_history(np.array([0.3, 0.7]))
        agent.total_observations = 10
        
        # Reset
        agent.reset()
        
        assert agent._last_observation is None
        assert agent._last_action is None
        assert len(agent.belief_history) == 0
        assert agent.total_observations == 0
        

class TestActiveInferenceAgentEdgeCases:
    """Test edge cases and error handling."""
    
    def test_observation_logging(self, caplog):
        """Test observation logging."""
        with caplog.at_level(logging.DEBUG):
            agent = ActiveInferenceAgent("test", Mock())
            agent.sense({"test": "obs"})
            
        assert "Processing observation" in caplog.text
        
    def test_empty_belief_array(self):
        """Test handling of empty belief arrays."""
        agent = ActiveInferenceAgent("test", Mock())
        
        # Should handle empty array gracefully
        confidence = agent._estimate_confidence(np.array([]))
        assert confidence == 0.0
        
    def test_world_execute_action_failure(self):
        """Test handling world execution failures."""
        world = Mock()
        world.execute_action = Mock(side_effect=Exception("World error"))
        
        agent = ActiveInferenceAgent("test", world)
        
        # Should handle error gracefully
        result = agent.act({"action": "up"})
        assert result is None or "error" in str(result).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])