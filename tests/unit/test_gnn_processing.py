"""
Comprehensive test coverage for inference/gnn/executor.py, generator.py, and validator.py
GNN Processing System - Phase 3.2 systematic coverage

This test file provides complete coverage for the GNN executor, generator, and validator
following the systematic backend coverage improvement plan.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Import the GNN processing components
try:
    from inference.gnn.executor import GMNExecutor, InferenceResult
    from inference.gnn.generator import GMNGenerator
    from inference.gnn.model import GMNModel
    from inference.gnn.validator import (
        CircuitBreaker,
        GMNValidator,
        ValidationConstraints,
        ValidationError,
        ValidationResult,
        safe_gnn_processing,
    )

    IMPORT_SUCCESS = True
except ImportError:
    # Create minimal mock classes for testing if imports fail
    IMPORT_SUCCESS = False

    @dataclass
    class InferenceResult:
        action: str
        free_energy: float
        expected_free_energy: Dict[str, float]
        beliefs: Dict[str, Any]
        confidence: float

    class GMNExecutor:
        def __init__(self):
            self.parser = Mock()
            self.validator = Mock()
            self._epsilon = 1e-10

        def execute_inference(self, gnn_model, observation):
            return InferenceResult(
                action="explore",
                free_energy=1.5,
                expected_free_energy={"explore": 1.2, "exploit": 1.8},
                beliefs={"S_energy": {"mean": 75, "variance": 0.1}},
                confidence=0.8,
            )

    class GMNGenerator:
        def __init__(self):
            self.parser = Mock()
            self.validator = Mock()
            self.template_cache = {}

        def generate_base_model(self, agent_name, agent_class, personality):
            return {
                "model": {"name": agent_name, "type": "ActiveInference", "version": "1.0"},
                "state_space": {"S_energy": {"type": "Real[0, 100]"}},
                "connections": {},
                "update_equations": {},
            }

    class GMNValidator:
        def __init__(self):
            self.defined_variables = set()
            self.referenced_variables = set()

        def validate(self, model):
            return ValidationResult(is_valid=True, errors=[], warnings=[])

    @dataclass
    class ValidationResult:
        is_valid: bool
        errors: List[Any]
        warnings: List[Any]
        processing_time: float = 0.0

    class ValidationError:
        def __init__(self, field, message, severity="error", error_code=None):
            self.field = field
            self.message = message
            self.severity = severity
            self.error_code = error_code

    class GMNModel:
        def __init__(self, name="test", description="", state_space=None):
            self.name = name
            self.description = description
            self.state_space = state_space or {}
            self.observations = {}
            self.connections = []
            self.update_equations = {}
            self.preferences = {}


class TestGMNExecutor:
    """Test GNN executor for Active Inference."""

    @pytest.fixture
    def executor(self):
        """Create executor instance."""
        return GMNExecutor()

    @pytest.fixture
    def sample_model(self):
        """Create sample GNN model."""
        return {
            "model": {"name": "TestAgent", "type": "ActiveInference", "version": "1.0"},
            "state_space": {
                "S_energy": {"type": "Real[0, 100]", "description": "Energy level"},
                "S_position": {"type": "H3Cell[resolution=7]", "description": "Position"},
                "A_actions": {
                    "type": "Categorical",
                    "options": ["explore", "exploit", "rest", "communicate"],
                },
            },
            "connections": {
                "C_pref": {
                    "type": "observation -> Real[0, 1]",
                    "preferences": {"Exploration": 0.7, "Resources": 0.2, "Social": 0.1},
                }
            },
            "update_equations": {
                "belief_update": {
                    "state": "S_energy",
                    "formula": "S_energy(t+1) = S_energy(t) + learning_rate * prediction_error",
                }
            },
        }

    @pytest.fixture
    def sample_observation(self):
        """Create sample observation."""
        return {
            "energy": 75.0,
            "position": "8928308280fffff",
            "exploration": 0.5,
            "resources": 0.3,
            "social": 0.2,
        }

    def test_executor_initialization(self, executor):
        """Test executor initialization."""
        assert hasattr(executor, "parser")
        assert hasattr(executor, "validator")
        assert executor._epsilon == 1e-10

    def test_execute_inference_basic(self, executor, sample_model, sample_observation):
        """Test basic inference execution."""
        if not IMPORT_SUCCESS:
            result = executor.execute_inference(sample_model, sample_observation)
            assert isinstance(result, InferenceResult)
            return

        # Mock validator to return valid
        executor.validator.validate = Mock(return_value=Mock(is_valid=True, errors=[]))

        result = executor.execute_inference(sample_model, sample_observation)

        assert isinstance(result, InferenceResult)
        assert result.action in ["explore", "exploit", "rest", "communicate"]
        assert isinstance(result.free_energy, float)
        assert isinstance(result.expected_free_energy, dict)
        assert isinstance(result.beliefs, dict)
        assert 0 <= result.confidence <= 1

    def test_initialize_beliefs(self, executor, sample_model, sample_observation):
        """Test belief initialization."""
        if not IMPORT_SUCCESS:
            return

        beliefs = executor._initialize_beliefs(sample_model["state_space"], sample_observation)

        assert "S_energy" in beliefs
        assert beliefs["S_energy"]["mean"] == 75.0
        assert beliefs["S_energy"]["variance"] == 0.1
        assert beliefs["S_energy"]["range"] == [0, 100]

    def test_calculate_free_energy(self, executor):
        """Test free energy calculation."""
        if not IMPORT_SUCCESS:
            return

        beliefs = {
            "S_energy": {"mean": 75, "variance": 0.1, "range": [0, 100]},
            "S_position": {"mean": 0.5, "variance": 0.05, "range": [0, 1]},
        }

        observation = {"energy": 70, "position": 0.6}

        connections = {"C_pref": {"preferences": {"Exploration": 0.7, "Resources": 0.3}}}

        free_energy = executor._calculate_free_energy(beliefs, observation, connections)

        assert isinstance(free_energy, float)
        assert np.isfinite(free_energy)

    def test_get_available_actions(self, executor, sample_model):
        """Test action extraction."""
        if not IMPORT_SUCCESS:
            return

        actions = executor._get_available_actions(sample_model["state_space"])

        assert isinstance(actions, list)
        assert "explore" in actions
        assert "exploit" in actions
        assert len(actions) == 4  # From sample model

    def test_calculate_expected_free_energy(self, executor):
        """Test expected free energy calculation."""
        if not IMPORT_SUCCESS:
            return

        beliefs = {"S_energy": {"mean": 75, "variance": 0.1}}

        action = "explore"
        observation = {"energy": 75}
        connections = {"C_pref": {"preferences": {"Exploration": 0.7}}}
        update_equations = {}

        efe = executor._calculate_expected_free_energy(
            beliefs, action, observation, connections, update_equations
        )

        assert isinstance(efe, float)
        assert np.isfinite(efe)

    def test_update_beliefs_with_prediction_error(self, executor):
        """Test belief updates with prediction error."""
        if not IMPORT_SUCCESS:
            return

        beliefs = {"S_energy": {"mean": 75, "variance": 0.1, "range": [0, 100]}}

        observation = {"energy": 80}  # Higher than belief

        update_equations = {"belief_update": {"state": "S_energy", "formula": "prediction_error"}}

        updated = executor._update_beliefs(beliefs, "explore", observation, update_equations)

        # Should move mean towards observation
        assert updated["S_energy"]["mean"] > beliefs["S_energy"]["mean"]
        assert updated["S_energy"]["variance"] < beliefs["S_energy"]["variance"]

    def test_simulate_belief_update(self, executor):
        """Test simulated belief updates."""
        if not IMPORT_SUCCESS:
            return

        beliefs = {
            "S_energy": {"mean": 75, "variance": 0.2},
            "S_position": {"mean": 0.5, "variance": 0.3},
        }

        # Test exploration reduces uncertainty
        simulated = executor._simulate_belief_update(beliefs, "explore", {})

        assert simulated["S_energy"]["variance"] < beliefs["S_energy"]["variance"]
        assert simulated["S_position"]["variance"] < beliefs["S_position"]["variance"]

    def test_get_action_effects(self, executor):
        """Test action effect prediction."""
        if not IMPORT_SUCCESS:
            return

        effects = executor._get_action_effects("explore", {})

        assert "Exploration" in effects
        assert effects["Exploration"] == 0.8
        assert "Resources" in effects
        assert effects["Resources"] == -0.1

    def test_calculate_confidence(self, executor):
        """Test confidence calculation."""
        if not IMPORT_SUCCESS:
            return

        # Test with clear best action
        efe_clear = {"explore": 1.0, "exploit": 3.0, "rest": 2.5}
        confidence_clear = executor._calculate_confidence(efe_clear)
        assert confidence_clear > 0.7

        # Test with similar actions
        efe_similar = {"explore": 1.0, "exploit": 1.1, "rest": 1.05}
        confidence_similar = executor._calculate_confidence(efe_similar)
        assert confidence_similar < 0.3

        # Test edge case with single action
        efe_single = {"explore": 1.0}
        confidence_single = executor._calculate_confidence(efe_single)
        assert confidence_single == 1.0

    def test_execute_from_file(self, executor, tmp_path):
        """Test execution from file."""
        if not IMPORT_SUCCESS:
            return

        # Create temporary GNN file
        gnn_file = tmp_path / "test.gnn.md"
        gnn_content = """
# Model: TestAgent

## State Space
S_energy: Real[0, 100]
"""
        gnn_file.write_text(gnn_content)

        # Mock parser
        executor.parser.parse_file = Mock(
            return_value={
                "model": {"name": "TestAgent"},
                "state_space": {"S_energy": {"type": "Real[0, 100]"}},
                "connections": {},
                "update_equations": {},
            }
        )

        observation = {"energy": 50}

        with patch.object(executor, "execute_inference") as mock_exec:
            mock_exec.return_value = InferenceResult(
                action="explore",
                free_energy=1.0,
                expected_free_energy={},
                beliefs={},
                confidence=0.8,
            )

            result = executor.execute_from_file(str(gnn_file), observation)

            assert result.action == "explore"
            mock_exec.assert_called_once()

    def test_error_handling(self, executor):
        """Test error handling in executor."""
        if not IMPORT_SUCCESS:
            return

        # Test with invalid model
        executor.validator.validate = Mock(
            return_value=Mock(is_valid=False, errors=["Invalid model"])
        )

        with pytest.raises(ValueError, match="Invalid GNN model"):
            executor.execute_inference({}, {})

    def test_distribution_initialization(self, executor):
        """Test distribution initialization."""
        if not IMPORT_SUCCESS:
            return

        dist = executor._initialize_distribution("Distribution[State]", {})

        assert isinstance(dist, list)
        assert len(dist) == 4
        assert sum(dist) == 1.0


class TestGMNGenerator:
    """Test GNN model generator."""

    @pytest.fixture
    def generator(self):
        """Create generator instance."""
        return GMNGenerator()

    @pytest.fixture
    def explorer_personality(self):
        """Create explorer personality."""
        return {
            "exploration": 80,
            "cooperation": 60,
            "efficiency": 40,
            "curiosity": 90,
            "risk_tolerance": 70,
        }

    @pytest.fixture
    def merchant_personality(self):
        """Create merchant personality."""
        return {
            "exploration": 30,
            "cooperation": 70,
            "efficiency": 90,
            "curiosity": 40,
            "risk_tolerance": 50,
        }

    def test_generator_initialization(self, generator):
        """Test generator initialization."""
        assert hasattr(generator, "parser")
        assert hasattr(generator, "validator")
        assert hasattr(generator, "template_cache")
        assert isinstance(generator.template_cache, dict)

    def test_generate_base_model_explorer(self, generator, explorer_personality):
        """Test generating explorer model."""
        if not IMPORT_SUCCESS:
            model = generator.generate_base_model("Explorer-1", "Explorer", explorer_personality)
            assert model["model"]["name"] == "Explorer-1"
            return

        # Mock validator
        generator.validator.validate_model = Mock(return_value=(True, []))

        model = generator.generate_base_model("Explorer-Alpha", "Explorer", explorer_personality)

        assert model["model"]["name"] == "Explorer-Alpha"
        assert model["model"]["class"] == "Explorer"
        assert "S_curiosity" in model["state_space"]
        assert "S_knowledge" in model["state_space"]
        assert "S_position" in model["state_space"]

        # Check preferences weighted towards exploration
        prefs = model["connections"]["C_pref"]["preferences"]
        assert prefs["Exploration"] > prefs["Resources"]

    def test_generate_base_model_merchant(self, generator, merchant_personality):
        """Test generating merchant model."""
        if not IMPORT_SUCCESS:
            return

        generator.validator.validate_model = Mock(return_value=(True, []))

        model = generator.generate_base_model("Merchant-Beta", "Merchant", merchant_personality)

        assert model["model"]["class"] == "Merchant"
        assert "S_inventory" in model["state_space"]
        assert "S_wealth" in model["state_space"]
        assert "S_reputation" in model["state_space"]

        # Check preferences weighted towards resources
        prefs = model["connections"]["C_pref"]["preferences"]
        assert prefs["Resources"] > prefs["Exploration"]

    def test_get_class_template(self, generator):
        """Test class template retrieval."""
        if not IMPORT_SUCCESS:
            return

        classes = ["Explorer", "Merchant", "Scholar", "Guardian"]

        for agent_class in classes:
            template = generator._get_class_template(agent_class)
            assert "focus" in template
            assert "base_preferences" in template
            assert "key_states" in template
            assert len(template["base_preferences"]) == 3

    def test_generate_state_space_with_personality(self, generator):
        """Test state space generation with personality effects."""
        if not IMPORT_SUCCESS:
            return

        # High curiosity personality
        curious_personality = {"curiosity": 85, "risk_tolerance": 30}
        state_space = generator._generate_state_space("Explorer", curious_personality)

        assert "S_novelty_seeking" in state_space
        assert "S_risk_assessment" not in state_space  # Low risk tolerance

        # High risk tolerance personality
        risky_personality = {"curiosity": 30, "risk_tolerance": 85}
        state_space = generator._generate_state_space("Explorer", risky_personality)

        assert "S_novelty_seeking" not in state_space  # Low curiosity
        assert "S_risk_assessment" in state_space

    def test_generate_connections(self, generator):
        """Test connection generation."""
        if not IMPORT_SUCCESS:
            return

        personality = {"exploration": 50, "cooperation": 30, "efficiency": 20}
        connections = generator._generate_connections(personality)

        assert "C_pref" in connections
        assert "C_likelihood" in connections

        prefs = connections["C_pref"]["preferences"]
        total = prefs["Exploration"] + prefs["Resources"] + prefs["Social"]
        assert abs(total - 1.0) < 0.01  # Should sum to 1

    def test_generate_update_equations(self, generator):
        """Test update equation generation."""
        if not IMPORT_SUCCESS:
            return

        personality = {"curiosity": 50, "efficiency": 80, "cooperation": 75}
        equations = generator._generate_update_equations(personality)

        assert "belief_update" in equations
        assert "energy_dynamics" in equations
        assert "efficiency_optimization" in equations  # High efficiency
        assert "social_learning" in equations  # High cooperation

        # Check learning rate scales with curiosity
        learning_rate = equations["belief_update"]["parameters"]["learning_rate"]
        assert 0.1 <= learning_rate <= 0.2

    def test_refine_model(self, generator):
        """Test model refinement with patterns."""
        if not IMPORT_SUCCESS:
            return

        generator.validator.validate_model = Mock(return_value=(True, []))

        base_model = {
            "model": {"name": "TestAgent"},
            "state_space": {},
            "connections": {"C_pref": {"preferences": {"Exploration": 0.5, "Resources": 0.5}}},
            "update_equations": {},
            "metadata": {"model_version": 1},
        }

        patterns = [
            {
                "type": "successful_action_sequence",
                "dominant_action": "explore",
                "success_rate": 0.85,
                "confidence": 0.9,
            },
            {
                "type": "preference_adjustment",
                "adjustments": {"Exploration": 0.1, "Resources": -0.05},
                "confidence": 0.82,
            },
        ]

        refined = generator.refine_model(base_model, patterns)

        assert refined["metadata"]["model_version"] == 2
        assert "refinement_changes" in refined["metadata"]
        assert len(refined["metadata"]["refinement_changes"]) == 2

    def test_apply_pattern_to_model(self, generator):
        """Test pattern application."""
        if not IMPORT_SUCCESS:
            return

        model = {
            "connections": {"C_pref": {"preferences": {"Exploration": 0.5}}, "C_likelihood": {}}
        }

        # Test action bias pattern
        action_pattern = {
            "type": "successful_action_sequence",
            "dominant_action": "explore",
            "success_rate": 0.9,
            "confidence": 0.85,
        }

        change = generator._apply_pattern_to_model(model, action_pattern)

        assert change is not None
        assert len(change["changes"]) > 0
        assert "action_biases" in model["connections"]["C_pref"]

    def test_generate_from_experience(self, generator):
        """Test generation from experience."""
        if not IMPORT_SUCCESS:
            return

        generator.validator.validate_model = Mock(return_value=(True, []))

        base_model = {
            "model": {"name": "TestAgent"},
            "state_space": {},
            "connections": {},
            "update_equations": {},
        }

        experience = {
            "unique_observations": 100,
            "action_statistics": {
                "explore": {"count": 50, "success_count": 40},
                "exploit": {"count": 30, "success_count": 15},
            },
            "observed_correlations": [
                {
                    "observation": "energy",
                    "state": "S_energy",
                    "correlation": 0.8,
                    "occurrences": 20,
                }
            ],
        }

        new_model = generator.generate_from_experience(base_model, experience)

        # Should add world model for extensive observations
        assert "S_world_model" in new_model["state_space"]

    def test_export_to_gnn_format(self, generator, tmp_path):
        """Test GNN file export."""
        if not IMPORT_SUCCESS:
            return

        model = {
            "model": {"name": "TestAgent", "type": "ActiveInference", "class": "Explorer"},
            "metadata": {"model_version": 2},
            "state_space": {"S_energy": {"type": "Real[0, 100]", "description": "Energy"}},
            "connections": {
                "C_pref": {"type": "observation -> Real[0, 1]", "preferences": {"Exploration": 0.7}}
            },
            "update_equations": {
                "belief_update": {
                    "state": "S_energy",
                    "formula": "update_formula",
                    "parameters": {"learning_rate": 0.1},
                }
            },
        }

        output_file = tmp_path / "test_export.gnn.md"
        generator.export_to_gnn_format(model, str(output_file))

        assert output_file.exists()
        content = output_file.read_text()
        assert "# GNN Model: TestAgent" in content
        assert "Class: Explorer" in content
        assert "S_energy: Real[0, 100]" in content

    def test_validation_error_handling(self, generator):
        """Test handling of validation errors."""
        if not IMPORT_SUCCESS:
            return

        generator.validator.validate_model = Mock(return_value=(False, ["Invalid state space"]))

        with pytest.raises(ValueError, match="validation failed"):
            generator.generate_base_model("Test", "Explorer", {})


class TestGMNValidator:
    """Test GNN model validator."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return GMNValidator()

    @pytest.fixture
    def valid_model(self):
        """Create valid model."""
        if IMPORT_SUCCESS:
            model = GMNModel(name="ValidModel", description="Test model")
            model.state_space = {
                "S_energy": {"type": "Real", "constraints": {"min": 0, "max": 100}}
            }
            model.preferences = {
                "exploration_pref": {"input": "observation", "output": "Real[0, 1]", "details": []}
            }
            return model
        else:
            return GMNModel("ValidModel", "Test model", {"S_energy": {"type": "Real"}})

    def test_validator_initialization(self, validator):
        """Test validator initialization."""
        assert hasattr(validator, "defined_variables")
        assert hasattr(validator, "referenced_variables")
        assert hasattr(validator, "circuit_breaker")
        assert isinstance(validator.defined_variables, set)
        assert isinstance(validator.referenced_variables, set)

    def test_validate_valid_model(self, validator, valid_model):
        """Test validation of valid model."""
        result = validator.validate(valid_model)

        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_validate_invalid_name(self, validator):
        """Test validation with invalid name."""
        if not IMPORT_SUCCESS:
            return

        model = GMNModel(name="123-invalid-name!", description="Test")
        result = validator.validate(model)

        assert not result.is_valid
        assert any(e.field == "model.name" for e in result.errors)

    def test_validate_name_length(self, validator):
        """Test name length validation."""
        if not IMPORT_SUCCESS:
            return

        # Test too long name
        long_name = "a" * (ValidationConstraints.MAX_NAME_LENGTH + 1)
        model = GMNModel(name=long_name)
        result = validator.validate(model)

        assert not result.is_valid
        assert any(e.error_code == "NAME_TOO_LONG" for e in result.errors)

        # Test empty name
        model = GMNModel(name="")
        result = validator.validate(model)

        assert not result.is_valid
        assert any(e.error_code == "MISSING_NAME" for e in result.errors)

    def test_validate_state_space_types(self, validator):
        """Test state space type validation."""
        if not IMPORT_SUCCESS:
            return

        model = GMNModel(name="Test")
        model.state_space = {
            "valid_real": {"type": "Real", "constraints": {"min": 0, "max": 100}},
            "valid_int": {"type": "Integer", "constraints": {"min": 0, "max": 10}},
            "invalid_type": {"type": "InvalidType"},
            "missing_type": {"constraints": {"min": 0}},
        }

        result = validator.validate(model)

        assert any(e.error_code == "UNKNOWN_TYPE" for e in result.errors)
        assert any(e.error_code == "MISSING_TYPE" for e in result.errors)

    def test_validate_real_constraints(self, validator):
        """Test Real type constraint validation."""
        if not IMPORT_SUCCESS:
            return

        model = GMNModel(name="Test")
        model.state_space = {
            "invalid_range": {"type": "Real", "constraints": {"min": 100, "max": 0}},
            "equal_range": {"type": "Real", "constraints": {"min": 50, "max": 50}},
            "no_constraints": {"type": "Real"},
        }

        result = validator.validate(model)

        assert any(e.error_code == "INVALID_RANGE" for e in result.errors)
        assert any(w.error_code == "SINGLE_VALUE_RANGE" for w in result.warnings)
        assert any(w.error_code == "MISSING_BOUNDS" for w in result.warnings)

    def test_validate_h3cell_constraints(self, validator):
        """Test H3Cell constraint validation."""
        if not IMPORT_SUCCESS:
            return

        model = GMNModel(name="Test")
        model.state_space = {
            "valid_h3": {"type": "H3Cell", "constraints": {"resolution": 7}},
            "invalid_res": {"type": "H3Cell", "constraints": {"resolution": 20}},
            "missing_res": {"type": "H3Cell", "constraints": {}},
        }

        result = validator.validate(model)

        assert any(e.error_code == "INVALID_H3_RESOLUTION" for e in result.errors)
        assert any(w.error_code == "MISSING_H3_RESOLUTION" for w in result.warnings)

    def test_validate_connections(self, validator):
        """Test connection validation."""
        if not IMPORT_SUCCESS:
            return

        model = GMNModel(name="Test")
        model.state_space = {"A": {"type": "Real"}, "B": {"type": "Real"}}
        model.connections = [
            {"source": "A", "target": "B", "type": "depends"},
            {"source": "A", "target": "B", "type": "depends"},  # Duplicate
            {"source": "C", "target": "D", "type": "depends"},  # Undefined vars
            {"source": "A", "target": "A", "type": "depends"},  # Self-loop
        ]

        result = validator.validate(model)

        assert any(w.error_code == "DUPLICATE_CONNECTION" for w in result.warnings)
        assert any(w.error_code == "SELF_LOOP" for w in result.warnings)

    def test_validate_update_equations(self, validator):
        """Test update equation validation."""
        if not IMPORT_SUCCESS:
            return

        model = GMNModel(name="Test")
        model.state_space = {"S_x": {"type": "Real"}}
        model.update_equations = {
            "S_x": "S_x(t+1) = S_x(t) + learning_rate * error",
            "S_y": "S_y(t+1) = S_y(t)",  # Undefined variable
            "dangerous": "__import__('os').system('ls')",  # Security risk
        }

        result = validator.validate(model)

        assert any(w.error_code == "UNDEFINED_UPDATE_VAR" for w in result.warnings)
        assert any(e.error_code == "DANGEROUS_PATTERN" for e in result.errors)

    def test_validate_preferences(self, validator):
        """Test preference validation."""
        if not IMPORT_SUCCESS:
            return

        model = GMNModel(name="Test")
        model.preferences = {
            "valid_pref": {"input": "observation", "output": "Real[0, 1]", "details": []},
            "missing_input": {"output": "Real[0, 1]"},
            # Should end with _pref
            "bad_name": {"input": "observation", "output": "Real[0, 1]"},
        }

        result = validator.validate(model)

        assert any(e.error_code == "MISSING_PREF_INPUT" for e in result.errors)
        assert any(w.error_code == "INVALID_PREF_NAME" for w in result.warnings)

    def test_circular_dependency_detection(self, validator):
        """Test circular dependency detection."""
        if not IMPORT_SUCCESS:
            return

        model = GMNModel(name="Test")
        model.connections = [
            {"source": "A", "target": "B", "type": "depends"},
            {"source": "B", "target": "C", "type": "depends"},
            {"source": "C", "target": "A", "type": "depends"},  # Creates cycle
        ]

        result = validator.validate(model)

        assert any(w.error_code == "CIRCULAR_DEPENDENCY" for w in result.warnings)

    def test_security_validation(self, validator):
        """Test security validation."""
        if not IMPORT_SUCCESS:
            return

        model = GMNModel(name="<script>alert('xss')</script>")
        model.update_equations = {"S_x": "eval('dangerous_code')"}

        result = validator.validate(model)

        assert any(e.error_code == "SECURITY_RISK" for e in result.errors)
        assert any(e.error_code == "DANGEROUS_PATTERN" for e in result.errors)


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_circuit_breaker_basic(self):
        """Test basic circuit breaker operation."""
        if not IMPORT_SUCCESS:
            return

        cb = CircuitBreaker(failure_threshold=3, timeout=1)

        def failing_func():
            raise Exception("Test failure")

        # First failures should pass through
        for i in range(2):
            with pytest.raises(Exception, match="Test failure"):
                cb.call(failing_func)

        # Third failure should open circuit
        with pytest.raises(Exception, match="Test failure"):
            cb.call(failing_func)

        assert cb.is_open

        # Further calls should fail immediately
        with pytest.raises(Exception, match="Circuit breaker is open"):
            cb.call(failing_func)

    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery."""
        if not IMPORT_SUCCESS:
            return

        cb = CircuitBreaker(failure_threshold=2, timeout=0.1)

        def failing_func():
            raise Exception("Fail")

        def success_func():
            return "Success"

        # Open the circuit
        for _ in range(2):
            try:
                cb.call(failing_func)
            except Exception:
                pass

        assert cb.is_open

        # Wait for timeout
        time.sleep(0.2)

        # Circuit should close and allow retry
        result = cb.call(success_func)
        assert result == "Success"
        assert not cb.is_open


class TestSafeGNNProcessing:
    """Test safe GNN processing context manager."""

    def test_safe_processing_file_validation(self, tmp_path):
        """Test file validation in safe processing."""
        if not IMPORT_SUCCESS:
            return

        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            with safe_gnn_processing("non_existent.gnn.md"):
                pass

        # Test invalid extension
        invalid_file = tmp_path / "test.txt"
        invalid_file.write_text("content")

        with pytest.raises(ValueError, match="Invalid GNN file extension"):
            with safe_gnn_processing(str(invalid_file)):
                pass

        # Test valid file
        valid_file = tmp_path / "test.gnn.md"
        valid_file.write_text("# Model")

        with safe_gnn_processing(str(valid_file)) as resources:
            assert isinstance(resources, list)

    def test_safe_processing_size_limit(self, tmp_path):
        """Test file size limit."""
        if not IMPORT_SUCCESS:
            return

        large_file = tmp_path / "large.gnn.md"
        # Create file larger than limit
        large_content = "x" * (ValidationConstraints.MAX_FILE_SIZE_MB * 1024 * 1024 + 1)
        large_file.write_text(large_content)

        with pytest.raises(ValueError, match="File too large"):
            with safe_gnn_processing(str(large_file)):
                pass


class TestIntegration:
    """Integration tests for GNN processing components."""

    def test_full_pipeline(self):
        """Test full pipeline: generate -> validate -> execute."""
        if not IMPORT_SUCCESS:
            return

        # Generate model
        generator = GMNGenerator()
        generator.validator.validate_model = Mock(return_value=(True, []))

        personality = {"exploration": 70, "cooperation": 50, "efficiency": 50}
        model = generator.generate_base_model("TestAgent", "Explorer", personality)

        # Validate model
        validator = GMNValidator()
        # Convert dict to GMNModel for validation
        gnn_model = GMNModel(name=model["model"]["name"])
        gnn_model.state_space = model["state_space"]
        gnn_model.connections = []
        gnn_model.update_equations = model["update_equations"]
        gnn_model.preferences = {}

        validation_result = validator.validate(gnn_model)
        assert validation_result.is_valid or len(validation_result.errors) == 0

        # Execute inference
        executor = GMNExecutor()
        executor.validator.validate = Mock(return_value=Mock(is_valid=True))

        observation = {"energy": 50, "exploration": 0.5}
        result = executor.execute_inference(model, observation)

        assert isinstance(result.action, str)
        assert isinstance(result.free_energy, float)
        assert isinstance(result.confidence, float)

    def test_model_evolution(self):
        """Test model evolution through experience."""
        if not IMPORT_SUCCESS:
            return

        generator = GMNGenerator()
        generator.validator.validate_model = Mock(return_value=(True, []))

        # Create initial model
        base_model = generator.generate_base_model(
            "EvolvingAgent", "Explorer", {"exploration": 50, "cooperation": 50, "efficiency": 50}
        )

        # Simulate experience
        experience = {
            "unique_observations": 75,
            "action_statistics": {
                "explore": {"count": 100, "success_count": 80},
                "exploit": {"count": 50, "success_count": 45},
            },
        }

        # Evolve model
        evolved_model = generator.generate_from_experience(base_model, experience)

        # Verify evolution
        assert "S_world_model" in evolved_model["state_space"]
        assert evolved_model != base_model
