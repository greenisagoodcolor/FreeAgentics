"""
Comprehensive tests for inference.gnn.generator module.

Tests GNN model generation functionality including base model creation,
refinement, pattern application, and export capabilities.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from inference.gnn.generator import GMNGenerator


@pytest.fixture
def gmn_generator():
    """Create a GMNGenerator instance for testing."""
    return GMNGenerator()


@pytest.fixture
def sample_personality():
    """Create a sample personality configuration."""
    return {
        "exploration": 80,
        "cooperation": 60,
        "efficiency": 40,
        "curiosity": 90,
        "risk_tolerance": 70,
    }


@pytest.fixture
def sample_base_model():
    """Create a sample base model for testing."""
    return {
        "model": {
            "name": "TestAgent",
            "type": "ActiveInference",
            "version": "1.0",
            "class": "Explorer",
            "created": "2024-01-01T00:00:00",
        },
        "state_space": {
            "S_energy": {"type": "Real[0, 100]", "description": "Agent energy level"},
            "S_position": {"type": "H3Cell[resolution=7]", "description": "Current hex position"},
        },
        "connections": {
            "C_pref": {
                "type": "observation -> Real[0, 1]",
                "description": "Preference function mapping observations to utilities",
                "preferences": {"Exploration": 0.7, "Resources": 0.2, "Social": 0.1},
            }
        },
        "update_equations": {
            "belief_update": {
                "state": "S_beliefs",
                "formula": "S_beliefs(t+1) = S_beliefs(t) + learning_rate * prediction_error",
                "parameters": {"learning_rate": 0.15},
            }
        },
        "metadata": {"personality": {}, "learning_history": [], "model_version": 1},
    }


@pytest.fixture
def sample_patterns():
    """Create sample patterns for testing."""
    return [
        {
            "type": "successful_action_sequence",
            "dominant_action": "explore",
            "success_rate": 0.85,
            "confidence": 0.9,
        },
        {
            "type": "environmental_correlation",
            "correlation": {
                "observation": "high_energy",
                "state": "successful_exploration",
                "strength": 0.75,
            },
            "confidence": 0.82,
        },
        {
            "type": "preference_adjustment",
            "adjustments": {"Exploration": 0.1, "Resources": -0.05},
            "confidence": 0.85,
        },
    ]


@pytest.fixture
def sample_experience_summary():
    """Create sample experience summary for testing."""
    return {
        "action_statistics": {
            "explore": {"count": 50, "success_count": 42},
            "exploit": {"count": 30, "success_count": 18},
            "communicate": {"count": 15, "success_count": 12},
        },
        "observed_correlations": [
            {
                "observation": "resource_nearby",
                "state": "successful_exploit",
                "correlation": 0.8,
                "occurrences": 10,
            }
        ],
        "unique_observations": 75,
    }


class TestGMNGeneratorInitialization:
    """Test GMNGenerator initialization."""

    def test_initialization(self, gmn_generator):
        """Test that GMNGenerator initializes correctly."""
        assert gmn_generator.parser is not None
        assert gmn_generator.validator is not None
        assert gmn_generator.template_cache == {}

    @patch("inference.gnn.generator.GMNParser")
    @patch("inference.gnn.generator.GMNValidator")
    def test_initialization_with_mocked_dependencies(
            self, mock_validator, mock_parser):
        """Test initialization with mocked dependencies."""
        generator = GMNGenerator()

        mock_parser.assert_called_once()
        mock_validator.assert_called_once()
        assert generator.parser is not None
        assert generator.validator is not None


class TestGenerateBaseModel:
    """Test base model generation functionality."""

    @patch.object(GMNGenerator, "_generate_state_space")
    @patch.object(GMNGenerator, "_generate_connections")
    @patch.object(GMNGenerator, "_generate_update_equations")
    def test_generate_base_model_structure(
            self,
            mock_equations,
            mock_connections,
            mock_state_space,
            gmn_generator,
            sample_personality):
        """Test that generate_base_model creates correct structure."""
        # Setup mocks
        mock_state_space.return_value = {"S_energy": {"type": "Real[0, 100]"}}
        mock_connections.return_value = {
            "C_pref": {"type": "observation -> Real[0, 1]"}}
        mock_equations.return_value = {"belief_update": {"state": "S_beliefs"}}

        # Mock validator to return valid
        gmn_generator.validator.validate_model = Mock(return_value=(True, []))

        result = gmn_generator.generate_base_model(
            "TestAgent", "Explorer", sample_personality)

        # Check structure
        assert "model" in result
        assert "state_space" in result
        assert "connections" in result
        assert "update_equations" in result
        assert "metadata" in result

        # Check model details
        assert result["model"]["name"] == "TestAgent"
        assert result["model"]["type"] == "ActiveInference"
        assert result["model"]["class"] == "Explorer"
        assert "created" in result["model"]

        # Check metadata
        assert result["metadata"]["personality"] == sample_personality
        assert result["metadata"]["learning_history"] == []
        assert result["metadata"]["model_version"] == 1

    def test_generate_base_model_all_agent_classes(
            self, gmn_generator, sample_personality):
        """Test base model generation for all agent classes."""
        agent_classes = ["Explorer", "Merchant", "Scholar", "Guardian"]

        # Mock validator to always return valid
        gmn_generator.validator.validate_model = Mock(return_value=(True, []))

        for agent_class in agent_classes:
            result = gmn_generator.generate_base_model(
                f"Test{agent_class}", agent_class, sample_personality
            )

            assert result["model"]["class"] == agent_class
            assert result["model"]["name"] == f"Test{agent_class}"

    def test_generate_base_model_validation_failure(
            self, gmn_generator, sample_personality):
        """Test handling of validation failure during base model generation."""
        # Mock validator to return invalid
        gmn_generator.validator.validate_model = Mock(
            return_value=(False, ["Test error"]))

        with pytest.raises(ValueError, match="Generated model validation failed"):
            gmn_generator.generate_base_model(
                "TestAgent", "Explorer", sample_personality)

    def test_generate_base_model_datetime_creation(
            self, gmn_generator, sample_personality):
        """Test that created timestamp is properly set."""
        gmn_generator.validator.validate_model = Mock(return_value=(True, []))

        with patch("inference.gnn.generator.datetime") as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"

            result = gmn_generator.generate_base_model(
                "TestAgent", "Explorer", sample_personality)

            assert result["model"]["created"] == "2024-01-01T12:00:00"


class TestClassTemplates:
    """Test agent class template functionality."""

    def test_get_class_template_explorer(self, gmn_generator):
        """Test Explorer class template."""
        template = gmn_generator._get_class_template("Explorer")

        assert template["focus"] == "discovery"
        assert template["base_preferences"]["Exploration"] == 0.7
        assert "S_curiosity" in template["key_states"]
        assert "S_knowledge" in template["key_states"]
        assert "S_position" in template["key_states"]

    def test_get_class_template_merchant(self, gmn_generator):
        """Test Merchant class template."""
        template = gmn_generator._get_class_template("Merchant")

        assert template["focus"] == "trading"
        assert template["base_preferences"]["Resources"] == 0.6
        assert "S_inventory" in template["key_states"]
        assert "S_reputation" in template["key_states"]
        assert "S_wealth" in template["key_states"]

    def test_get_class_template_scholar(self, gmn_generator):
        """Test Scholar class template."""
        template = gmn_generator._get_class_template("Scholar")

        assert template["focus"] == "learning"
        assert template["base_preferences"]["Social"] == 0.6
        assert "S_knowledge" in template["key_states"]
        assert "S_theories" in template["key_states"]
        assert "S_connections" in template["key_states"]

    def test_get_class_template_guardian(self, gmn_generator):
        """Test Guardian class template."""
        template = gmn_generator._get_class_template("Guardian")

        assert template["focus"] == "protection"
        assert template["base_preferences"]["Social"] == 0.5
        assert "S_territory" in template["key_states"]
        assert "S_alertness" in template["key_states"]
        assert "S_allies" in template["key_states"]

    def test_get_class_template_unknown_defaults_to_explorer(
            self, gmn_generator):
        """Test that unknown class defaults to Explorer template."""
        template = gmn_generator._get_class_template("UnknownClass")
        explorer_template = gmn_generator._get_class_template("Explorer")

        assert template == explorer_template


class TestStateSpaceGeneration:
    """Test state space generation functionality."""

    def test_generate_state_space_basic_states(self, gmn_generator):
        """Test that basic states are always included."""
        state_space = gmn_generator._generate_state_space("Explorer", {})

        # Check basic states
        assert "S_energy" in state_space
        assert "S_position" in state_space
        assert "S_beliefs" in state_space
        assert "A_actions" in state_space

        # Check action options
        assert "move_north" in state_space["A_actions"]["options"]
        assert "explore" in state_space["A_actions"]["options"]
        assert "communicate" in state_space["A_actions"]["options"]

    def test_generate_state_space_agent_specific_states(self, gmn_generator):
        """Test agent-specific state generation."""
        # Test Explorer states
        explorer_states = gmn_generator._generate_state_space("Explorer", {})
        assert "S_curiosity" in explorer_states
        assert "S_knowledge" in explorer_states

        # Test Merchant states
        merchant_states = gmn_generator._generate_state_space("Merchant", {})
        assert "S_inventory" in merchant_states
        assert "S_wealth" in merchant_states

        # Test Guardian states
        guardian_states = gmn_generator._generate_state_space("Guardian", {})
        assert "S_territory" in guardian_states
        assert "S_alertness" in guardian_states

    def test_generate_state_space_personality_driven_states(
            self, gmn_generator):
        """Test personality-driven state generation."""
        high_curiosity = {"curiosity": 0.8}
        high_risk = {"risk_tolerance": 0.9}

        curious_states = gmn_generator._generate_state_space(
            "Explorer", high_curiosity)
        assert "S_novelty_seeking" in curious_states

        risky_states = gmn_generator._generate_state_space(
            "Explorer", high_risk)
        assert "S_risk_assessment" in risky_states

    def test_generate_state_space_low_personality_traits(self, gmn_generator):
        """Test that low personality traits don't add extra states."""
        low_traits = {"curiosity": 0.3, "risk_tolerance": 0.2}

        states = gmn_generator._generate_state_space("Explorer", low_traits)
        assert "S_novelty_seeking" not in states
        assert "S_risk_assessment" not in states


class TestConnectionGeneration:
    """Test connection generation functionality."""

    def test_generate_connections_basic_structure(self, gmn_generator):
        """Test basic connection structure."""
        personality = {
            "exploration": 0.6,
            "cooperation": 0.3,
            "efficiency": 0.1}
        connections = gmn_generator._generate_connections(personality)

        assert "C_pref" in connections
        assert "C_likelihood" in connections

        # Check preference structure
        pref = connections["C_pref"]
        assert "preferences" in pref
        assert "Exploration" in pref["preferences"]
        assert "Resources" in pref["preferences"]
        assert "Social" in pref["preferences"]

    def test_generate_connections_personality_weights(self, gmn_generator):
        """Test that personality traits affect connection weights."""
        high_exploration = {
            "exploration": 90,
            "cooperation": 10,
            "efficiency": 10}
        connections = gmn_generator._generate_connections(high_exploration)

        prefs = connections["C_pref"]["preferences"]
        assert prefs["Exploration"] > prefs["Social"]
        assert prefs["Exploration"] > prefs["Resources"]

    def test_generate_connections_weight_normalization(self, gmn_generator):
        """Test that connection weights are properly normalized."""
        personality = {"exploration": 50, "cooperation": 30, "efficiency": 20}
        connections = gmn_generator._generate_connections(personality)

        prefs = connections["C_pref"]["preferences"]
        total_weight = sum(prefs.values())
        # Should sum to approximately 1.0
        assert abs(total_weight - 1.0) < 0.01

    def test_generate_connections_zero_personality(self, gmn_generator):
        """Test handling of zero or missing personality traits."""
        zero_personality = {
            "exploration": 0,
            "cooperation": 0,
            "efficiency": 0}
        connections = gmn_generator._generate_connections(zero_personality)

        prefs = connections["C_pref"]["preferences"]
        # Should default to equal weights
        assert abs(prefs["Exploration"] - 1 / 3) < 0.01
        assert abs(prefs["Resources"] - 1 / 3) < 0.01
        assert abs(prefs["Social"] - 1 / 3) < 0.01

    def test_generate_connections_high_curiosity_novelty(self, gmn_generator):
        """Test that high curiosity adds novelty connection."""
        high_curiosity = {"curiosity": 0.8}
        connections = gmn_generator._generate_connections(high_curiosity)

        assert "C_novelty" in connections
        assert "novelty detection" in connections["C_novelty"]["description"].lower(
        )


class TestUpdateEquationGeneration:
    """Test update equation generation functionality."""

    def test_generate_update_equations_basic_equations(self, gmn_generator):
        """Test that basic update equations are generated."""
        equations = gmn_generator._generate_update_equations({})

        assert "belief_update" in equations
        assert "energy_dynamics" in equations

        # Check belief update structure
        belief_eq = equations["belief_update"]
        assert belief_eq["state"] == "S_beliefs"
        assert "learning_rate" in belief_eq["parameters"]

        # Check energy dynamics structure
        energy_eq = equations["energy_dynamics"]
        assert energy_eq["state"] == "S_energy"
        assert "action_cost" in energy_eq["parameters"]

    def test_generate_update_equations_curiosity_affects_learning_rate(
            self, gmn_generator):
        """Test that curiosity affects learning rate."""
        low_curiosity = {"curiosity": 0.1}
        high_curiosity = {"curiosity": 0.9}

        low_eq = gmn_generator._generate_update_equations(low_curiosity)
        high_eq = gmn_generator._generate_update_equations(high_curiosity)

        low_lr = low_eq["belief_update"]["parameters"]["learning_rate"]
        high_lr = high_eq["belief_update"]["parameters"]["learning_rate"]

        assert high_lr > low_lr

    def test_generate_update_equations_efficiency_optimization(
            self, gmn_generator):
        """Test that high efficiency adds optimization equation."""
        high_efficiency = {"efficiency": 0.8}
        equations = gmn_generator._generate_update_equations(high_efficiency)

        assert "efficiency_optimization" in equations
        assert "minimize" in equations["efficiency_optimization"]["formula"]

    def test_generate_update_equations_social_learning(self, gmn_generator):
        """Test that high cooperation adds social learning equation."""
        high_cooperation = {"cooperation": 0.8}
        equations = gmn_generator._generate_update_equations(high_cooperation)

        assert "social_learning" in equations
        assert "social_weight" in equations["social_learning"]["parameters"]

    def test_generate_update_equations_action_costs(self, gmn_generator):
        """Test that action costs are properly defined."""
        equations = gmn_generator._generate_update_equations({})

        action_costs = equations["energy_dynamics"]["parameters"]["action_cost"]
        assert "explore" in action_costs
        assert "exploit" in action_costs
        assert "communicate" in action_costs
        assert "rest" in action_costs
        assert "move" in action_costs

        # Rest should be negative (recovery)
        assert action_costs["rest"] < 0


class TestModelRefinement:
    """Test model refinement functionality."""

    def test_refine_model_basic_functionality(
            self, gmn_generator, sample_base_model):
        """Test basic model refinement."""
        patterns = []  # No patterns
        gmn_generator.validator.validate_model = Mock(return_value=(True, []))

        with patch("inference.gnn.generator.datetime") as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"

            result = gmn_generator.refine_model(sample_base_model, patterns)

            assert result["metadata"]["last_refined"] == "2024-01-01T12:00:00"
            assert result["metadata"]["model_version"] == 2
            assert result["metadata"]["refinement_changes"] == []

    def test_refine_model_with_patterns(
            self,
            gmn_generator,
            sample_base_model,
            sample_patterns):
        """Test model refinement with patterns."""
        gmn_generator.validator.validate_model = Mock(return_value=(True, []))

        with patch.object(gmn_generator, "_apply_pattern_to_model") as mock_apply:
            mock_apply.return_value = {"type": "test_change"}

            result = gmn_generator.refine_model(
                sample_base_model, sample_patterns)

            # Should have called apply_pattern_to_model for each pattern
            assert mock_apply.call_count == len(sample_patterns)

            # Should have recorded changes
            assert len(result["metadata"]["refinement_changes"]) == len(
                sample_patterns)

    def test_refine_model_confidence_threshold(
            self, gmn_generator, sample_base_model):
        """Test confidence threshold filtering."""
        low_confidence_patterns = [{"confidence": 0.5}
                                   ]  # Below default threshold of 0.8
        high_confidence_patterns = [{"confidence": 0.9}]  # Above threshold

        gmn_generator.validator.validate_model = Mock(return_value=(True, []))

        with patch.object(gmn_generator, "_apply_pattern_to_model") as mock_apply:
            mock_apply.return_value = {"type": "test_change"}

            # Test with low confidence
            gmn_generator.refine_model(
                sample_base_model, low_confidence_patterns)
            assert mock_apply.call_count == 0

            mock_apply.reset_mock()

            # Test with high confidence
            gmn_generator.refine_model(
                sample_base_model, high_confidence_patterns)
            assert mock_apply.call_count == 1

    def test_refine_model_custom_threshold(
            self, gmn_generator, sample_base_model):
        """Test custom confidence threshold."""
        patterns = [{"confidence": 0.6}]
        gmn_generator.validator.validate_model = Mock(return_value=(True, []))

        with patch.object(gmn_generator, "_apply_pattern_to_model") as mock_apply:
            mock_apply.return_value = {"type": "test_change"}

            # With default threshold (0.8), should not apply
            gmn_generator.refine_model(sample_base_model, patterns)
            assert mock_apply.call_count == 0

            mock_apply.reset_mock()

            # With custom threshold (0.5), should apply
            gmn_generator.refine_model(
                sample_base_model, patterns, confidence_threshold=0.5)
            assert mock_apply.call_count == 1

    def test_refine_model_validation_failure_reverts(
        self, gmn_generator, sample_base_model, sample_patterns
    ):
        """Test that validation failure reverts changes."""
        gmn_generator.validator.validate_model = Mock(
            return_value=(False, ["Validation error"]))

        with patch.object(gmn_generator, "_apply_pattern_to_model") as mock_apply:
            mock_apply.return_value = {"type": "test_change"}

            result = gmn_generator.refine_model(
                sample_base_model, sample_patterns)

            # Should return original model
            assert result == sample_base_model


class TestPatternApplication:
    """Test pattern application functionality."""

    def test_apply_successful_action_sequence_pattern(
            self, gmn_generator, sample_base_model):
        """Test applying successful action sequence pattern."""
        pattern = {
            "type": "successful_action_sequence",
            "dominant_action": "explore",
            "success_rate": 0.85,
            "confidence": 0.9,
        }

        with patch("inference.gnn.generator.datetime") as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"

            result = gmn_generator._apply_pattern_to_model(
                sample_base_model, pattern)

            assert result is not None
            assert result["pattern_type"] == "successful_action_sequence"
            assert result["confidence"] == 0.9
            assert len(result["changes"]) == 1

            # Check that action bias was added
            action_biases = sample_base_model["connections"]["C_pref"]["action_biases"]
            assert "explore" in action_biases

    def test_apply_environmental_correlation_pattern(
            self, gmn_generator, sample_base_model):
        """Test applying environmental correlation pattern."""
        pattern = {
            "type": "environmental_correlation",
            "correlation": {
                "observation": "resource_nearby",
                "state": "successful_exploit",
                "strength": 0.8,
            },
            "confidence": 0.85,
        }

        result = gmn_generator._apply_pattern_to_model(
            sample_base_model, pattern)

        assert result is not None
        assert result["pattern_type"] == "environmental_correlation"
        assert len(result["changes"]) == 1

        # Check that correlation was added
        correlations = sample_base_model["connections"]["C_likelihood"]["correlations"]
        assert len(correlations) == 1
        assert correlations[0]["observation"] == "resource_nearby"

    def test_apply_preference_adjustment_pattern(
            self, gmn_generator, sample_base_model):
        """Test applying preference adjustment pattern."""
        pattern = {
            "type": "preference_adjustment",
            "adjustments": {"Exploration": 0.1, "Resources": -0.05},
            "confidence": 0.8,
        }

        original_exploration = sample_base_model["connections"]["C_pref"]["preferences"][
            "Exploration"
        ]

        result = gmn_generator._apply_pattern_to_model(
            sample_base_model, pattern)

        assert result is not None
        assert result["pattern_type"] == "preference_adjustment"
        assert len(result["changes"]) == 2

        # Check that preferences were adjusted
        new_exploration = sample_base_model["connections"]["C_pref"]["preferences"]["Exploration"]
        assert new_exploration == original_exploration + 0.1

    def test_apply_pattern_unknown_type(
            self, gmn_generator, sample_base_model):
        """Test applying unknown pattern type."""
        pattern = {"type": "unknown_pattern_type", "confidence": 0.9}

        result = gmn_generator._apply_pattern_to_model(
            sample_base_model, pattern)

        # Should return None for unknown pattern types
        assert result is None

    def test_apply_pattern_no_changes(self, gmn_generator, sample_base_model):
        """Test pattern application that results in no changes."""
        # Pattern with action that already exists
        pattern = {
            "type": "successful_action_sequence",
            "dominant_action": "",  # Empty action
            "confidence": 0.9,
        }

        result = gmn_generator._apply_pattern_to_model(
            sample_base_model, pattern)

        # Should return None when no changes are made
        assert result is None

    def test_apply_pattern_preference_bounds_checking(
            self, gmn_generator, sample_base_model):
        """Test that preference adjustments respect bounds [0, 1]."""
        # Set initial preference to near maximum
        sample_base_model["connections"]["C_pref"]["preferences"]["Exploration"] = 0.95

        pattern = {
            "type": "preference_adjustment",
            "adjustments": {"Exploration": 0.2},  # Would exceed 1.0
            "confidence": 0.8,
        }

        gmn_generator._apply_pattern_to_model(sample_base_model, pattern)

        # Should be clamped to 1.0
        final_pref = sample_base_model["connections"]["C_pref"]["preferences"]["Exploration"]
        assert final_pref == 1.0


class TestExperienceBasedGeneration:
    """Test experience-based model generation functionality."""

    def test_generate_from_experience_basic(
        self, gmn_generator, sample_base_model, sample_experience_summary
    ):
        """Test basic experience-based generation."""
        gmn_generator.validator.validate_model = Mock(return_value=(True, []))

        with patch.object(gmn_generator, "_extract_patterns_from_experience") as mock_extract:
            mock_extract.return_value = []

            result = gmn_generator.generate_from_experience(
                sample_base_model, sample_experience_summary
            )

            mock_extract.assert_called_once_with(sample_experience_summary)
            assert result is not None

    def test_generate_from_experience_adds_world_model(
            self, gmn_generator, sample_base_model):
        """Test that sufficient observations add world model state."""
        experience_with_many_observations = {"unique_observations": 75}
        gmn_generator.validator.validate_model = Mock(return_value=(True, []))

        with patch.object(gmn_generator, "_extract_patterns_from_experience") as mock_extract:
            mock_extract.return_value = []

            result = gmn_generator.generate_from_experience(
                sample_base_model, experience_with_many_observations
            )

            assert "S_world_model" in result["state_space"]
            assert "Graph[Location, Connection]" in result["state_space"]["S_world_model"]["type"]

    def test_extract_patterns_from_experience_action_patterns(
        self, gmn_generator, sample_experience_summary
    ):
        """Test extraction of action patterns from experience."""
        patterns = gmn_generator._extract_patterns_from_experience(
            sample_experience_summary)

        # Should extract successful action sequence pattern for 'explore'
        action_patterns = [p for p in patterns if p["type"]
                           == "successful_action_sequence"]
        assert len(action_patterns) >= 1

        explore_pattern = next(
            (p for p in action_patterns if p["dominant_action"] == "explore"), None)
        assert explore_pattern is not None
        assert explore_pattern["success_rate"] == 42 / \
            50  # success_count/count

    def test_extract_patterns_from_experience_correlation_patterns(
        self, gmn_generator, sample_experience_summary
    ):
        """Test extraction of correlation patterns from experience."""
        patterns = gmn_generator._extract_patterns_from_experience(
            sample_experience_summary)

        # Should extract environmental correlation pattern
        corr_patterns = [p for p in patterns if p["type"]
                         == "environmental_correlation"]
        assert len(corr_patterns) >= 1

        corr_pattern = corr_patterns[0]
        assert corr_pattern["correlation"]["observation"] == "resource_nearby"
        assert corr_pattern["confidence"] == 0.8  # correlation strength

    def test_extract_patterns_confidence_calculation(self, gmn_generator):
        """Test confidence calculation in pattern extraction."""
        experience = {
            "action_statistics": {
                # High count, high success
                "test_action": {"count": 100, "success_count": 90},
            }
        }

        patterns = gmn_generator._extract_patterns_from_experience(experience)
        action_pattern = patterns[0]

        # Confidence should be high due to high count and success rate
        assert action_pattern["confidence"] > 0.9

    def test_extract_patterns_minimum_thresholds(self, gmn_generator):
        """Test that minimum thresholds are respected."""
        low_experience = {
            "action_statistics": {
                # Below count threshold
                "rare_action": {"count": 5, "success_count": 4},
            },
            # Below thresholds
            "observed_correlations": [{"correlation": 0.3, "occurrences": 2}],
        }

        patterns = gmn_generator._extract_patterns_from_experience(
            low_experience)

        # Should not extract patterns due to low counts/correlations
        assert len(patterns) == 0


class TestModelExport:
    """Test model export functionality."""

    def test_export_to_gnn_format_file_creation(
            self, gmn_generator, sample_base_model):
        """Test that export creates a file with correct content."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".gnn.md", delete=False) as tmp_file:
            file_path = tmp_file.name

        try:
            gmn_generator.export_to_gnn_format(sample_base_model, file_path)

            # Check that file was created
            assert Path(file_path).exists()

            # Check file content
            with open(file_path, "r") as f:
                content = f.read()

            assert "# GNN Model: TestAgent" in content
            assert "Class: Explorer" in content
            assert "## State Space" in content
            assert "## Connections" in content
            assert "## Update Equations" in content

        finally:
            # Clean up
            Path(file_path).unlink(missing_ok=True)

    def test_export_to_gnn_format_content_structure(
            self, gmn_generator, sample_base_model):
        """Test the structure and content of exported GNN format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".gnn.md", delete=False) as tmp_file:
            file_path = tmp_file.name

        try:
            gmn_generator.export_to_gnn_format(sample_base_model, file_path)

            with open(file_path, "r") as f:
                lines = f.readlines()

            content = "".join(lines)

            # Check model information
            assert sample_base_model["model"]["name"] in content
            assert str(sample_base_model["metadata"]
                       ["model_version"]) in content

            # Check state space entries
            for state_name in sample_base_model["state_space"]:
                assert state_name in content

            # Check connections
            for conn_name in sample_base_model["connections"]:
                assert conn_name in content

            # Check update equations
            for eq_name in sample_base_model["update_equations"]:
                assert eq_name in content

        finally:
            Path(file_path).unlink(missing_ok=True)

    @patch("inference.gnn.generator.logger")
    def test_export_to_gnn_format_logging(
            self, mock_logger, gmn_generator, sample_base_model):
        """Test that export logs completion."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".gnn.md", delete=False) as tmp_file:
            file_path = tmp_file.name

        try:
            gmn_generator.export_to_gnn_format(sample_base_model, file_path)

            mock_logger.info.assert_called_with(
                f"Exported GNN model to {file_path}")

        finally:
            Path(file_path).unlink(missing_ok=True)


class TestIntegrationScenarios:
    """Test complete integration scenarios."""

    def test_full_model_lifecycle(self, gmn_generator, sample_personality):
        """Test complete model lifecycle from generation to refinement."""
        # Mock validator to always return valid
        gmn_generator.validator.validate_model = Mock(return_value=(True, []))

        # 1. Generate base model
        base_model = gmn_generator.generate_base_model(
            "TestAgent", "Explorer", sample_personality)
        assert base_model["metadata"]["model_version"] == 1

        # 2. Create some patterns
        patterns = [
            {
                "type": "successful_action_sequence",
                "dominant_action": "explore",
                "success_rate": 0.9,
                "confidence": 0.85,
            }
        ]

        # 3. Refine model
        refined_model = gmn_generator.refine_model(base_model, patterns)
        assert refined_model["metadata"]["model_version"] == 2
        assert len(refined_model["metadata"]["refinement_changes"]) > 0

        # 4. Generate from experience
        experience = {
            "action_statistics": {
                "explore": {
                    "count": 50,
                    "success_count": 45}},
            "unique_observations": 60,
        }
        final_model = gmn_generator.generate_from_experience(
            refined_model, experience)
        assert "S_world_model" in final_model["state_space"]

    def test_different_agent_classes_generate_different_models(
        self, gmn_generator, sample_personality
    ):
        """Test that different agent classes produce different models."""
        gmn_generator.validator.validate_model = Mock(return_value=(True, []))

        agent_classes = ["Explorer", "Merchant", "Scholar", "Guardian"]
        models = {}

        for agent_class in agent_classes:
            model = gmn_generator.generate_base_model(
                f"Test{agent_class}", agent_class, sample_personality
            )
            models[agent_class] = model

        # Check that models have different characteristics
        explorer_prefs = models["Explorer"]["connections"]["C_pref"]["preferences"]
        merchant_prefs = models["Merchant"]["connections"]["C_pref"]["preferences"]

        # Explorer should prefer exploration more than merchant
        assert explorer_prefs["Exploration"] > merchant_prefs["Exploration"]
        assert merchant_prefs["Resources"] > explorer_prefs["Resources"]

    def test_personality_extremes_produce_different_models(
            self, gmn_generator):
        """Test that extreme personalities produce noticeably different models."""
        gmn_generator.validator.validate_model = Mock(return_value=(True, []))

        high_exploration = {
            "exploration": 100,
            "cooperation": 0,
            "efficiency": 0,
            "curiosity": 100}
        high_cooperation = {
            "exploration": 0,
            "cooperation": 100,
            "efficiency": 0}

        explorer_model = gmn_generator.generate_base_model(
            "HighExplorer", "Explorer", high_exploration
        )
        social_model = gmn_generator.generate_base_model(
            "HighSocial", "Explorer", high_cooperation)

        # Check different characteristics
        explorer_model["connections"]
        social_model["connections"]

        # High exploration should have novelty seeking
        assert "S_novelty_seeking" in explorer_model["state_space"]
        assert "S_novelty_seeking" not in social_model["state_space"]


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_model_refinement_with_empty_patterns(
            self, gmn_generator, sample_base_model):
        """Test model refinement with empty pattern list."""
        gmn_generator.validator.validate_model = Mock(return_value=(True, []))

        result = gmn_generator.refine_model(sample_base_model, [])

        # Should still update metadata but make no changes
        assert result["metadata"]["model_version"] == 2
        assert result["metadata"]["refinement_changes"] == []

    def test_pattern_application_with_missing_connections(self, gmn_generator):
        """Test pattern application when expected connections are missing."""
        minimal_model = {
            "connections": {},  # Missing expected connections
            "state_space": {},
            "update_equations": {},
            "metadata": {},
        }

        pattern = {
            "type": "successful_action_sequence",
            "dominant_action": "explore",
            "confidence": 0.9,
        }

        # Should handle gracefully without crashing
        result = gmn_generator._apply_pattern_to_model(minimal_model, pattern)
        assert result is None

    def test_experience_extraction_with_missing_data(self, gmn_generator):
        """Test pattern extraction with incomplete experience data."""
        incomplete_experience = {
            # Missing action_statistics
            "observed_correlations": []
        }

        patterns = gmn_generator._extract_patterns_from_experience(
            incomplete_experience)

        # Should handle gracefully
        assert isinstance(patterns, list)
        assert len(patterns) == 0

    def test_state_space_generation_with_empty_personality(
            self, gmn_generator):
        """Test state space generation with empty personality."""
        empty_personality = {}

        state_space = gmn_generator._generate_state_space(
            "Explorer", empty_personality)

        # Should still generate basic states
        assert "S_energy" in state_space
        assert "S_position" in state_space
        assert "A_actions" in state_space
