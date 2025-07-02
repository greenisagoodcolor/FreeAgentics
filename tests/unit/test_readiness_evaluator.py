"""
Comprehensive tests for coalitions.readiness.readiness_evaluator module.

Tests the agent readiness evaluation system including comprehensive scoring
across multiple dimensions for deployment readiness decisions.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from coalitions.readiness.readiness_evaluator import (
    AgentReadinessChecker,
    AgentReadinessEvaluator,
    ReadinessEvaluator,
    ReadinessLevel,
    ReadinessMetrics,
    ReadinessScore,
    ReadinessThresholds,
)


@pytest.fixture
def readiness_evaluator():
    """Create a ReadinessEvaluator instance for testing."""
    return ReadinessEvaluator()


@pytest.fixture
def agent_readiness_evaluator():
    """Create an AgentReadinessEvaluator instance for testing."""
    return AgentReadinessEvaluator()


@pytest.fixture
def agent_readiness_checker():
    """Create an AgentReadinessChecker instance for testing."""
    return AgentReadinessChecker()


@pytest.fixture
def sample_readiness_thresholds():
    """Create sample readiness thresholds for testing."""
    return ReadinessThresholds(
        min_experiences=500,
        min_patterns=25,
        pattern_confidence=0.80,
        success_rate=0.85,
        complex_goals_completed=3,
        overall_threshold=0.80,
    )


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = Mock()
    agent.id = "test_agent_001"

    # Mock knowledge graph
    agent.knowledge_graph = Mock()
    agent.knowledge_graph.experiences = [
        Mock() for _ in range(750)]  # 750 experiences
    agent.knowledge_graph.patterns = {
        f"pattern_{i}": Mock(confidence=0.85 + (i * 0.01)) for i in range(30)
    }

    # Mock stats
    agent.stats = {
        "total_goals_attempted": 100,
        "successful_goals": 90,
        "complex_goals_completed": 8,
        "total_interactions": 200,
        "successful_interactions": 180,
        "knowledge_items_shared": 15,
        "collaborators": {"agent_002", "agent_003", "agent_004"},
        "energy_efficiency": 0.85,
        "sustainability_score": 0.90,
    }

    # Mock model update history
    agent.model_update_history = [{"magnitude": 0.005} for _ in range(80)] + [
        {"magnitude": 0.15} for _ in range(20)
    ]

    return agent


@pytest.fixture
def mock_agent_poor():
    """Create a mock agent with poor performance for testing."""
    agent = Mock()
    agent.id = "poor_agent_001"

    # Minimal knowledge graph
    agent.knowledge_graph = Mock()
    agent.knowledge_graph.experiences = [
        Mock() for _ in range(50)]  # Few experiences
    agent.knowledge_graph.patterns = {
        # Low confidence
        f"pattern_{i}": Mock(confidence=0.60) for i in range(5)
    }

    # Poor stats
    agent.stats = {
        "total_goals_attempted": 50,
        "successful_goals": 25,  # 50% success rate
        "complex_goals_completed": 1,
        "total_interactions": 100,
        "successful_interactions": 60,  # 60% success rate
        "knowledge_items_shared": 2,
        "collaborators": {"agent_002"},
        "energy_efficiency": 0.40,
        "sustainability_score": 0.35,
    }

    # Unstable model
    agent.model_update_history = [{"magnitude": 0.25}
                                  for _ in range(100)]  # Large updates

    return agent


class TestReadinessLevel:
    """Test the ReadinessLevel enum."""

    def test_readiness_level_values(self):
        """Test that readiness level enum has correct values."""
        assert ReadinessLevel.NOT_READY.value == "not_ready"
        assert ReadinessLevel.PARTIALLY_READY.value == "partially_ready"
        assert ReadinessLevel.READY.value == "ready"
        assert ReadinessLevel.FULLY_READY.value == "fully_ready"


class TestReadinessThresholds:
    """Test the ReadinessThresholds dataclass."""

    def test_thresholds_creation(self):
        """Test creating readiness thresholds with custom values."""
        thresholds = ReadinessThresholds(
            min_experiences=2000,
            min_patterns=100,
            pattern_confidence=0.95,
            success_rate=0.95,
            complex_goals_completed=10,
            overall_threshold=0.90,
        )

        assert thresholds.min_experiences == 2000
        assert thresholds.min_patterns == 100
        assert thresholds.pattern_confidence == 0.95
        assert thresholds.success_rate == 0.95
        assert thresholds.complex_goals_completed == 10
        assert thresholds.overall_threshold == 0.90

    def test_thresholds_defaults(self):
        """Test default threshold values."""
        thresholds = ReadinessThresholds()

        assert thresholds.min_experiences == 1000
        assert thresholds.min_patterns == 50
        assert thresholds.pattern_confidence == 0.85
        assert thresholds.success_rate == 0.9
        assert thresholds.complex_goals_completed == 5
        assert thresholds.overall_threshold == 0.85


class TestReadinessScore:
    """Test the ReadinessScore dataclass."""

    def test_readiness_score_creation(self):
        """Test creating a readiness score with all fields."""
        timestamp = datetime.now()
        score = ReadinessScore(
            agent_id="test_agent",
            timestamp=timestamp,
            overall_score=0.87,
            is_ready=True,
            knowledge_maturity=0.90,
            goal_achievement=0.85,
            model_stability=0.88,
            collaboration=0.82,
            resource_management=0.90,
            metrics={"test": "data"},
            recommendations=["Improve collaboration"],
        )

        assert score.agent_id == "test_agent"
        assert score.timestamp == timestamp
        assert score.overall_score == 0.87
        assert score.is_ready is True
        assert score.knowledge_maturity == 0.90
        assert score.goal_achievement == 0.85
        assert score.model_stability == 0.88
        assert score.collaboration == 0.82
        assert score.resource_management == 0.90
        assert score.metrics == {"test": "data"}
        assert score.recommendations == ["Improve collaboration"]

    def test_readiness_score_defaults(self):
        """Test readiness score with default values."""
        score = ReadinessScore(
            agent_id="minimal_test",
            timestamp=datetime.now(),
        )

        assert score.overall_score == 0.0
        assert score.is_ready is False
        assert score.knowledge_maturity == 0.0
        assert score.goal_achievement == 0.0
        assert score.model_stability == 0.0
        assert score.collaboration == 0.0
        assert score.resource_management == 0.0
        assert score.metrics == {}
        assert score.recommendations == []


class TestReadinessMetrics:
    """Test the ReadinessMetrics dataclass."""

    def test_readiness_metrics_creation(self):
        """Test creating readiness metrics."""
        metrics = ReadinessMetrics(
            overall_score=0.85,
            component_scores={"model_trained": 1.0, "knowledge_graph": 0.8},
            missing_components=["testing"],
            recommendations=["Complete testing component"],
        )

        assert metrics.overall_score == 0.85
        assert metrics.component_scores == {
            "model_trained": 1.0, "knowledge_graph": 0.8}
        assert metrics.missing_components == ["testing"]
        assert metrics.recommendations == ["Complete testing component"]


class TestReadinessEvaluator:
    """Test ReadinessEvaluator functionality."""

    def test_evaluator_initialization(self, readiness_evaluator):
        """Test that evaluator initializes with correct criteria."""
        expected_criteria = {
            "model_trained": 0.3,
            "knowledge_graph": 0.2,
            "configuration": 0.2,
            "dependencies": 0.15,
            "testing": 0.15,
        }

        assert readiness_evaluator.criteria == expected_criteria

    def test_evaluate_readiness_all_components_present(
            self, readiness_evaluator):
        """Test evaluation with all components present."""
        agent_data = {
            "model_trained": True,
            "knowledge_graph": True,
            "configuration": True,
            "dependencies": True,
            "testing": True,
        }

        metrics = readiness_evaluator.evaluate_readiness(agent_data)

        assert metrics.overall_score == 1.0
        assert len(metrics.missing_components) == 0
        assert len(metrics.recommendations) == 0
        assert all(score == 1.0 for score in metrics.component_scores.values())

    def test_evaluate_readiness_missing_components(self, readiness_evaluator):
        """Test evaluation with missing components."""
        agent_data = {
            "model_trained": True,
            "knowledge_graph": False,
            "configuration": True,
            # missing dependencies and testing
        }

        metrics = readiness_evaluator.evaluate_readiness(agent_data)

        # Should have: model_trained (0.3) + configuration (0.2) = 0.5
        assert metrics.overall_score == 0.5
        assert "knowledge_graph" in metrics.missing_components
        assert "dependencies" in metrics.missing_components
        assert "testing" in metrics.missing_components
        assert len(metrics.recommendations) == 3

    def test_evaluate_readiness_empty_data(self, readiness_evaluator):
        """Test evaluation with empty agent data."""
        agent_data = {}

        metrics = readiness_evaluator.evaluate_readiness(agent_data)

        assert metrics.overall_score == 0.0
        assert len(metrics.missing_components) == 5
        assert len(metrics.recommendations) == 5

    def test_get_readiness_level_fully_ready(self, readiness_evaluator):
        """Test readiness level determination for fully ready."""
        metrics = ReadinessMetrics(
            overall_score=0.95,
            component_scores={},
            missing_components=[],
            recommendations=[],
        )

        level = readiness_evaluator.get_readiness_level(metrics)
        assert level == ReadinessLevel.FULLY_READY

    def test_get_readiness_level_ready(self, readiness_evaluator):
        """Test readiness level determination for ready."""
        metrics = ReadinessMetrics(
            overall_score=0.75,
            component_scores={},
            missing_components=[],
            recommendations=[],
        )

        level = readiness_evaluator.get_readiness_level(metrics)
        assert level == ReadinessLevel.READY

    def test_get_readiness_level_partially_ready(self, readiness_evaluator):
        """Test readiness level determination for partially ready."""
        metrics = ReadinessMetrics(
            overall_score=0.55,
            component_scores={},
            missing_components=[],
            recommendations=[],
        )

        level = readiness_evaluator.get_readiness_level(metrics)
        assert level == ReadinessLevel.PARTIALLY_READY

    def test_get_readiness_level_not_ready(self, readiness_evaluator):
        """Test readiness level determination for not ready."""
        metrics = ReadinessMetrics(
            overall_score=0.25,
            component_scores={},
            missing_components=[],
            recommendations=[],
        )

        level = readiness_evaluator.get_readiness_level(metrics)
        assert level == ReadinessLevel.NOT_READY


class TestAgentReadinessEvaluator:
    """Test AgentReadinessEvaluator functionality."""

    def test_evaluator_initialization_default_thresholds(
            self, agent_readiness_evaluator):
        """Test evaluator initialization with default thresholds."""
        assert isinstance(
            agent_readiness_evaluator.thresholds,
            ReadinessThresholds)
        assert agent_readiness_evaluator.thresholds.min_experiences == 1000

        expected_weights = {
            "knowledge": 0.25,
            "goals": 0.20,
            "model_stability": 0.20,
            "collaboration": 0.20,
            "resources": 0.15,
        }
        assert agent_readiness_evaluator.dimension_weights == expected_weights

    def test_evaluator_initialization_custom_thresholds(
            self, sample_readiness_thresholds):
        """Test evaluator initialization with custom thresholds."""
        evaluator = AgentReadinessEvaluator(sample_readiness_thresholds)
        assert evaluator.thresholds == sample_readiness_thresholds
        assert evaluator.thresholds.min_experiences == 500

    def test_evaluate_agent_high_performance(
            self, agent_readiness_evaluator, mock_agent):
        """Test evaluating a high-performance agent."""
        score = agent_readiness_evaluator.evaluate_agent(mock_agent)

        assert score.agent_id == "test_agent_001"
        assert isinstance(score.timestamp, datetime)
        assert score.overall_score > 0.8  # Should be high-scoring
        assert score.is_ready is True
        assert score.knowledge_maturity > 0.7
        assert score.goal_achievement > 0.8
        assert score.model_stability > 0.7
        assert score.collaboration > 0.7
        assert score.resource_management > 0.8
        assert len(score.recommendations) >= 0

    def test_evaluate_agent_poor_performance(
            self, agent_readiness_evaluator, mock_agent_poor):
        """Test evaluating a poor-performance agent."""
        score = agent_readiness_evaluator.evaluate_agent(mock_agent_poor)

        assert score.agent_id == "poor_agent_001"
        assert score.overall_score < 0.7  # Should be low-scoring
        assert score.is_ready is False
        assert score.knowledge_maturity < 0.7
        assert score.goal_achievement < 0.7
        assert score.model_stability < 0.7
        assert len(score.recommendations) > 0  # Should have recommendations

    def test_evaluate_knowledge_dimension(
            self, agent_readiness_evaluator, mock_agent):
        """Test knowledge dimension evaluation."""
        score = ReadinessScore(agent_id="test", timestamp=datetime.now())
        knowledge_score = agent_readiness_evaluator._evaluate_knowledge(
            mock_agent, score)

        assert 0.0 <= knowledge_score <= 1.0
        assert knowledge_score > 0.5  # Should be good with 750 experiences and 30 patterns
        assert "knowledge" in score.metrics
        assert "experience_count" in score.metrics["knowledge"]
        assert "pattern_count" in score.metrics["knowledge"]
        assert "avg_pattern_confidence" in score.metrics["knowledge"]

    def test_evaluate_goals_dimension(
            self, agent_readiness_evaluator, mock_agent):
        """Test goals dimension evaluation."""
        score = ReadinessScore(agent_id="test", timestamp=datetime.now())
        goals_score = agent_readiness_evaluator._evaluate_goals(
            mock_agent, score)

        assert 0.0 <= goals_score <= 1.0
        assert goals_score > 0.8  # Should be high with 90% success rate
        assert "goals" in score.metrics
        assert "success_rate" in score.metrics["goals"]
        assert "complex_completed" in score.metrics["goals"]

    def test_evaluate_model_stability(
            self, agent_readiness_evaluator, mock_agent):
        """Test model stability evaluation."""
        score = ReadinessScore(agent_id="test", timestamp=datetime.now())
        stability_score = agent_readiness_evaluator._evaluate_model_stability(
            mock_agent, score)

        assert 0.0 <= stability_score <= 1.0
        assert "model_stability" in score.metrics
        assert "is_converged" in score.metrics["model_stability"]
        assert "stable_iterations" in score.metrics["model_stability"]

    def test_evaluate_collaboration(
            self, agent_readiness_evaluator, mock_agent):
        """Test collaboration evaluation."""
        score = ReadinessScore(agent_id="test", timestamp=datetime.now())
        collab_score = agent_readiness_evaluator._evaluate_collaboration(
            mock_agent, score)

        assert 0.0 <= collab_score <= 1.0
        assert collab_score > 0.7  # Should be high with good interaction rates
        assert "collaboration" in score.metrics
        assert "successful_interactions" in score.metrics["collaboration"]
        assert "knowledge_shared" in score.metrics["collaboration"]
        assert "unique_collaborators" in score.metrics["collaboration"]

    def test_evaluate_resources(self, agent_readiness_evaluator, mock_agent):
        """Test resource management evaluation."""
        score = ReadinessScore(agent_id="test", timestamp=datetime.now())
        resource_score = agent_readiness_evaluator._evaluate_resources(
            mock_agent, score)

        assert 0.0 <= resource_score <= 1.0
        assert resource_score > 0.8  # Should be high with good efficiency scores
        assert "resources" in score.metrics
        assert "resource_efficiency" in score.metrics["resources"]
        assert "sustainability_score" in score.metrics["resources"]

    def test_evaluation_with_missing_attributes(
            self, agent_readiness_evaluator):
        """Test evaluation with agent missing attributes."""
        minimal_agent = Mock()
        minimal_agent.id = "minimal_agent"

        score = agent_readiness_evaluator.evaluate_agent(minimal_agent)

        assert score.agent_id == "minimal_agent"
        assert score.overall_score >= 0.0  # Should handle gracefully
        assert score.is_ready is False

    def test_recommendation_generation(
            self,
            agent_readiness_evaluator,
            mock_agent_poor):
        """Test recommendation generation for poor-performing agent."""
        score = agent_readiness_evaluator.evaluate_agent(mock_agent_poor)

        assert len(score.recommendations) > 0
        # Should have recommendations for each low-scoring dimension
        recommendation_text = " ".join(score.recommendations)
        assert any(
            keyword in recommendation_text.lower()
            for keyword in [
                "experience",
                "pattern",
                "goal",
                "training",
                "collaboration",
                "efficiency",
            ]
        )


class TestEvaluationHistory:
    """Test evaluation history and trend functionality."""

    def test_evaluation_history_storage(
            self, agent_readiness_evaluator, mock_agent):
        """Test that evaluation history is stored correctly."""
        # Perform multiple evaluations
        score1 = agent_readiness_evaluator.evaluate_agent(mock_agent)
        score2 = agent_readiness_evaluator.evaluate_agent(mock_agent)

        history = agent_readiness_evaluator.get_readiness_history(
            "test_agent_001")

        assert len(history) == 2
        assert history[0] == score1
        assert history[1] == score2

    def test_get_readiness_history_empty(self, agent_readiness_evaluator):
        """Test getting history for agent with no evaluations."""
        history = agent_readiness_evaluator.get_readiness_history(
            "nonexistent_agent")
        assert history == []

    def test_get_readiness_trend(self, agent_readiness_evaluator, mock_agent):
        """Test readiness trend generation."""
        # Perform evaluations
        agent_readiness_evaluator.evaluate_agent(mock_agent)
        agent_readiness_evaluator.evaluate_agent(mock_agent)

        trends = agent_readiness_evaluator.get_readiness_trend(
            "test_agent_001")

        assert "timestamps" in trends
        assert "overall" in trends
        assert "knowledge" in trends
        assert "goals" in trends
        assert "model_stability" in trends
        assert "collaboration" in trends
        assert "resources" in trends

        assert len(trends["timestamps"]) == 2
        assert len(trends["overall"]) == 2

    def test_get_readiness_trend_empty(self, agent_readiness_evaluator):
        """Test getting trend for agent with no history."""
        trends = agent_readiness_evaluator.get_readiness_trend(
            "nonexistent_agent")
        assert trends == {}

    def test_batch_evaluate(
            self,
            agent_readiness_evaluator,
            mock_agent,
            mock_agent_poor):
        """Test batch evaluation of multiple agents."""
        agents = [mock_agent, mock_agent_poor]

        results = agent_readiness_evaluator.batch_evaluate(agents)

        assert len(results) == 2
        assert "test_agent_001" in results
        assert "poor_agent_001" in results
        assert isinstance(results["test_agent_001"], ReadinessScore)
        assert isinstance(results["poor_agent_001"], ReadinessScore)

    def test_export_readiness_report(
            self, agent_readiness_evaluator, mock_agent):
        """Test exporting readiness report to file."""
        # Perform evaluation
        agent_readiness_evaluator.evaluate_agent(mock_agent)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            report_file = agent_readiness_evaluator.export_readiness_report(
                "test_agent_001", output_dir
            )

            assert report_file.exists()
            assert report_file.name == "readiness_report_test_agent_001.json"

            # Verify file content
            with open(report_file, "r") as f:
                report_data = json.load(f)

            assert report_data["agent_id"] == "test_agent_001"
            assert "latest_evaluation" in report_data
            assert "trends" in report_data
            assert "recommendation_summary" in report_data

    def test_export_readiness_report_no_history(
            self, agent_readiness_evaluator):
        """Test exporting report for agent with no evaluation history."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            report_file = agent_readiness_evaluator.export_readiness_report(
                "nonexistent_agent", output_dir
            )

            assert report_file.exists()

            with open(report_file, "r") as f:
                report_data = json.load(f)

            assert report_data["agent_id"] == "nonexistent_agent"
            assert report_data["latest_evaluation"] is None
            assert report_data["trends"] == {}
            assert report_data["recommendation_summary"] == []


class TestAgentReadinessChecker:
    """Test AgentReadinessChecker functionality."""

    def test_checker_initialization(self, agent_readiness_checker):
        """Test checker initialization."""
        assert isinstance(
            agent_readiness_checker.evaluator,
            ReadinessEvaluator)

    def test_check_agent_readiness(self, agent_readiness_checker):
        """Test checking agent readiness."""
        agent_config = {
            "model_trained": True,
            "knowledge_graph": True,
            "configuration": False,
            "dependencies": True,
            "testing": False,
        }

        metrics = agent_readiness_checker.check_agent_readiness(
            "test_agent", agent_config)

        assert isinstance(metrics, ReadinessMetrics)
        # Should have model_trained (0.3) + knowledge_graph (0.2) + dependencies
        # (0.15) = 0.65
        assert metrics.overall_score == 0.65
        assert "configuration" in metrics.missing_components
        assert "testing" in metrics.missing_components


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    def test_evaluate_agent_with_exception(self, agent_readiness_evaluator):
        """Test evaluation handles exceptions gracefully."""
        # Create agent that will cause exceptions
        broken_agent = Mock()
        broken_agent.id = "broken_agent"
        broken_agent.knowledge_graph = None  # Will cause AttributeError

        score = agent_readiness_evaluator.evaluate_agent(broken_agent)

        assert score.agent_id == "broken_agent"
        assert score.overall_score == 0.0
        assert score.is_ready is False

    def test_knowledge_evaluation_with_no_knowledge_graph(
            self, agent_readiness_evaluator):
        """Test knowledge evaluation with missing knowledge graph."""
        agent = Mock()
        agent.knowledge_graph = None

        score = ReadinessScore(agent_id="test", timestamp=datetime.now())
        knowledge_score = agent_readiness_evaluator._evaluate_knowledge(
            agent, score)

        assert knowledge_score == 0.0

    def test_goals_evaluation_with_no_stats(self, agent_readiness_evaluator):
        """Test goals evaluation with missing stats."""
        agent = Mock()
        # No stats attribute

        score = ReadinessScore(agent_id="test", timestamp=datetime.now())
        goals_score = agent_readiness_evaluator._evaluate_goals(agent, score)

        assert goals_score == 0.0

    def test_model_stability_insufficient_history(
            self, agent_readiness_evaluator):
        """Test model stability with insufficient update history."""
        agent = Mock()
        agent.model_update_history = [{"magnitude": 0.1}]  # Only 1 update

        score = ReadinessScore(agent_id="test", timestamp=datetime.now())
        stability_score = agent_readiness_evaluator._evaluate_model_stability(
            agent, score)

        assert stability_score == 0.0
        assert score.metrics["model_stability"]["is_converged"] is False

    def test_collaboration_zero_interactions(self, agent_readiness_evaluator):
        """Test collaboration evaluation with zero interactions."""
        agent = Mock()
        agent.stats = {
            "total_interactions": 0,
            "successful_interactions": 0,
            "knowledge_items_shared": 0,
            "collaborators": set(),
        }

        score = ReadinessScore(agent_id="test", timestamp=datetime.now())
        collab_score = agent_readiness_evaluator._evaluate_collaboration(
            agent, score)

        assert collab_score == 0.0

    def test_dimension_weights_sum_to_one(self, agent_readiness_evaluator):
        """Test that dimension weights sum to approximately 1.0."""
        total_weight = sum(
            agent_readiness_evaluator.dimension_weights.values())
        # Allow small floating point differences
        assert abs(total_weight - 1.0) < 0.01


class TestIntegrationScenarios:
    """Test complex integration scenarios."""

    def test_complete_evaluation_workflow(self, sample_readiness_thresholds):
        """Test complete evaluation workflow with custom thresholds."""
        evaluator = AgentReadinessEvaluator(sample_readiness_thresholds)

        # Create agent that meets custom thresholds
        agent = Mock()
        agent.id = "workflow_test_agent"

        # Knowledge graph meeting custom thresholds
        agent.knowledge_graph = Mock()
        agent.knowledge_graph.experiences = [
            Mock() for _ in range(600)]  # > 500
        agent.knowledge_graph.patterns = {
            # > 25, > 0.80 conf
            f"pattern_{i}": Mock(confidence=0.82) for i in range(30)
        }

        # Stats meeting custom thresholds
        agent.stats = {
            "total_goals_attempted": 50,
            "successful_goals": 45,  # 90% > 85%
            "complex_goals_completed": 5,  # > 3
            "total_interactions": 100,
            "successful_interactions": 90,
            "knowledge_items_shared": 12,
            "collaborators": {"agent_002", "agent_003"},
            "energy_efficiency": 0.85,
            "sustainability_score": 0.80,
        }

        # Stable model
        agent.model_update_history = [{"magnitude": 0.005} for _ in range(100)]

        # Evaluate
        score = evaluator.evaluate_agent(agent)

        # Should meet custom threshold of 0.80
        assert score.overall_score >= 0.80
        assert score.is_ready is True

        # Test history and trends
        history = evaluator.get_readiness_history("workflow_test_agent")
        assert len(history) == 1

        trends = evaluator.get_readiness_trend("workflow_test_agent")
        assert len(trends["overall"]) == 1

        # Test export
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            report_file = evaluator.export_readiness_report(
                "workflow_test_agent", output_dir)
            assert report_file.exists()

    def test_readiness_improvement_over_time(self, agent_readiness_evaluator):
        """Test tracking improvement over multiple evaluations."""
        # Create agent that improves over time
        agent = Mock()
        agent.id = "improving_agent"

        # Initial poor state
        agent.knowledge_graph = Mock()
        agent.knowledge_graph.experiences = [Mock() for _ in range(100)]
        agent.knowledge_graph.patterns = {
            f"pattern_{i}": Mock(
                confidence=0.60) for i in range(10)}

        agent.stats = {
            "total_goals_attempted": 20,
            "successful_goals": 10,  # 50% success
            "complex_goals_completed": 1,
            "total_interactions": 30,
            "successful_interactions": 20,
            "knowledge_items_shared": 2,
            "collaborators": {"agent_002"},
            "energy_efficiency": 0.50,
            "sustainability_score": 0.45,
        }

        agent.model_update_history = [{"magnitude": 0.20} for _ in range(50)]

        # First evaluation (poor)
        score1 = agent_readiness_evaluator.evaluate_agent(agent)

        # Simulate improvement
        agent.knowledge_graph.experiences = [
            Mock() for _ in range(800)]  # More experience
        agent.knowledge_graph.patterns = {
            f"pattern_{i}": Mock(
                confidence=0.85) for i in range(40)}

        agent.stats.update(
            {
                "total_goals_attempted": 100,
                "successful_goals": 90,  # 90% success
                "complex_goals_completed": 8,
                "successful_interactions": 85,
                "knowledge_items_shared": 15,
                "collaborators": {"agent_002", "agent_003", "agent_004"},
                "energy_efficiency": 0.85,
                "sustainability_score": 0.90,
            }
        )

        agent.model_update_history = [{"magnitude": 0.005}
                                      for _ in range(100)]  # Stable

        # Second evaluation (improved)
        score2 = agent_readiness_evaluator.evaluate_agent(agent)

        # Verify improvement
        assert score2.overall_score > score1.overall_score
        assert len(score2.recommendations) < len(score1.recommendations)

        # Check trends show improvement
        trends = agent_readiness_evaluator.get_readiness_trend(
            "improving_agent")
        assert trends["overall"][1] > trends["overall"][0]
