"""
Module for FreeAgentics Active Inference implementation.
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from typing import Optional

from coalitions.readiness.readiness_evaluator import (
    AgentReadinessEvaluator,
    ReadinessScore,
    ReadinessThresholds,
)


class TestReadinessEvaluator:
    """Test suite for AgentReadinessEvaluator."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator with default thresholds."""
        return AgentReadinessEvaluator()

    @pytest.fixture
    def custom_evaluator(self):
        """Create evaluator with custom thresholds."""
        thresholds = ReadinessThresholds(
            min_experiences=500,
            min_patterns=25,
            pattern_confidence=0.8,
            success_rate=0.85,
            complex_goals_completed=3,
            overall_threshold=0.8,
        )
        return AgentReadinessEvaluator(thresholds)

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent with test data."""
        agent = Mock()
        agent.id = "test_agent_123"
        # Mock knowledge graph
        agent.knowledge_graph = Mock()
        agent.knowledge_graph.experiences = [Mock() for _ in range(1200)]
        # Mock patterns with confidence scores
        patterns = []
        for i in range(60):
            pattern = Mock()
            pattern.confidence = 0.85 + (i % 10) * 0.01
            patterns.append(pattern)
        agent.knowledge_graph.patterns = {f"p_{i}": p for i, p in enumerate(patterns)}
        # Mock stats
        agent.stats = {
            "total_goals_attempted": 100,
            "successful_goals": 92,
            "complex_goals_completed": 6,
            "total_interactions": 50,
            "successful_interactions": 45,
            "knowledge_items_shared": 15,
            "collaborators": {"agent_1", "agent_2", "agent_3"},
            "energy_efficiency": 0.85,
            "resource_waste_ratio": 0.15,
            "sustainability_score": 0.88,
            "avg_energy_level": 0.75,
        }
        # Mock model update history
        agent.model_update_history = []
        for i in range(150):
            magnitude = 0.1 * np.exp(-i / 30)  # Exponential decay
            agent.model_update_history.append({"magnitude": magnitude})
        return agent

    def test_evaluate_ready_agent(self, evaluator, mock_agent) -> None:
        """Test evaluation of an agent that meets all criteria."""
        score = evaluator.evaluate_agent(mock_agent)
        assert score.agent_id == "test_agent_123"
        assert score.is_ready is True
        assert score.overall_score >= 0.85
        # Check all dimensions are evaluated
        assert score.knowledge_maturity > 0
        assert score.goal_achievement > 0
        assert score.model_stability > 0
        assert score.collaboration > 0
        assert score.resource_management > 0
        # Check metrics are populated
        assert "knowledge" in score.metrics
        assert "goals" in score.metrics
        assert "model_stability" in score.metrics
        assert "collaboration" in score.metrics
        assert "resources" in score.metrics

    def test_evaluate_not_ready_agent(self, evaluator) -> None:
        """Test evaluation of an agent that doesn't meet criteria."""
        agent = Mock()
        agent.id = "novice_agent"
        # Insufficient experience
        agent.knowledge_graph = Mock()
        agent.knowledge_graph.experiences = [Mock() for _ in range(100)]
        agent.knowledge_graph.patterns = {}
        agent.stats = {
            "total_goals_attempted": 10,
            "successful_goals": 5,
            "complex_goals_completed": 0,
            "total_interactions": 5,
            "successful_interactions": 3,
            "knowledge_items_shared": 1,
            "collaborators": set(),
            "energy_efficiency": 0.5,
            "resource_waste_ratio": 0.5,
            "sustainability_score": 0.4,
            "avg_energy_level": 0.5,
        }
        agent.model_update_history = [{"magnitude": 0.1} for _ in range(10)]
        score = evaluator.evaluate_agent(agent)
        assert score.is_ready is False
        assert score.overall_score < 0.85
        assert len(score.recommendations) > 0

    def test_knowledge_evaluation(self, evaluator, mock_agent) -> None:
        """Test knowledge maturity evaluation specifically."""
        score = ReadinessScore(agent_id=mock_agent.id, timestamp=datetime.now())
        knowledge_score = evaluator._evaluate_knowledge(mock_agent, score)
        assert knowledge_score > 0.8
        assert score.metrics["knowledge"]["experience_count"] == 1200
        assert score.metrics["knowledge"]["pattern_count"] == 60
        assert score.metrics["knowledge"]["avg_pattern_confidence"] > 0.85

    def test_goal_evaluation(self, evaluator, mock_agent) -> None:
        """Test goal achievement evaluation."""
        score = ReadinessScore(agent_id=mock_agent.id, timestamp=datetime.now())
        goal_score = evaluator._evaluate_goals(mock_agent, score)
        assert goal_score > 0.9
        assert score.metrics["goals"]["success_rate"] == 0.92
        assert score.metrics["goals"]["complex_completed"] == 6

    def test_model_stability_evaluation(self, evaluator, mock_agent) -> None:
        """Test model stability evaluation."""
        score = ReadinessScore(agent_id=mock_agent.id, timestamp=datetime.now())
        stability_score = evaluator._evaluate_model_stability(mock_agent, score)
        assert stability_score > 0.8
        assert score.metrics["model_stability"]["is_converged"] is True
        assert score.metrics["model_stability"]["stable_iterations"] > 100

    def test_collaboration_evaluation(self, evaluator, mock_agent) -> None:
        """Test collaboration evaluation."""
        score = ReadinessScore(agent_id=mock_agent.id, timestamp=datetime.now())
        collab_score = evaluator._evaluate_collaboration(mock_agent, score)
        assert collab_score > 0.8
        assert score.metrics["collaboration"]["successful_interactions"] == 45
        assert score.metrics["collaboration"]["knowledge_shared"] == 15
        assert score.metrics["collaboration"]["unique_collaborators"] == 3

    def test_resource_evaluation(self, evaluator, mock_agent) -> None:
        """Test resource management evaluation."""
        score = ReadinessScore(agent_id=mock_agent.id, timestamp=datetime.now())
        resource_score = evaluator._evaluate_resources(mock_agent, score)
        assert resource_score > 0.8
        assert score.metrics["resources"]["resource_efficiency"] == 0.85
        assert score.metrics["resources"]["sustainability_score"] == 0.88

    def test_edge_case_barely_ready(self, custom_evaluator) -> None:
        """Test edge case of agent barely meeting thresholds."""
        agent = Mock()
        agent.id = "edge_case_agent"
        # Exactly at thresholds
        agent.knowledge_graph = Mock()
        agent.knowledge_graph.experiences = [Mock() for _ in range(500)]
        patterns = []
        for i in range(25):
            pattern = Mock()
            pattern.confidence = 0.8
            patterns.append(pattern)
        agent.knowledge_graph.patterns = {f"p_{i}": p for i, p in enumerate(patterns)}
        agent.stats = {
            "total_goals_attempted": 100,
            "successful_goals": 85,
            "complex_goals_completed": 3,
            "total_interactions": 25,
            "successful_interactions": 20,
            "knowledge_items_shared": 10,
            "collaborators": {"a1", "a2", "a3", "a4", "a5"},
            "energy_efficiency": 0.8,
            "resource_waste_ratio": 0.2,
            "sustainability_score": 0.9,
            "avg_energy_level": 0.7,
        }
        # Barely stable model
        agent.model_update_history = []
        for i in range(110):
            magnitude = 0.009 if i > 10 else 0.02
            agent.model_update_history.append({"magnitude": magnitude})
        score = custom_evaluator.evaluate_agent(agent)
        # Should be ready but with lower scores
        assert score.is_ready is True
        assert 0.8 <= score.overall_score <= 0.85

    def test_recommendations_generation(self, evaluator) -> None:
        """Test that appropriate recommendations are generated."""
        agent = Mock()
        agent.id = "needs_improvement"
        # Poor performance across dimensions
        agent.knowledge_graph = Mock()
        agent.knowledge_graph.experiences = [Mock() for _ in range(500)]
        agent.knowledge_graph.patterns = {f"p_{i}": Mock(confidence=0.7) for i in range(20)}
        agent.stats = {
            "total_goals_attempted": 50,
            "successful_goals": 35,
            "complex_goals_completed": 1,
            "total_interactions": 15,
            "successful_interactions": 10,
            "knowledge_items_shared": 5,
            "collaborators": {"a1", "a2"},
            "energy_efficiency": 0.6,
            "resource_waste_ratio": 0.4,
            "sustainability_score": 0.6,
            "avg_energy_level": 0.5,
        }
        agent.model_update_history = [{"magnitude": 0.05} for _ in range(50)]
        score = evaluator.evaluate_agent(agent)
        assert not score.is_ready
        assert len(score.recommendations) >= 5
        # Check specific recommendations
        rec_text = " ".join(score.recommendations)
        assert "experience" in rec_text.lower()
        assert "pattern" in rec_text.lower()
        assert "goal" in rec_text.lower()
        assert "converged" in rec_text.lower()
        assert "efficiency" in rec_text.lower()

    def test_readiness_history(self, evaluator, mock_agent) -> None:
        """Test that evaluation history is maintained."""
        # Evaluate multiple times
        scores = []
        for _ in range(3):
            score = evaluator.evaluate_agent(mock_agent)
            scores.append(score)
        history = evaluator.get_readiness_history(mock_agent.id)
        assert len(history) == 3
        assert all(s.agent_id == mock_agent.id for s in history)
        assert history[-1] == scores[-1]

    def test_readiness_trends(self, evaluator, mock_agent) -> None:
        """Test trend calculation over time."""
        # Create evaluations over time
        for i in range(5):
            # Modify agent stats slightly
            mock_agent.stats["successful_goals"] = 90 + i
            score = evaluator.evaluate_agent(mock_agent)
            # Simulate time passing
            score.timestamp = datetime.now() - timedelta(hours=5 - i)
            evaluator._evaluation_history[mock_agent.id][-1] = score
        trends = evaluator.get_readiness_trend(mock_agent.id)
        assert "timestamps" in trends
        assert "overall" in trends
        assert len(trends["timestamps"]) == 5
        assert len(trends["overall"]) == 5
        # Check trend direction (should be improving)
        assert trends["goals"][-1] >= trends["goals"][0]

    def test_export_readiness_report(self, evaluator, mock_agent) -> None:
        """Test report export functionality."""
        # Evaluate agent
        evaluator.evaluate_agent(mock_agent)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            report_file = evaluator.export_readiness_report(mock_agent.id, output_path)
            assert report_file.exists()
            assert report_file.suffix == ".json"
            # Load and verify report
            with open(report_file) as f:
                report = json.load(f)
            assert report["agent_id"] == mock_agent.id
            assert "latest_evaluation" in report
            assert "trends" in report
            assert "recommendation_summary" in report

    def test_batch_evaluation(self, evaluator) -> None:
        """Test evaluating multiple agents at once."""
        agents = []
        for i in range(3):
            agent = Mock()
            agent.id = f"agent_{i}"
            agent.knowledge_graph = Mock()
            agent.knowledge_graph.experiences = [Mock() for _ in range(1000 + i * 100)]
            agent.knowledge_graph.patterns = {
                f"p_{j}": Mock(confidence=0.85) for j in range(50 + i * 5)
            }
            agent.stats = {
                "total_goals_attempted": 100,
                "successful_goals": 90 + i,
                "complex_goals_completed": 5 + i,
                "total_interactions": 30,
                "successful_interactions": 25,
                "knowledge_items_shared": 12,
                "collaborators": {f"a{j}" for j in range(4)},
                "energy_efficiency": 0.8,
                "resource_waste_ratio": 0.2,
                "sustainability_score": 0.85,
                "avg_energy_level": 0.7,
            }
            agent.model_update_history = [{"magnitude": 0.008} for _ in range(120)]
            agents.append(agent)
        results = evaluator.batch_evaluate(agents)
        assert len(results) == 3
        assert all(f"agent_{i}" in results for i in range(3))
        assert all(isinstance(score, ReadinessScore) for score in results.values())

    def test_error_handling(self, evaluator) -> None:
        """Test graceful error handling."""
        # Agent with missing attributes
        bad_agent = Mock()
        bad_agent.id = "bad_agent"
        bad_agent.knowledge_graph = None  # Will cause AttributeError
        bad_agent.stats = {}
        bad_agent.model_update_history = []
        score = evaluator.evaluate_agent(bad_agent)
        # Should still return a score, but with 0 values
        assert score.agent_id == "bad_agent"
        assert score.overall_score == 0.0
        assert not score.is_ready

    def test_custom_weights(self) -> None:
        """Test that dimension weights affect overall score correctly."""
        agent = Mock()
        agent.id = "weighted_test"
        # Perfect in one dimension, poor in others
        agent.knowledge_graph = Mock()
        agent.knowledge_graph.experiences = [Mock() for _ in range(2000)]
        agent.knowledge_graph.patterns = {f"p_{i}": Mock(confidence=0.95) for i in range(100)}
        agent.stats = {
            "total_goals_attempted": 20,
            "successful_goals": 10,
            "complex_goals_completed": 1,
            "total_interactions": 5,
            "successful_interactions": 3,
            "knowledge_items_shared": 2,
            "collaborators": {"a1"},
            "energy_efficiency": 0.5,
            "resource_waste_ratio": 0.5,
            "sustainability_score": 0.5,
            "avg_energy_level": 0.5,
        }
        agent.model_update_history = [{"magnitude": 0.1} for _ in range(20)]
        evaluator = AgentReadinessEvaluator()
        score = evaluator.evaluate_agent(agent)
        # Knowledge should be high, others low
        assert score.knowledge_maturity > 0.9
        assert score.goal_achievement < 0.6
        assert score.model_stability < 0.3
        assert score.collaboration < 0.3
        assert score.resource_management < 0.6
        # Overall should be weighted average
        expected = (
            score.knowledge_maturity * 0.25
            + score.goal_achievement * 0.20
            + score.model_stability * 0.20
            + score.collaboration * 0.20
            + score.resource_management * 0.15
        )
        assert abs(score.overall_score - expected) < 0.01
