"""
Module for FreeAgentics Active Inference implementation.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class ReadinessLevel(Enum):
    """Readiness levels for agents"""

    NOT_READY = "not_ready"
    PARTIALLY_READY = "partially_ready"
    READY = "ready"
    FULLY_READY = "fully_ready"


@dataclass
class ReadinessThresholds:
    """Configuration for readiness evaluation thresholds"""

    min_experiences: int = 1000
    min_patterns: int = 50
    pattern_confidence: float = 0.85
    success_rate: float = 0.9
    complex_goals_completed: int = 5
    overall_threshold: float = 0.85


@dataclass
class ReadinessScore:
    """Comprehensive readiness evaluation results"""

    agent_id: str
    timestamp: datetime
    overall_score: float = 0.0
    is_ready: bool = False
    knowledge_maturity: float = 0.0
    goal_achievement: float = 0.0
    model_stability: float = 0.0
    collaboration: float = 0.0
    resource_management: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ReadinessMetrics:
    """Metrics for evaluating agent readiness"""

    overall_score: float
    component_scores: Dict[str, float]
    missing_components: List[str]
    recommendations: List[str]


class ReadinessEvaluator:
    """Evaluates agent readiness for deployment"""

    def __init__(self) -> None:
        self.criteria = {
            "model_trained": 0.3,
            "knowledge_graph": 0.2,
            "configuration": 0.2,
            "dependencies": 0.15,
            "testing": 0.15,
        }

    def evaluate_readiness(self, agent_data: Dict[str, Any]) -> ReadinessMetrics:
        """Evaluate agent readiness based on criteria"""
        component_scores = {}
        missing_components = []

        for component, weight in self.criteria.items():
            if component in agent_data and agent_data[component]:
                component_scores[component] = 1.0
            else:
                component_scores[component] = 0.0
                missing_components.append(component)

        overall_score = sum(
            score * self.criteria[comp] for comp,
            score in component_scores.items())

        recommendations = []
        for missing in missing_components:
            recommendations.append(f"Complete {missing} component")

        return ReadinessMetrics(
            overall_score=overall_score,
            component_scores=component_scores,
            missing_components=missing_components,
            recommendations=recommendations,
        )

    def get_readiness_level(self, metrics: ReadinessMetrics) -> ReadinessLevel:
        """Determine readiness level from metrics"""
        if metrics.overall_score >= 0.9:
            return ReadinessLevel.FULLY_READY
        elif metrics.overall_score >= 0.7:
            return ReadinessLevel.READY
        elif metrics.overall_score >= 0.4:
            return ReadinessLevel.PARTIALLY_READY
        else:
            return ReadinessLevel.NOT_READY


class AgentReadinessEvaluator:
    """Comprehensive agent readiness evaluation system"""

    def __init__(self, thresholds: Optional[ReadinessThresholds] = None) -> None:
        self.thresholds = thresholds or ReadinessThresholds()
        self.dimension_weights = {
            "knowledge": 0.25,
            "goals": 0.20,
            "model_stability": 0.20,
            "collaboration": 0.20,
            "resources": 0.15,
        }
        self._evaluation_history: Dict[str, List[ReadinessScore]] = {}

    def evaluate_agent(self, agent) -> ReadinessScore:
        """Evaluate an agent's readiness across all dimensions"""
        score = ReadinessScore(agent_id=agent.id, timestamp=datetime.now())

        try:
            # Evaluate each dimension
            score.knowledge_maturity = self._evaluate_knowledge(agent, score)
            score.goal_achievement = self._evaluate_goals(agent, score)
            score.model_stability = self._evaluate_model_stability(agent, score)
            score.collaboration = self._evaluate_collaboration(agent, score)
            score.resource_management = self._evaluate_resources(agent, score)

            # Calculate overall score
            score.overall_score = (
                score.knowledge_maturity * self.dimension_weights["knowledge"]
                + score.goal_achievement * self.dimension_weights["goals"]
                + score.model_stability * self.dimension_weights["model_stability"]
                + score.collaboration * self.dimension_weights["collaboration"]
                + score.resource_management * self.dimension_weights["resources"]
            )

            # Determine readiness
            score.is_ready = score.overall_score >= self.thresholds.overall_threshold

            # Generate recommendations
            self._generate_recommendations(score)

        except Exception:
            # Handle errors gracefully
            score.overall_score = 0.0
            score.is_ready = False

        # Store in history
        if agent.id not in self._evaluation_history:
            self._evaluation_history[agent.id] = []
        self._evaluation_history[agent.id].append(score)

        return score

    def _evaluate_knowledge(self, agent, score: ReadinessScore) -> float:
        """Evaluate knowledge maturity"""
        try:
            kg = agent.knowledge_graph
            experience_count = len(
                kg.experiences) if kg and hasattr(
                kg, "experiences") else 0
            patterns = kg.patterns if kg and hasattr(kg, "patterns") else {}
            pattern_count = len(patterns)

            # Calculate average pattern confidence
            if patterns:
                confidences = [
                    p.confidence for p in patterns.values() if hasattr(
                        p, "confidence")]
                avg_confidence = sum(confidences) / \
                    len(confidences) if confidences else 0
            else:
                avg_confidence = 0

            # Store metrics
            score.metrics["knowledge"] = {
                "experience_count": experience_count,
                "pattern_count": pattern_count,
                "avg_pattern_confidence": avg_confidence,
            }

            # Calculate score
            exp_score = min(experience_count / self.thresholds.min_experiences, 1.0)
            pattern_score = min(pattern_count / self.thresholds.min_patterns, 1.0)
            conf_score = (
                avg_confidence /
                self.thresholds.pattern_confidence if avg_confidence > 0 else 0)

            return (exp_score + pattern_score + conf_score) / 3

        except Exception:
            return 0.0

    def _evaluate_goals(self, agent, score: ReadinessScore) -> float:
        """Evaluate goal achievement"""
        try:
            stats = getattr(agent, "stats", {})
            total_attempted = stats.get("total_goals_attempted", 0)
            successful = stats.get("successful_goals", 0)
            complex_completed = stats.get("complex_goals_completed", 0)

            success_rate = successful / total_attempted if total_attempted > 0 else 0

            score.metrics["goals"] = {
                "success_rate": success_rate,
                "complex_completed": complex_completed,
            }

            rate_score = float(success_rate / self.thresholds.success_rate)
            complex_score = float(
                min(complex_completed / self.thresholds.complex_goals_completed, 1.0)
            )

            return float((rate_score + complex_score) / 2)

        except Exception:
            return 0.0

    def _evaluate_model_stability(self, agent, score: ReadinessScore) -> float:
        """Evaluate model stability"""
        try:
            history = getattr(agent, "model_update_history", [])
            if len(history) < 10:
                score.metrics["model_stability"] = {
                    "is_converged": False,
                    "stable_iterations": 0,
                }
                return 0.0

            # Check for convergence (small updates)
            recent_updates = history[-100:] if len(history) >= 100 else history
            stable_count = sum(
                1 for update in recent_updates if update.get("magnitude", 1.0) < 0.01
            )

            is_converged = stable_count >= len(recent_updates) * 0.8

            score.metrics["model_stability"] = {
                "is_converged": is_converged,
                "stable_iterations": stable_count,
            }

            return 1.0 if is_converged else stable_count / len(recent_updates)

        except Exception:
            return 0.0

    def _evaluate_collaboration(self, agent, score: ReadinessScore) -> float:
        """Evaluate collaboration capabilities"""
        try:
            stats = getattr(agent, "stats", {})
            total_interactions = stats.get("total_interactions", 0)
            successful_interactions = stats.get("successful_interactions", 0)
            knowledge_shared = stats.get("knowledge_items_shared", 0)
            collaborators = stats.get("collaborators", set())

            interaction_rate = (successful_interactions /
                                total_interactions if total_interactions > 0 else 0)

            score.metrics["collaboration"] = {
                "successful_interactions": successful_interactions,
                "knowledge_shared": knowledge_shared,
                "unique_collaborators": len(collaborators),
            }

            # Weighted score
            return float(
                interaction_rate * 0.4
                + min(knowledge_shared / 10, 1.0) * 0.3
                + min(len(collaborators) / 5, 1.0) * 0.3
            )

        except Exception:
            return 0.0

    def _evaluate_resources(self, agent, score: ReadinessScore) -> float:
        """Evaluate resource management"""
        try:
            stats = getattr(agent, "stats", {})
            efficiency = stats.get("energy_efficiency", 0)
            sustainability = stats.get("sustainability_score", 0)

            score.metrics["resources"] = {
                "resource_efficiency": efficiency,
                "sustainability_score": sustainability,
            }

            return float((efficiency + sustainability) / 2)

        except Exception:
            return 0.0

    def _generate_recommendations(self, score: ReadinessScore):
        """Generate improvement recommendations"""
        recommendations = []

        if score.knowledge_maturity < 0.8:
            recommendations.append("Increase experience through more interactions")
            recommendations.append("Develop more reliable patterns")

        if score.goal_achievement < 0.8:
            recommendations.append("Improve goal completion rate")
            recommendations.append("Attempt more complex goals")

        if score.model_stability < 0.8:
            recommendations.append(
                "Allow model to converged through continued training")

        if score.collaboration < 0.8:
            recommendations.append("Engage in more collaborative activities")

        if score.resource_management < 0.8:
            recommendations.append("Improve resource efficiency")

        score.recommendations = recommendations

    def get_readiness_history(self, agent_id: str) -> List[ReadinessScore]:
        """Get evaluation history for an agent"""
        return self._evaluation_history.get(agent_id, [])

    def get_readiness_trend(self, agent_id: str) -> Dict[str, List[float]]:
        """Get readiness trends over time"""
        history = self.get_readiness_history(agent_id)
        if not history:
            return {}

        return {
            "timestamps": [score.timestamp.timestamp() for score in history],
            "overall": [score.overall_score for score in history],
            "knowledge": [score.knowledge_maturity for score in history],
            "goals": [score.goal_achievement for score in history],
            "model_stability": [score.model_stability for score in history],
            "collaboration": [score.collaboration for score in history],
            "resources": [score.resource_management for score in history],
        }

    def export_readiness_report(self, agent_id: str, output_dir: Path) -> Path:
        """Export readiness report to file"""
        history = self.get_readiness_history(agent_id)
        trends = self.get_readiness_trend(agent_id)

        latest_eval = history[-1].__dict__ if history else None
        recommendations = history[-1].recommendations if history else []

        report: Dict[str, Any] = {
            "agent_id": agent_id,
            "latest_evaluation": latest_eval,
            "trends": trends,
            "recommendation_summary": recommendations,
        }

        # Convert datetime objects to strings for JSON serialization
        if latest_eval and "timestamp" in latest_eval:
            latest_eval["timestamp"] = latest_eval["timestamp"].isoformat()

        report_file = output_dir / f"readiness_report_{agent_id}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        return report_file

    def batch_evaluate(self, agents: List[Any]) -> Dict[str, ReadinessScore]:
        """Evaluate multiple agents"""
        results = {}
        for agent in agents:
            results[agent.id] = self.evaluate_agent(agent)
        return results


class AgentReadinessChecker:
    """Checks specific agent readiness criteria"""

    def __init__(self) -> None:
        self.evaluator = ReadinessEvaluator()

    def check_agent_readiness(
        self, agent_id: str, agent_config: Dict[str, Any]
    ) -> ReadinessMetrics:
        """Check readiness for a specific agent"""
        return self.evaluator.evaluate_readiness(agent_config)
