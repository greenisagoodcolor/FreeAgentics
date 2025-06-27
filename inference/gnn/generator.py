"""
Module for FreeAgentics Active Inference implementation.
"""

import copy
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .parser import GMNParser
from .validator import GMNValidator

"""
GNN Generator - Enable agents to write and modify their own GNN models.
This module allows agents to generate and refine their cognitive models
based on experience and learning, implementing Daniel Friedman's vision:
"Agents should write their own GNN models."
"""
logger = logging.getLogger(__name__)


class GMNGenerator:
    """Generate and modify GMN models for Active Inference agents"""

    def __init__(self) -> None:
        self.parser = GMNParser()
        self.validator = GMNValidator()
        self.template_cache = {}

    def generate_base_model(
        self, agent_name: str, agent_class: str, personality: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Generate a base GNN model from agent class and personality.
        Args:
            agent_name: Name of the agent
            agent_class: Class type (Explorer, Merchant, Scholar, Guardian)
            personality: Dict with keys: exploration, cooperation, efficiency, curiosity, risk_tolerance
        Returns:
            Complete GNN model dictionary
        """
        template = self._get_class_template(agent_class)
        model = {
            "model": {
                "name": agent_name,
                "type": "ActiveInference",
                "version": "1.0",
                "class": agent_class,
                "created": datetime.now().isoformat(),
            },
            "state_space": self._generate_state_space(agent_class, personality),
            "connections": self._generate_connections(personality),
            "update_equations": self._generate_update_equations(personality),
            "metadata": {
                "personality": personality,
                "learning_history": [],
                "model_version": 1,
            },
        }
        is_valid, errors = self.validator.validate_model(model)
        if not is_valid:
            logger.error(f"Generated invalid model: {errors}")
            raise ValueError(f"Generated model validation failed: {errors}")
        return model

    def refine_model(
        self,
        current_model: Dict[str, Any],
        patterns: List[Dict[str, Any]],
        confidence_threshold: float = 0.8,
    ) -> Dict[str, Any]:
        """
        Refine a GNN model based on learned patterns.
        Args:
            current_model: Current GNN model
            patterns: List of learned patterns with confidence scores
            confidence_threshold: Minimum confidence to incorporate pattern
        Returns:
            Refined GNN model
        """
        refined_model = copy.deepcopy(current_model)
        changes = []
        for pattern in patterns:
            if pattern.get("confidence", 0) >= confidence_threshold:
                change = self._apply_pattern_to_model(refined_model, pattern)
                if change:
                    changes.append(change)
        if "metadata" not in refined_model:
            refined_model["metadata"] = {}
        refined_model["metadata"]["last_refined"] = datetime.now().isoformat()
        refined_model["metadata"]["refinement_changes"] = changes
        refined_model["metadata"]["model_version"] = (
            refined_model["metadata"].get("model_version", 1) + 1
        )
        is_valid, errors = self.validator.validate_model(refined_model)
        if not is_valid:
            logger.warning(f"Refined model validation failed: {errors}. Reverting changes.")
            return current_model
        return refined_model

    def _get_class_template(self, agent_class: str) -> Dict[str, Any]:
        """Get base template for agent class"""
        templates = {
            "Explorer": {
                "focus": "discovery",
                "base_preferences": {
                    "Exploration": 0.7,
                    "Resources": 0.2,
                    "Social": 0.1,
                },
                "key_states": ["S_curiosity", "S_knowledge", "S_position"],
            },
            "Merchant": {
                "focus": "trading",
                "base_preferences": {
                    "Exploration": 0.2,
                    "Resources": 0.6,
                    "Social": 0.2,
                },
                "key_states": ["S_inventory", "S_reputation", "S_wealth"],
            },
            "Scholar": {
                "focus": "learning",
                "base_preferences": {
                    "Exploration": 0.3,
                    "Resources": 0.1,
                    "Social": 0.6,
                },
                "key_states": ["S_knowledge", "S_theories", "S_connections"],
            },
            "Guardian": {
                "focus": "protection",
                "base_preferences": {
                    "Exploration": 0.1,
                    "Resources": 0.4,
                    "Social": 0.5,
                },
                "key_states": ["S_territory", "S_alertness", "S_allies"],
            },
        }
        return templates.get(agent_class, templates["Explorer"])

    def _generate_state_space(
        self, agent_class: str, personality: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate state space based on class and personality"""
        state_space = {
            "S_energy": {"type": "Real[0, 100]", "description": "Agent energy level"},
            "S_position": {
                "type": "H3Cell[resolution=7]",
                "description": "Current hex position",
            },
            "S_beliefs": {
                "type": "Distribution[State]",
                "description": "Beliefs about world state",
            },
            "A_actions": {
                "type": "Categorical",
                "options": [
                    "move_north",
                    "move_south",
                    "move_east",
                    "move_west",
                    "explore",
                    "exploit",
                    "communicate",
                    "rest",
                ],
                "description": "Available actions",
            },
        }
        template = self._get_class_template(agent_class)
        for state in template["key_states"]:
            if state not in state_space:
                if "knowledge" in state:
                    state_space[state] = {
                        "type": "Real[0, 100]",
                        "description": "Accumulated knowledge",
                    }
                elif "position" in state:
                    pass
                elif "inventory" in state:
                    state_space[state] = {
                        "type": "List[Resource]",
                        "description": "Carried resources",
                    }
                elif "wealth" in state:
                    state_space[state] = {
                        "type": "Real[0, 1000]",
                        "description": "Accumulated wealth",
                    }
                elif "territory" in state:
                    state_space[state] = {
                        "type": "Set[H3Cell]",
                        "description": "Protected territory",
                    }
                elif "alertness" in state:
                    state_space[state] = {
                        "type": "Real[0, 1]",
                        "description": "Threat awareness level",
                    }
                else:
                    state_space[state] = {
                        "type": "Real[0, 100]",
                        "description": f"Level of {state.replace('S_', '')}",
                    }
        if personality.get("curiosity", 0) > 0.7:
            state_space["S_novelty_seeking"] = {
                "type": "Real[0, 1]",
                "description": "Drive to find new experiences",
            }
        if personality.get("risk_tolerance", 0) > 0.7:
            state_space["S_risk_assessment"] = {
                "type": "Real[0, 1]",
                "description": "Current risk evaluation",
            }
        return state_space

    def _generate_connections(self, personality: Dict[str, float]) -> Dict[str, Any]:
        """Generate connections based on personality"""
        exploration_weight = personality.get("exploration", 0.5) / 100
        cooperation_weight = personality.get("cooperation", 0.5) / 100
        efficiency_weight = personality.get("efficiency", 0.5) / 100
        total = exploration_weight + cooperation_weight + efficiency_weight
        if total > 0:
            exploration_weight /= total
            cooperation_weight /= total
            efficiency_weight /= total
        else:
            exploration_weight = cooperation_weight = efficiency_weight = 1 / 3
        connections = {
            "C_pref": {
                "type": "observation -> Real[0, 1]",
                "description": "Preference function mapping observations to utilities",
                "preferences": {
                    "Exploration": round(exploration_weight, 2),
                    "Resources": round(efficiency_weight, 2),
                    "Social": round(cooperation_weight, 2),
                },
            },
            "C_likelihood": {
                "type": "state x observation -> Real[0, 1]",
                "description": "Likelihood mapping P(o|s)",
            },
        }
        if personality.get("curiosity", 0) > 0.7:
            connections["C_novelty"] = {
                "type": "observation -> Real[0, 1]",
                "description": "Novelty detection function",
            }
        return connections

    def _generate_update_equations(self, personality: Dict[str, float]) -> Dict[str, Any]:
        """Generate update equations based on personality"""
        base_learning_rate = 0.1 + personality.get("curiosity", 0.5) / 100 * 0.1
        equations = {
            "belief_update": {
                "state": "S_beliefs",
                "formula": "S_beliefs(t+1) = S_beliefs(t) + learning_rate * prediction_error",
                "parameters": {"learning_rate": round(base_learning_rate, 3)},
            },
            "energy_dynamics": {
                "state": "S_energy",
                "formula": "S_energy(t+1) = S_energy(t) - action_cost + rest_recovery",
                "parameters": {
                    "action_cost": {
                        "explore": 2.0,
                        "exploit": 1.5,
                        "communicate": 1.0,
                        "rest": -3.0,
                        "move": 1.0,
                    }
                },
            },
        }
        if personality.get("efficiency", 0) > 0.7:
            equations["efficiency_optimization"] = {
                "state": "action_selection",
                "formula": "minimize(action_cost / expected_reward)",
                "description": "Optimize actions for efficiency",
            }
        if personality.get("cooperation", 0) > 0.7:
            equations["social_learning"] = {
                "state": "S_beliefs",
                "formula": "S_beliefs(t+1) = weighted_average(S_beliefs(t), shared_beliefs)",
                "parameters": {"social_weight": 0.3},
            }
        return equations

    def _apply_pattern_to_model(
        self, model: Dict[str, Any], pattern: Dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Apply a learned pattern to the model"""
        change_record = {
            "timestamp": datetime.now().isoformat(),
            "pattern_type": pattern.get("type", "unknown"),
            "confidence": pattern.get("confidence", 0),
            "changes": [],
        }
        pattern_type = pattern.get("type", "")
        if pattern_type == "successful_action_sequence":
            if "C_pref" in model["connections"]:
                action = pattern.get("dominant_action", "")
                if action and action not in model["connections"]["C_pref"].get("action_biases", {}):
                    if "action_biases" not in model["connections"]["C_pref"]:
                        model["connections"]["C_pref"]["action_biases"] = {}
                    bias = pattern.get("success_rate", 0.5) * pattern.get("confidence", 0.8)
                    model["connections"]["C_pref"]["action_biases"][action] = round(bias, 3)
                    change_record["changes"].append(
                        {
                            "type": "add_action_bias",
                            "action": action,
                            "bias": round(bias, 3),
                        }
                    )
        elif pattern_type == "environmental_correlation":
            if "C_likelihood" in model["connections"]:
                correlation = pattern.get("correlation", {})
                if correlation:
                    if "correlations" not in model["connections"]["C_likelihood"]:
                        model["connections"]["C_likelihood"]["correlations"] = []
                    model["connections"]["C_likelihood"]["correlations"].append(
                        {
                            "observation": correlation.get("observation", ""),
                            "state": correlation.get("state", ""),
                            "strength": round(correlation.get("strength", 0.5), 3),
                        }
                    )
                    change_record["changes"].append(
                        {"type": "add_correlation", "details": correlation}
                    )
        elif pattern_type == "preference_adjustment":
            if "C_pref" in model["connections"] and "preferences" in model["connections"]["C_pref"]:
                adjustments = pattern.get("adjustments", {})
                for pref, adjustment in adjustments.items():
                    if pref in model["connections"]["C_pref"]["preferences"]:
                        old_value = model["connections"]["C_pref"]["preferences"][pref]
                        new_value = max(0, min(1, old_value + adjustment))
                        model["connections"]["C_pref"]["preferences"][pref] = round(new_value, 3)
                        change_record["changes"].append(
                            {
                                "type": "adjust_preference",
                                "preference": pref,
                                "old_value": old_value,
                                "new_value": round(new_value, 3),
                            }
                        )
        return change_record if change_record["changes"] else None

    def generate_from_experience(
        self, base_model: Dict[str, Any], experience_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a new model version based on accumulated experience.
        Args:
            base_model: Current model
            experience_summary: Summary of agent's experiences
        Returns:
            New model incorporating learned behaviors
        """
        new_model = copy.deepcopy(base_model)
        patterns = self._extract_patterns_from_experience(experience_summary)
        refined_model = self.refine_model(new_model, patterns)
        if experience_summary.get("unique_observations", 0) > 50:
            if "S_world_model" not in refined_model["state_space"]:
                refined_model["state_space"]["S_world_model"] = {
                    "type": "Graph[Location, Connection]",
                    "description": "Learned model of world structure",
                }
        return refined_model

    def _extract_patterns_from_experience(
        self, experience_summary: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract learnable patterns from experience summary"""
        patterns = []
        action_stats = experience_summary.get("action_statistics", {})
        for action, stats in action_stats.items():
            if stats.get("count", 0) > 10:
                success_rate = stats.get("success_count", 0) / stats["count"]
                if success_rate > 0.7:
                    patterns.append(
                        {
                            "type": "successful_action_sequence",
                            "dominant_action": action,
                            "success_rate": success_rate,
                            "confidence": min(0.95, 0.5 + stats["count"] / 100),
                        }
                    )
        correlations = experience_summary.get("observed_correlations", [])
        for corr in correlations:
            if corr.get("occurrences", 0) > 5 and corr.get("correlation", 0) > 0.6:
                patterns.append(
                    {
                        "type": "environmental_correlation",
                        "correlation": corr,
                        "confidence": min(0.9, corr["correlation"]),
                    }
                )
        return patterns

    def export_to_gnn_format(self, model: Dict[str, Any], file_path: str):
        """
        Export model to .gnn.md format.
        Args:
            model: GNN model dictionary
            file_path: Path to save the .gnn.md file
        """
        lines = []
        lines.append(f"# GNN Model: {model['model']['name']}")
        lines.append(f"Class: {model['model'].get('class', 'Unknown')}")
        lines.append(f"Version: {model['metadata'].get('model_version', 1)}")
        lines.append("")
        lines.append("## Model")
        lines.append(f"Name: {model['model']['name']}")
        lines.append(f"Type: {model['model']['type']}")
        lines.append("")
        lines.append("## State Space")
        for state_name, state_def in model["state_space"].items():
            lines.append(f"{state_name}: {state_def['type']}")
            if "description" in state_def:
                lines.append(f"  - {state_def['description']}")
        lines.append("")
        lines.append("## Connections")
        for conn_name, conn_def in model["connections"].items():
            lines.append(f"{conn_name}: {conn_def['type']}")
            if "preferences" in conn_def:
                for pref, weight in conn_def["preferences"].items():
                    lines.append(f"  - {pref}: {weight}")
        lines.append("")
        lines.append("## Update Equations")
        for eq_name, equation in model["update_equations"].items():
            lines.append(f"{eq_name}:")
            lines.append(f"  state: {equation.get('state', 'unknown')}")
            lines.append(f"  formula: {equation.get('formula', 'unknown')}")
            if "parameters" in equation:
                lines.append("  parameters:")
                for param, value in equation["parameters"].items():
                    lines.append(f"    {param}: {value}")
        with open(file_path, "w") as f:
            f.write("\n".join(lines))
        logger.info(f"Exported GNN model to {file_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generator = GMNGenerator()
    personality = {
        "exploration": 80,
        "cooperation": 60,
        "efficiency": 40,
        "curiosity": 90,
        "risk_tolerance": 70,
    }
    model = generator.generate_base_model(
        agent_name="Explorer-Alpha", agent_class="Explorer", personality=personality
    )
    print("Generated GNN Model:")
    print(json.dumps(model, indent=2))
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
    refined = generator.refine_model(model, patterns)
    print("\nRefined model changes:", refined["metadata"].get("refinement_changes", []))
    generator.export_to_gnn_format(refined, "example_agent.gnn.md")
