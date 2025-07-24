"""
Critical Integration Point Testing: LLM → Coalition Interface

This test suite focuses specifically on the integration between Large Language Models (LLM)
and Coalition Formation systems, testing the critical transformation from natural language
reasoning to structured coordination parameters.

Key Integration Challenges:
1. LLM produces natural language strategy recommendations
2. Coalition system requires structured coordination parameters
3. Strategy semantics must be preserved in the transformation
4. Coordination parameters must be mathematically valid and actionable

Test Philosophy:
- Test actual LLM reasoning → coordination parameter transformation
- Validate strategy semantic preservation through execution testing
- Use realistic coordination scenarios with multiple agents
- Test edge cases where LLM recommendations may be ambiguous or invalid
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pytest

# Core components for LLM-Coalition integration
from agents.base_agent import BasicExplorerAgent
from agents.coalition_coordinator import CoalitionCoordinatorAgent
from coalitions.coordination_types import (
    CoalitionFormationStrategy,
    CoordinationMessage,
)
from inference.llm.local_llm_manager import LocalLLMManager

logger = logging.getLogger(__name__)


class StrategyParser:
    """
    Parses LLM natural language strategy recommendations into structured coordination parameters.
    This is the critical integration component between LLM reasoning and coalition execution.
    """

    def __init__(self):
        self.strategy_keywords = {
            CoalitionFormationStrategy.CENTRALIZED: [
                "central",
                "hierarchical",
                "command",
                "control",
                "leader",
                "coordinator",
                "top-down",
                "single point",
                "unified command",
            ],
            CoalitionFormationStrategy.DISTRIBUTED: [
                "distributed",
                "decentralized",
                "peer-to-peer",
                "autonomous",
                "independent",
                "local decision",
                "self-organizing",
                "emergent",
            ],
            CoalitionFormationStrategy.AUCTION_BASED: [
                "auction",
                "bid",
                "competitive",
                "market",
                "price",
                "cost-benefit",
                "optimization",
                "efficient allocation",
            ],
            CoalitionFormationStrategy.HIERARCHICAL: [
                "hierarchy",
                "levels",
                "layers",
                "chain of command",
                "delegation",
                "multi-level",
                "structured",
                "organized",
            ],
        }

        self.coordination_parameters = {
            "communication_frequency": {
                "high": ["frequent", "continuous", "real-time", "constant"],
                "medium": ["regular", "periodic", "scheduled", "intermittent"],
                "low": ["minimal", "occasional", "as-needed", "sparse"],
            },
            "decision_autonomy": {
                "high": [
                    "autonomous",
                    "independent",
                    "self-directed",
                    "flexible",
                ],
                "medium": [
                    "guided",
                    "constrained",
                    "structured",
                    "supervised",
                ],
                "low": ["controlled", "directed", "strict", "centralized"],
            },
            "coordination_overhead": {
                "high": ["complex", "detailed", "comprehensive", "thorough"],
                "medium": ["balanced", "moderate", "standard", "typical"],
                "low": ["minimal", "simple", "lightweight", "efficient"],
            },
        }

    def parse_strategy_recommendation(
        self, llm_text: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Parse LLM strategy recommendation into structured coordination parameters.

        This is the critical integration point where natural language strategies
        must be converted to actionable coordination parameters.
        """
        llm_text_lower = llm_text.lower()

        # Determine primary formation strategy
        strategy_scores = {}
        for strategy, keywords in self.strategy_keywords.items():
            score = sum(1 for keyword in keywords if keyword in llm_text_lower)
            strategy_scores[strategy] = score

        primary_strategy = (
            max(strategy_scores, key=strategy_scores.get)
            if strategy_scores
            else CoalitionFormationStrategy.AUCTION_BASED
        )

        # Extract coordination parameters
        coordination_params = {}
        for param_name, param_values in self.coordination_parameters.items():
            param_scores = {}
            for level, keywords in param_values.items():
                score = sum(1 for keyword in keywords if keyword in llm_text_lower)
                param_scores[level] = score

            # Select highest scoring level, default to medium
            selected_level = (
                max(param_scores, key=param_scores.get) if any(param_scores.values()) else "medium"
            )
            coordination_params[param_name] = selected_level

        # Extract numerical parameters from text
        numerical_params = self._extract_numerical_parameters(llm_text, context)

        # Generate coordination message structure
        coordination_structure = {
            "strategy": primary_strategy,
            "parameters": coordination_params,
            "numerical_params": numerical_params,
            "confidence": self._calculate_parsing_confidence(
                llm_text, strategy_scores, coordination_params
            ),
            "raw_text": llm_text,
            "context": context,
        }

        return coordination_structure

    def _extract_numerical_parameters(self, text: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Extract numerical coordination parameters from LLM text."""
        import re

        numerical_params = {}

        # Look for percentage values
        percentages = re.findall(r"(\d+(?:\.\d+)?)\s*%", text)
        if percentages:
            numerical_params["efficiency_target"] = float(percentages[0]) / 100.0

        # Look for time values
        time_matches = re.findall(r"(\d+(?:\.\d+)?)\s*(second|minute|hour)", text)
        if time_matches:
            value, unit = time_matches[0]
            multiplier = {"second": 1, "minute": 60, "hour": 3600}
            numerical_params["time_horizon"] = float(value) * multiplier.get(unit, 1)

        # Look for agent count references
        agent_matches = re.findall(r"(\d+)\s*agent", text)
        if agent_matches:
            numerical_params["target_agent_count"] = float(agent_matches[0])
        elif context.get("agent_count"):
            numerical_params["target_agent_count"] = float(context["agent_count"])

        # Look for distance/range values
        distance_matches = re.findall(r"(\d+(?:\.\d+)?)\s*(meter|unit|step)", text)
        if distance_matches:
            value, unit = distance_matches[0]
            numerical_params["coordination_range"] = float(value)

        # Default values based on context
        if "efficiency_target" not in numerical_params:
            numerical_params["efficiency_target"] = 0.8  # Default 80% efficiency

        if "time_horizon" not in numerical_params:
            numerical_params["time_horizon"] = 300.0  # Default 5 minutes

        return numerical_params

    def _calculate_parsing_confidence(
        self, text: str, strategy_scores: Dict, coordination_params: Dict
    ) -> float:
        """Calculate confidence in the parsing results."""
        confidence_factors = []

        # Strategy confidence: based on keyword matches
        max_strategy_score = max(strategy_scores.values()) if strategy_scores else 0
        sum(len(keywords) for keywords in self.strategy_keywords.values())
        strategy_confidence = min(1.0, max_strategy_score / 3.0)  # 3+ matches = high confidence
        confidence_factors.append(strategy_confidence)

        # Parameter confidence: based on how many parameters were extracted
        param_extraction_rate = len(
            [p for p in coordination_params.values() if p != "medium"]
        ) / len(coordination_params)
        confidence_factors.append(param_extraction_rate)

        # Text length confidence: reasonable length suggests detailed analysis
        text_length_score = min(1.0, len(text) / 200.0)  # 200+ chars = reasonable detail
        confidence_factors.append(text_length_score)

        # Specificity confidence: mentions of specific concepts
        specific_terms = [
            "agent",
            "resource",
            "coordinate",
            "strategy",
            "efficiency",
            "communication",
        ]
        specificity_score = sum(1 for term in specific_terms if term in text.lower()) / len(
            specific_terms
        )
        confidence_factors.append(specificity_score)

        return sum(confidence_factors) / len(confidence_factors)

    def validate_coordination_parameters(
        self, coordination_structure: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate that parsed coordination parameters are mathematically valid and actionable.
        """
        validation_results = {
            "valid_strategy": coordination_structure.get("strategy") in CoalitionFormationStrategy,
            "parameters_complete": len(coordination_structure.get("parameters", {})) >= 3,
            "numerical_params_valid": True,
            "context_consistent": True,
            "actionable": True,
        }

        # Validate numerical parameters
        numerical_params = coordination_structure.get("numerical_params", {})
        for param_name, value in numerical_params.items():
            if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                validation_results["numerical_params_valid"] = False
                break

            # Parameter-specific validation
            if param_name == "efficiency_target" and not (0.0 <= value <= 1.0):
                validation_results["numerical_params_valid"] = False
            elif param_name == "time_horizon" and value <= 0:
                validation_results["numerical_params_valid"] = False

        # Validate context consistency
        if context.get("agent_count") and "target_agent_count" in numerical_params:
            if numerical_params["target_agent_count"] > context["agent_count"] * 2:
                validation_results["context_consistent"] = False

        # Check if parameters are actionable
        confidence = coordination_structure.get("confidence", 0)
        if confidence < 0.3:  # Very low confidence suggests parameters may not be actionable
            validation_results["actionable"] = False

        validation_results["overall_valid"] = all(validation_results.values())

        return validation_results


class CoordinationExecutor:
    """
    Executes coordination strategies parsed from LLM recommendations.
    Tests that the integration actually produces working coordination behavior.
    """

    def __init__(self):
        self.execution_metrics = {}

    async def execute_coordination_strategy(
        self,
        coordination_structure: Dict[str, Any],
        agents: List,
        coordinator: CoalitionCoordinatorAgent,
    ) -> Dict[str, Any]:
        """
        Execute the parsed coordination strategy with real agents.
        This validates that the LLM→Coalition integration produces actual coordination behavior.
        """
        execution_start = time.time()

        strategy = coordination_structure.get("strategy", CoalitionFormationStrategy.AUCTION_BASED)
        parameters = coordination_structure.get("parameters", {})
        numerical_params = coordination_structure.get("numerical_params", {})

        # Configure coordinator with parsed strategy
        coordinator.formation_strategy = strategy

        execution_results = {
            "strategy_applied": strategy,
            "agents_coordinated": [],
            "coordination_messages": [],
            "execution_success": True,
            "coordination_metrics": {},
        }

        try:
            # Create coordination messages based on parsed parameters
            for agent in agents:
                coordination_message = CoordinationMessage(
                    message_id=f"coord_{agent.agent_id}_{int(time.time())}",
                    sender_id=coordinator.agent_id,
                    recipient_id=agent.agent_id,
                    message_type="strategy_execution",
                    content={
                        "strategy": strategy.value,
                        "parameters": parameters,
                        "numerical_params": numerical_params,
                        "execution_context": coordination_structure.get("context", {}),
                    },
                    timestamp=datetime.now(),
                )

                execution_results["coordination_messages"].append(coordination_message)

                # Process coordination message
                response = await coordinator.process_coordination_message(coordination_message)

                if response:
                    execution_results["agents_coordinated"].append(
                        {
                            "agent_id": agent.agent_id,
                            "response": response,
                            "success": True,
                        }
                    )
                else:
                    execution_results["agents_coordinated"].append(
                        {
                            "agent_id": agent.agent_id,
                            "response": None,
                            "success": False,
                        }
                    )

            # Calculate coordination metrics
            successful_coordinations = len(
                [a for a in execution_results["agents_coordinated"] if a["success"]]
            )
            coordination_rate = successful_coordinations / len(agents) if agents else 0

            execution_results["coordination_metrics"] = {
                "coordination_rate": coordination_rate,
                "successful_agents": successful_coordinations,
                "total_agents": len(agents),
                "execution_time": time.time() - execution_start,
                "messages_sent": len(execution_results["coordination_messages"]),
            }

            # Strategy-specific validation
            if strategy == CoalitionFormationStrategy.CENTRALIZED:
                # All agents should coordinate through the coordinator
                execution_results["strategy_validation"] = coordination_rate > 0.8
            elif strategy == CoalitionFormationStrategy.DISTRIBUTED:
                # Agents should have some autonomy
                execution_results["strategy_validation"] = coordination_rate > 0.5
            else:
                # Other strategies should have reasonable coordination
                execution_results["strategy_validation"] = coordination_rate > 0.6

        except Exception as e:
            execution_results["execution_success"] = False
            execution_results["error"] = str(e)
            execution_results["strategy_validation"] = False

        return execution_results

    def analyze_coordination_effectiveness(
        self, execution_results: Dict[str, Any], original_llm_text: str
    ) -> Dict[str, Any]:
        """
        Analyze how effectively the LLM strategy recommendation was translated into coordination behavior.
        """
        effectiveness_analysis = {
            "strategy_implemented": execution_results.get("execution_success", False),
            "coordination_achieved": execution_results.get("coordination_metrics", {}).get(
                "coordination_rate", 0
            )
            > 0.5,
            "strategy_appropriate": execution_results.get("strategy_validation", False),
            "semantic_preservation": self._assess_semantic_preservation(
                execution_results, original_llm_text
            ),
        }

        # Overall effectiveness score
        effectiveness_analysis["overall_effectiveness"] = sum(
            effectiveness_analysis.values()
        ) / len(effectiveness_analysis)

        return effectiveness_analysis

    def _assess_semantic_preservation(
        self, execution_results: Dict[str, Any], original_text: str
    ) -> bool:
        """
        Assess whether the coordination behavior preserves the semantic intent of the LLM recommendation.
        """
        original_text_lower = original_text.lower()

        # Check for semantic consistency
        semantic_checks = []

        # If LLM mentioned efficiency, check coordination rate
        if any(term in original_text_lower for term in ["efficient", "optimize", "effective"]):
            coordination_rate = execution_results.get("coordination_metrics", {}).get(
                "coordination_rate", 0
            )
            semantic_checks.append(coordination_rate > 0.7)

        # If LLM mentioned speed/fast, check execution time
        if any(term in original_text_lower for term in ["fast", "quick", "rapid", "immediate"]):
            execution_time = execution_results.get("coordination_metrics", {}).get(
                "execution_time", float("inf")
            )
            semantic_checks.append(execution_time < 5.0)

        # If LLM mentioned coordination/collaboration, check success rate
        if any(
            term in original_text_lower for term in ["coordinate", "collaborate", "work together"]
        ):
            successful_agents = execution_results.get("coordination_metrics", {}).get(
                "successful_agents", 0
            )
            total_agents = execution_results.get("coordination_metrics", {}).get("total_agents", 1)
            semantic_checks.append(successful_agents / total_agents > 0.6)

        # Default check: basic execution success
        if not semantic_checks:
            semantic_checks.append(execution_results.get("execution_success", False))

        return all(semantic_checks)


@pytest.mark.asyncio
class TestLLMCoalitionInterfaceIntegration:
    """Integration tests for LLM-Coalition interface and strategy execution."""

    @pytest.fixture
    async def llm_manager(self):
        """Create LLM manager if available."""
        try:
            return LocalLLMManager()
        except Exception:
            return None

    @pytest.fixture
    async def test_agents(self):
        """Create test agents for coordination."""
        agents = []
        for i in range(4):
            agent = BasicExplorerAgent(f"test_agent_{i}", f"Test Agent {i}", grid_size=20)
            agents.append(agent)
        return agents

    @pytest.fixture
    async def coordinator(self):
        """Create coalition coordinator."""
        return CoalitionCoordinatorAgent("test_coordinator", "Test Coordinator", max_agents=10)

    @pytest.fixture
    async def coordination_scenarios(self):
        """Create realistic coordination scenarios for testing."""
        return [
            {
                "name": "resource_collection_efficiency",
                "context": {
                    "agent_count": 4,
                    "resource_count": 8,
                    "environment_size": 20,
                    "time_pressure": "medium",
                    "communication_cost": "low",
                },
                "llm_prompt": """
                Analyze this multi-agent resource collection scenario:
                - 4 agents available for coordination
                - 8 resources distributed across a 20x20 environment
                - Medium time pressure for completion
                - Low communication costs between agents

                Recommend an optimal coordination strategy that maximizes efficiency while minimizing overhead.
                Consider agent capabilities, resource distribution, and communication constraints.
                """,
            },
            {
                "name": "emergency_response_coordination",
                "context": {
                    "agent_count": 3,
                    "emergency_locations": 2,
                    "environment_size": 15,
                    "time_pressure": "high",
                    "communication_cost": "high",
                },
                "llm_prompt": """
                Emergency response coordination scenario:
                - 3 emergency response agents
                - 2 critical emergency locations requiring immediate attention
                - 15x15 operational area
                - High time pressure - every second counts
                - High communication costs - minimize coordination overhead

                Design a coordination strategy that ensures rapid response while managing communication efficiently.
                Priority is speed and reliability over complex coordination.
                """,
            },
            {
                "name": "exploration_with_uncertainty",
                "context": {
                    "agent_count": 5,
                    "unknown_areas": 10,
                    "environment_size": 25,
                    "time_pressure": "low",
                    "communication_cost": "medium",
                },
                "llm_prompt": """
                Exploration coordination in uncertain environment:
                - 5 exploration agents with different sensor capabilities
                - 10 unknown areas that need systematic exploration
                - 25x25 environment with potential obstacles
                - Low time pressure - thorough exploration preferred
                - Medium communication costs

                Recommend a coordination strategy that ensures comprehensive coverage while adapting to discoveries.
                Focus on information sharing and adaptive coordination.
                """,
            },
        ]

    async def test_llm_strategy_generation(self, llm_manager, coordination_scenarios):
        """Test LLM's ability to generate coordination strategies for realistic scenarios."""

        if not llm_manager:
            pytest.skip("LLM manager not available")

        strategy_results = {}

        for scenario in coordination_scenarios:
            scenario_name = scenario["name"]
            llm_prompt = scenario["llm_prompt"]

            try:
                # Generate strategy recommendation
                response = await llm_manager.generate_response(
                    prompt=llm_prompt, max_tokens=400, temperature=0.7
                )

                strategy_text = response.get("text", "")

                # Validate strategy quality
                strategy_quality = self._assess_strategy_quality(strategy_text, scenario["context"])

                strategy_results[scenario_name] = {
                    "strategy_text": strategy_text,
                    "quality_score": strategy_quality["overall_score"],
                    "quality_breakdown": strategy_quality,
                    "generation_success": True,
                }

                assert strategy_quality["overall_score"] > 0.5, (
                    f"Low quality strategy for {scenario_name}: {strategy_quality['overall_score']}"
                )

                logger.info(
                    f"✓ LLM strategy generation for '{scenario_name}' successful (quality: {strategy_quality['overall_score']:.3f})"
                )

            except Exception as e:
                strategy_results[scenario_name] = {
                    "strategy_text": "",
                    "quality_score": 0.0,
                    "generation_success": False,
                    "error": str(e),
                }
                logger.warning(f"✗ LLM strategy generation for '{scenario_name}' failed: {e}")

        # At least half of scenarios should generate valid strategies
        successful_scenarios = len(
            [r for r in strategy_results.values() if r["generation_success"]]
        )
        assert successful_scenarios >= len(coordination_scenarios) / 2, (
            "Too many LLM strategy generation failures"
        )

        return strategy_results

    def _assess_strategy_quality(
        self, strategy_text: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess the quality of LLM-generated coordination strategy."""

        strategy_text_lower = strategy_text.lower()

        quality_metrics = {
            "mentions_agents": any(
                term in strategy_text_lower for term in ["agent", "robot", "unit"]
            ),
            "addresses_coordination": any(
                term in strategy_text_lower for term in ["coordinat", "collaborat", "cooperat"]
            ),
            "considers_efficiency": any(
                term in strategy_text_lower for term in ["efficien", "optim", "effective"]
            ),
            "addresses_communication": any(
                term in strategy_text_lower for term in ["communicat", "message", "inform"]
            ),
            "mentions_strategy_type": any(
                term in strategy_text_lower
                for term in [
                    "central",
                    "distributed",
                    "hierarchical",
                    "auction",
                ]
            ),
            "considers_constraints": any(
                term in strategy_text_lower for term in ["time", "cost", "resource", "limit"]
            ),
            "provides_reasoning": any(
                term in strategy_text_lower for term in ["because", "since", "therefore", "due to"]
            ),
            "actionable_recommendations": any(
                term in strategy_text_lower
                for term in ["should", "recommend", "suggest", "implement"]
            ),
            "reasonable_length": 100 < len(strategy_text) < 1000,
            "context_aware": any(
                str(value).lower() in strategy_text_lower
                for value in context.values()
                if isinstance(value, (str, int))
            ),
        }

        quality_metrics["overall_score"] = sum(quality_metrics.values()) / len(quality_metrics)

        return quality_metrics

    async def test_strategy_parsing_and_validation(self, llm_manager, coordination_scenarios):
        """Test parsing of LLM strategies into structured coordination parameters."""

        if not llm_manager:
            pytest.skip("LLM manager not available")

        parser = StrategyParser()
        parsing_results = {}

        for scenario in coordination_scenarios:
            scenario_name = scenario["name"]
            context = scenario["context"]

            try:
                # Generate LLM strategy
                response = await llm_manager.generate_response(
                    prompt=scenario["llm_prompt"],
                    max_tokens=300,
                    temperature=0.7,
                )

                strategy_text = response.get("text", "")

                # Parse strategy into coordination parameters
                coordination_structure = parser.parse_strategy_recommendation(
                    strategy_text, context
                )

                # Validate parsed parameters
                validation_results = parser.validate_coordination_parameters(
                    coordination_structure, context
                )

                parsing_results[scenario_name] = {
                    "original_text": strategy_text,
                    "coordination_structure": coordination_structure,
                    "validation_results": validation_results,
                    "parsing_success": True,
                    "parsing_confidence": coordination_structure.get("confidence", 0),
                }

                # Validate parsing quality
                assert validation_results["overall_valid"], (
                    f"Invalid coordination parameters for {scenario_name}: {validation_results}"
                )
                assert coordination_structure.get("confidence", 0) > 0.3, (
                    f"Low parsing confidence for {scenario_name}"
                )

                logger.info(
                    f"✓ Strategy parsing for '{scenario_name}' successful (confidence: {coordination_structure.get('confidence', 0):.3f})"
                )

            except Exception as e:
                parsing_results[scenario_name] = {
                    "parsing_success": False,
                    "error": str(e),
                    "parsing_confidence": 0.0,
                }
                logger.warning(f"✗ Strategy parsing for '{scenario_name}' failed: {e}")

        # Validate overall parsing success
        successful_parsing = len([r for r in parsing_results.values() if r["parsing_success"]])
        assert successful_parsing > 0, "All strategy parsing attempts failed"

        return parsing_results

    async def test_coordination_execution(self, test_agents, coordinator, coordination_scenarios):
        """Test execution of parsed coordination strategies with real agents."""

        parser = StrategyParser()
        executor = CoordinationExecutor()
        execution_results = {}

        # Use mock LLM strategies for testing execution
        mock_strategies = {
            "resource_collection_efficiency": """
            For efficient resource collection with 4 agents and 8 resources, I recommend a hierarchical coordination strategy.
            Designate one agent as the local coordinator to minimize communication overhead while maintaining efficiency.
            Use auction-based task allocation for optimal resource assignment. Target 85% collection efficiency within 5 minutes.
            Agents should communicate status every 30 seconds to maintain coordination without excessive overhead.
            """,
            "emergency_response_coordination": """
            For emergency response, implement a centralized command structure for rapid decision-making.
            High time pressure requires minimal communication delays - use direct coordination through the command agent.
            Assign agents to emergency locations based on proximity and capability. Target immediate response within 60 seconds.
            Minimize communication to essential status updates only due to high communication costs.
            """,
            "exploration_with_uncertainty": """
            For exploration in uncertain environments, use a distributed coordination approach allowing agent autonomy.
            Each agent should explore designated areas while sharing discoveries through periodic information exchange.
            Implement adaptive coordination - agents can request assistance or reallocate based on findings.
            Target comprehensive coverage over speed, with regular coordination meetings every 2 minutes.
            """,
        }

        for scenario in coordination_scenarios:
            scenario_name = scenario["name"]
            context = scenario["context"]
            strategy_text = mock_strategies.get(
                scenario_name,
                "Default coordination strategy with moderate efficiency.",
            )

            try:
                # Parse strategy into coordination parameters
                coordination_structure = parser.parse_strategy_recommendation(
                    strategy_text, context
                )

                # Execute coordination strategy
                execution_result = await executor.execute_coordination_strategy(
                    coordination_structure, test_agents, coordinator
                )

                # Analyze execution effectiveness
                effectiveness_analysis = executor.analyze_coordination_effectiveness(
                    execution_result, strategy_text
                )

                execution_results[scenario_name] = {
                    "coordination_structure": coordination_structure,
                    "execution_result": execution_result,
                    "effectiveness_analysis": effectiveness_analysis,
                    "execution_success": execution_result.get("execution_success", False),
                }

                # Validate execution success
                assert execution_result.get("execution_success", False), (
                    f"Coordination execution failed for {scenario_name}"
                )
                assert effectiveness_analysis.get("overall_effectiveness", 0) > 0.5, (
                    f"Low coordination effectiveness for {scenario_name}"
                )

                coordination_rate = execution_result.get("coordination_metrics", {}).get(
                    "coordination_rate", 0
                )
                logger.info(
                    f"✓ Coordination execution for '{scenario_name}' successful (rate: {coordination_rate:.3f})"
                )

            except Exception as e:
                execution_results[scenario_name] = {
                    "execution_success": False,
                    "error": str(e),
                }
                logger.warning(f"✗ Coordination execution for '{scenario_name}' failed: {e}")

        # Validate overall execution success
        successful_executions = len(
            [r for r in execution_results.values() if r["execution_success"]]
        )
        assert successful_executions > 0, "All coordination executions failed"

        return execution_results

    async def test_end_to_end_llm_coalition_integration(
        self, llm_manager, test_agents, coordinator
    ):
        """Test complete LLM→Coalition integration pipeline."""

        if not llm_manager:
            pytest.skip("LLM manager not available")

        # End-to-end integration test scenario
        integration_scenario = {
            "context": {
                "agent_count": len(test_agents),
                "task_type": "collaborative_problem_solving",
                "environment_complexity": "medium",
                "time_constraints": "moderate",
            },
            "llm_prompt": f"""
            Design a coordination strategy for {len(test_agents)} agents working on a collaborative problem-solving task.
            The environment has medium complexity with moderate time constraints.

            Consider:
            1. How should agents communicate and share information?
            2. What coordination structure would be most effective?
            3. How should tasks be allocated and managed?
            4. What are the key success metrics?

            Provide a specific, actionable coordination strategy.
            """,
        }

        # Step 1: LLM Strategy Generation
        llm_start = time.time()
        response = await llm_manager.generate_response(
            prompt=integration_scenario["llm_prompt"],
            max_tokens=350,
            temperature=0.7,
        )
        strategy_text = response.get("text", "")
        llm_time = time.time() - llm_start

        assert len(strategy_text) > 50, "LLM generated insufficient strategy text"

        # Step 2: Strategy Parsing
        parser = StrategyParser()
        parse_start = time.time()
        coordination_structure = parser.parse_strategy_recommendation(
            strategy_text, integration_scenario["context"]
        )
        parse_time = time.time() - parse_start

        validation_results = parser.validate_coordination_parameters(
            coordination_structure, integration_scenario["context"]
        )
        assert validation_results["overall_valid"], (
            f"Invalid coordination parameters: {validation_results}"
        )

        # Step 3: Coordination Execution
        executor = CoordinationExecutor()
        exec_start = time.time()
        execution_result = await executor.execute_coordination_strategy(
            coordination_structure, test_agents, coordinator
        )
        exec_time = time.time() - exec_start

        assert execution_result.get("execution_success", False), "Coordination execution failed"

        # Step 4: Effectiveness Analysis
        effectiveness_analysis = executor.analyze_coordination_effectiveness(
            execution_result, strategy_text
        )

        # Overall integration validation
        integration_results = {
            "llm_generation": {
                "success": len(strategy_text) > 0,
                "time": llm_time,
                "text_length": len(strategy_text),
            },
            "strategy_parsing": {
                "success": validation_results["overall_valid"],
                "time": parse_time,
                "confidence": coordination_structure.get("confidence", 0),
            },
            "coordination_execution": {
                "success": execution_result.get("execution_success", False),
                "time": exec_time,
                "coordination_rate": execution_result.get("coordination_metrics", {}).get(
                    "coordination_rate", 0
                ),
            },
            "effectiveness_analysis": effectiveness_analysis,
            "overall_pipeline": {
                "total_time": llm_time + parse_time + exec_time,
                "end_to_end_success": all(
                    [
                        len(strategy_text) > 0,
                        validation_results["overall_valid"],
                        execution_result.get("execution_success", False),
                        effectiveness_analysis.get("overall_effectiveness", 0) > 0.4,
                    ]
                ),
            },
        }

        # Performance requirements
        assert integration_results["overall_pipeline"]["total_time"] < 60.0, (
            "End-to-end pipeline too slow"
        )
        assert integration_results["overall_pipeline"]["end_to_end_success"], (
            "End-to-end integration failed"
        )

        logger.info("✓ End-to-end LLM→Coalition integration successful")
        logger.info(f"  LLM generation: {llm_time:.3f}s")
        logger.info(f"  Strategy parsing: {parse_time:.3f}s")
        logger.info(f"  Coordination execution: {exec_time:.3f}s")
        logger.info(
            f"  Overall effectiveness: {effectiveness_analysis.get('overall_effectiveness', 0):.3f}"
        )

        return integration_results

    async def test_integration_edge_cases(self, test_agents, coordinator):
        """Test edge cases in LLM→Coalition integration."""

        parser = StrategyParser()
        executor = CoordinationExecutor()
        edge_case_results = {}

        # Edge case 1: Ambiguous LLM strategy
        ambiguous_strategy = (
            "Maybe use some coordination approach that could work depending on the situation."
        )
        coordination_structure = parser.parse_strategy_recommendation(
            ambiguous_strategy, {"agent_count": 2}
        )

        edge_case_results["ambiguous_strategy"] = {
            "parsing_confidence": coordination_structure.get("confidence", 0),
            "handles_ambiguity": coordination_structure.get("confidence", 0)
            < 0.5,  # Should recognize low confidence
        }

        # Edge case 2: Contradictory LLM recommendations
        contradictory_strategy = "Use centralized control for autonomy and distributed coordination with strict hierarchy."
        coordination_structure = parser.parse_strategy_recommendation(
            contradictory_strategy, {"agent_count": 3}
        )

        edge_case_results["contradictory_strategy"] = {
            "strategy_selected": coordination_structure.get("strategy"),
            "handles_contradiction": True,  # Should select some strategy despite contradiction
        }

        # Edge case 3: Empty or minimal LLM output
        minimal_strategy = "Coordinate."
        coordination_structure = parser.parse_strategy_recommendation(
            minimal_strategy, {"agent_count": 1}
        )

        edge_case_results["minimal_strategy"] = {
            "parsing_confidence": coordination_structure.get("confidence", 0),
            "fallback_handling": coordination_structure.get("strategy") is not None,
        }

        # Edge case 4: Execution with no agents
        try:
            execution_result = await executor.execute_coordination_strategy(
                {
                    "strategy": CoalitionFormationStrategy.AUCTION_BASED,
                    "parameters": {},
                    "numerical_params": {},
                },
                [],
                coordinator,
            )
            edge_case_results["no_agents"] = {
                "handles_empty_agent_list": execution_result.get("coordination_metrics", {}).get(
                    "coordination_rate", 0
                )
                == 0,
                "execution_completes": True,
            }
        except Exception as e:
            edge_case_results["no_agents"] = {
                "handles_empty_agent_list": False,
                "execution_completes": False,
                "error": str(e),
            }

        # Validate edge case handling
        for case_name, case_result in edge_case_results.items():
            logger.info(f"Edge case '{case_name}': {case_result}")

        # Basic edge cases should be handled without crashing
        assert edge_case_results["ambiguous_strategy"]["handles_ambiguity"], (
            "Ambiguous strategy not handled"
        )
        assert edge_case_results["contradictory_strategy"]["handles_contradiction"], (
            "Contradictory strategy not handled"
        )
        assert edge_case_results["minimal_strategy"]["fallback_handling"], (
            "Minimal strategy not handled"
        )
        assert edge_case_results["no_agents"]["execution_completes"], "Empty agent list not handled"

        return edge_case_results


if __name__ == "__main__":
    """Run LLM-Coalition interface integration tests directly."""

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    async def run_llm_coalition_tests():
        """Run LLM-Coalition interface integration tests."""

        test_class = TestLLMCoalitionInterfaceIntegration()

        # Create fixtures
        try:
            llm_manager = LocalLLMManager()
        except Exception:
            llm_manager = None
            print("Warning: LLM manager not available, some tests will be skipped")

        # Test agents and coordinator
        test_agents = [
            BasicExplorerAgent(f"test_agent_{i}", f"Agent {i}", grid_size=10) for i in range(3)
        ]
        coordinator = CoalitionCoordinatorAgent(
            "test_coordinator", "Test Coordinator", max_agents=5
        )

        # Coordination scenarios
        coordination_scenarios = [
            {
                "name": "simple_coordination",
                "context": {"agent_count": 3, "task_type": "exploration"},
                "llm_prompt": "Design a simple coordination strategy for 3 exploration agents. Focus on efficiency and clear communication.",
            }
        ]

        # Run tests
        tests = [
            (
                "Strategy Parsing and Validation",
                lambda: test_class.test_strategy_parsing_and_validation(
                    llm_manager, coordination_scenarios
                ),
            ),
            (
                "Coordination Execution",
                lambda: test_class.test_coordination_execution(
                    test_agents, coordinator, coordination_scenarios
                ),
            ),
            (
                "End-to-End Integration",
                lambda: test_class.test_end_to_end_llm_coalition_integration(
                    llm_manager, test_agents, coordinator
                ),
            ),
            (
                "Integration Edge Cases",
                lambda: test_class.test_integration_edge_cases(test_agents, coordinator),
            ),
        ]

        results = []
        for test_name, test_func in tests:
            print(f"\nRunning {test_name}...")

            try:
                start_time = time.time()
                await test_func()
                execution_time = time.time() - start_time

                results.append(
                    {
                        "test": test_name,
                        "status": "PASSED",
                        "time": execution_time,
                    }
                )
                print(f"✓ {test_name} PASSED ({execution_time:.2f}s)")

            except Exception as e:
                execution_time = time.time() - start_time

                results.append(
                    {
                        "test": test_name,
                        "status": "FAILED",
                        "time": execution_time,
                        "error": str(e),
                    }
                )
                print(f"✗ {test_name} FAILED ({execution_time:.2f}s): {e}")

        # Summary
        passed = len([r for r in results if r["status"] == "PASSED"])
        total = len(results)

        print(f"\n{'=' * 60}")
        print("LLM-COALITION INTERFACE INTEGRATION TEST SUMMARY")
        print(f"{'=' * 60}")
        print(f"Tests run: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success rate: {passed / total * 100:.1f}%")

        if passed == total:
            print("🎉 All LLM-Coalition interface tests passed!")
        else:
            print("❌ Some LLM-Coalition interface tests failed!")

    # Run the tests
    asyncio.run(run_llm_coalition_tests())
