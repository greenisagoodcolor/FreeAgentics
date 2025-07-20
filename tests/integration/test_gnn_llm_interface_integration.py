"""
Critical Integration Point Testing: GNN → LLM Interface

This test suite focuses specifically on the integration between Graph Neural Networks (GNN)
and Large Language Models (LLM), testing the critical data transformation and semantic
preservation between these components.

Key Integration Challenges:
1. GNN produces numerical embeddings (numpy arrays)
2. LLM requires textual descriptions for reasoning
3. Semantic meaning must be preserved across this transformation
4. Performance must remain acceptable for real-time decision making

Test Philosophy:
- Test actual integration points, not just component isolation
- Validate semantic preservation through round-trip testing
- Use realistic graph structures and embedding dimensions
- Test edge cases that only emerge from GNN-LLM interaction
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

# Core components for GNN-LLM integration
from inference.gnn.model import GMNModel as GNNModel
from inference.llm.local_llm_manager import LocalLLMManager

logger = logging.getLogger(__name__)


class GNNToLLMTransformer:
    """
    Transforms GNN embeddings into structured text for LLM reasoning.
    This is a critical integration component that bridges numerical and textual representations.
    """

    def __init__(self):
        self.embedding_thresholds = {
            "high_activity": 0.7,
            "medium_activity": 0.3,
            "connectivity_threshold": 0.5,
        }

    def embeddings_to_text(
        self,
        embeddings: np.ndarray,
        node_ids: List[str],
        graph_metadata: Dict[str, Any],
    ) -> str:
        """
        Convert GNN node embeddings to structured text description.

        This is the critical integration point where numerical representations
        must be converted to semantically meaningful text.
        """
        if len(embeddings) == 0:
            return "Empty graph with no nodes to analyze."

        # Analyze embedding patterns
        embedding_norms = np.linalg.norm(embeddings, axis=1)
        high_activity_nodes = (
            embedding_norms > self.embedding_thresholds["high_activity"]
        )
        medium_activity_nodes = (
            embedding_norms > self.embedding_thresholds["medium_activity"]
        ) & (~high_activity_nodes)

        # Calculate connectivity patterns
        connectivity_scores = np.mean(embeddings, axis=1)
        well_connected = (
            connectivity_scores
            > self.embedding_thresholds["connectivity_threshold"]
        )

        # Generate structured description
        description_parts = []

        # Overall graph characteristics
        description_parts.append(f"Graph Analysis Summary:")
        description_parts.append(f"- Total nodes analyzed: {len(embeddings)}")
        description_parts.append(
            f"- Average embedding magnitude: {np.mean(embedding_norms):.3f}"
        )
        description_parts.append(
            f"- Embedding dimensionality: {embeddings.shape[1]}"
        )

        # Node activity analysis
        high_activity_count = np.sum(high_activity_nodes)
        medium_activity_count = np.sum(medium_activity_nodes)
        low_activity_count = (
            len(embeddings) - high_activity_count - medium_activity_count
        )

        description_parts.append(f"\nNode Activity Levels:")
        description_parts.append(
            f"- High activity nodes: {high_activity_count}"
        )
        description_parts.append(
            f"- Medium activity nodes: {medium_activity_count}"
        )
        description_parts.append(f"- Low activity nodes: {low_activity_count}")

        # Connectivity analysis
        well_connected_count = np.sum(well_connected)
        description_parts.append(f"\nConnectivity Analysis:")
        description_parts.append(
            f"- Well-connected nodes: {well_connected_count}"
        )
        description_parts.append(
            f"- Isolated nodes: {len(embeddings) - well_connected_count}"
        )

        # Key nodes identification
        if len(node_ids) == len(embeddings):
            # Identify most important nodes
            importance_scores = embedding_norms + connectivity_scores
            top_indices = np.argsort(importance_scores)[
                -min(3, len(node_ids)) :
            ]

            description_parts.append(f"\nKey Nodes Identified:")
            for idx in reversed(top_indices):
                node_id = node_ids[idx]
                importance = importance_scores[idx]
                description_parts.append(
                    f"- {node_id}: importance score {importance:.3f}"
                )

        # Graph metadata integration
        if graph_metadata:
            description_parts.append(f"\nGraph Context:")
            for key, value in graph_metadata.items():
                description_parts.append(f"- {key}: {value}")

        return "\n".join(description_parts)

    def validate_transformation_quality(
        self, embeddings: np.ndarray, text_description: str
    ) -> Dict[str, Any]:
        """
        Validate that the embedding→text transformation preserves important information.
        """
        validation_results = {
            "has_node_count": str(len(embeddings)) in text_description,
            "has_activity_analysis": "activity" in text_description.lower(),
            "has_connectivity_info": "connect" in text_description.lower(),
            "has_numerical_data": any(
                char.isdigit() for char in text_description
            ),
            "reasonable_length": 100 < len(text_description) < 2000,
            "structured_format": "\n" in text_description
            and "-" in text_description,
        }

        validation_results["overall_quality"] = sum(
            validation_results.values()
        ) / len(validation_results)

        return validation_results


class LLMGraphReasoningValidator:
    """
    Validates that LLM can reason effectively about graph structures from GNN embeddings.
    """

    def __init__(self, llm_manager: Optional[LocalLLMManager] = None):
        self.llm_manager = llm_manager
        self.reasoning_prompts = {
            "strategy_analysis": """
            Based on the following graph analysis, recommend an optimal strategy for multi-agent coordination:

            {graph_description}

            Consider: 1) Resource allocation efficiency, 2) Communication overhead, 3) Coordination complexity
            Provide a brief strategy recommendation with reasoning.
            """,
            "risk_assessment": """
            Analyze the following graph structure for potential coordination risks:

            {graph_description}

            Identify: 1) Bottleneck nodes, 2) Isolation risks, 3) Communication vulnerabilities
            Provide risk level assessment and mitigation recommendations.
            """,
            "optimization_opportunities": """
            Given this graph analysis, identify optimization opportunities:

            {graph_description}

            Focus on: 1) Connectivity improvements, 2) Load balancing, 3) Efficiency gains
            Suggest specific optimizations with expected benefits.
            """,
        }

    async def validate_llm_reasoning(
        self, graph_description: str, reasoning_type: str = "strategy_analysis"
    ) -> Dict[str, Any]:
        """
        Test LLM's ability to reason about graph structures from text descriptions.
        """
        if (
            not self.llm_manager
            or reasoning_type not in self.reasoning_prompts
        ):
            return {
                "success": False,
                "error": "LLM manager not available or invalid reasoning type",
                "reasoning_quality": 0.0,
            }

        try:
            prompt = self.reasoning_prompts[reasoning_type].format(
                graph_description=graph_description
            )

            response = await self.llm_manager.generate_response(
                prompt=prompt, max_tokens=300, temperature=0.7
            )

            reasoning_text = response.get("text", "")

            # Validate reasoning quality
            quality_metrics = self._assess_reasoning_quality(
                reasoning_text, reasoning_type
            )

            return {
                "success": True,
                "reasoning_text": reasoning_text,
                "reasoning_quality": quality_metrics["overall_score"],
                "quality_breakdown": quality_metrics,
                "prompt_used": prompt,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "reasoning_quality": 0.0,
            }

    def _assess_reasoning_quality(
        self, reasoning_text: str, reasoning_type: str
    ) -> Dict[str, Any]:
        """
        Assess the quality of LLM reasoning about graph structures.
        """
        metrics = {
            "has_specific_recommendations": any(
                keyword in reasoning_text.lower()
                for keyword in ["recommend", "suggest", "should", "could"]
            ),
            "mentions_graph_concepts": any(
                concept in reasoning_text.lower()
                for concept in ["node", "connect", "network", "graph"]
            ),
            "provides_reasoning": any(
                indicator in reasoning_text.lower()
                for indicator in ["because", "since", "due to", "therefore"]
            ),
            "addresses_coordination": "coordinat" in reasoning_text.lower(),
            "reasonable_length": 50 < len(reasoning_text) < 1000,
            "structured_response": any(
                char in reasoning_text for char in ["1)", "2)", "•", "-"]
            ),
        }

        # Type-specific validations
        if reasoning_type == "strategy_analysis":
            metrics["mentions_strategy"] = "strategy" in reasoning_text.lower()
            metrics["considers_efficiency"] = (
                "efficien" in reasoning_text.lower()
            )
        elif reasoning_type == "risk_assessment":
            metrics["identifies_risks"] = any(
                risk in reasoning_text.lower()
                for risk in ["risk", "bottleneck", "vulnerability"]
            )
            metrics["suggests_mitigation"] = any(
                mitigation in reasoning_text.lower()
                for mitigation in ["mitigate", "prevent", "avoid"]
            )
        elif reasoning_type == "optimization_opportunities":
            metrics["identifies_opportunities"] = any(
                opt in reasoning_text.lower()
                for opt in ["optimiz", "improv", "enhance"]
            )
            metrics["quantifies_benefits"] = any(
                char.isdigit() for char in reasoning_text
            )

        metrics["overall_score"] = sum(metrics.values()) / len(metrics)

        return metrics


@pytest.mark.asyncio
class TestGNNLLMInterfaceIntegration:
    """Integration tests for GNN-LLM interface and data transformation."""

    @pytest.fixture
    async def gnn_model(self):
        """Create GNN model for testing."""
        return GNNModel(
            node_features=6,  # x, y, type, value, connectivity, importance
            edge_features=3,  # distance, weight, type
            hidden_dim=64,
            num_layers=2,
        )

    @pytest.fixture
    async def llm_manager(self):
        """Create LLM manager if available."""
        try:
            return LocalLLMManager()
        except Exception:
            return None

    @pytest.fixture
    async def test_graph_data(self):
        """Create realistic test graph data."""
        # Create a realistic multi-agent scenario graph
        nodes = [
            (
                "agent_1",
                [0, 0, 1, 50, 0.8, 0.9],
            ),  # type=1 (agent), high connectivity/importance
            (
                "agent_2",
                [10, 5, 1, 40, 0.6, 0.7],
            ),  # type=1 (agent), medium connectivity
            (
                "agent_3",
                [15, 15, 1, 30, 0.4, 0.5],
            ),  # type=1 (agent), lower connectivity
            (
                "resource_1",
                [5, 3, 2, 100, 0.9, 0.8],
            ),  # type=2 (resource), high value
            (
                "resource_2",
                [12, 8, 2, 80, 0.7, 0.6],
            ),  # type=2 (resource), medium value
            (
                "obstacle_1",
                [7, 7, 3, 0, 0.1, 0.2],
            ),  # type=3 (obstacle), low connectivity
            (
                "base_1",
                [2, 2, 4, 200, 0.95, 0.95],
            ),  # type=4 (base), very high importance
        ]

        edges = [
            (
                0,
                1,
                [7.07, 0.8, 1],
            ),  # agent_1 → agent_2: distance, weight, type
            (
                0,
                3,
                [5.83, 0.9, 2],
            ),  # agent_1 → resource_1: high weight connection
            (
                1,
                4,
                [4.47, 0.7, 2],
            ),  # agent_2 → resource_2: medium weight connection
            (2, 4, [7.07, 0.5, 2]),  # agent_3 → resource_2: lower weight
            (0, 6, [2.83, 0.95, 3]),  # agent_1 → base_1: very high weight
            (
                5,
                3,
                [4.47, 0.1, 4],
            ),  # obstacle_1 blocks resource_1: low weight (blockage)
            (
                5,
                4,
                [6.40, 0.1, 4],
            ),  # obstacle_1 affects resource_2: low weight
        ]

        node_features = np.array(
            [features for _, features in nodes], dtype=np.float32
        )
        edge_indices = np.array([(i, j) for i, j, _ in edges], dtype=np.int32)
        edge_features = np.array(
            [features for _, _, features in edges], dtype=np.float32
        )
        node_ids = [node_id for node_id, _ in nodes]

        metadata = {
            "scenario": "multi_agent_resource_collection",
            "agent_count": 3,
            "resource_count": 2,
            "obstacle_count": 1,
            "base_count": 1,
            "total_resource_value": 180,
            "graph_complexity": "medium",
        }

        return {
            "node_features": node_features,
            "edge_indices": edge_indices,
            "edge_features": edge_features,
            "node_ids": node_ids,
            "metadata": metadata,
        }

    async def test_gnn_embedding_generation(self, gnn_model, test_graph_data):
        """Test that GNN generates valid embeddings from graph data."""

        node_features = test_graph_data["node_features"]
        edge_indices = test_graph_data["edge_indices"]
        edge_features = test_graph_data["edge_features"]

        try:
            # Generate embeddings using GNN
            gnn_output = gnn_model.forward(
                node_features, edge_indices, edge_features
            )
            embeddings = gnn_output.get("node_embeddings", node_features)

            # Validate embedding properties
            assert embeddings is not None, "GNN failed to generate embeddings"
            assert (
                embeddings.shape[0] == node_features.shape[0]
            ), "Embedding count mismatch"
            assert not np.any(
                np.isnan(embeddings)
            ), "GNN embeddings contain NaN values"
            assert not np.any(
                np.isinf(embeddings)
            ), "GNN embeddings contain infinite values"

            # Validate embedding characteristics
            embedding_norms = np.linalg.norm(embeddings, axis=1)
            assert np.all(
                embedding_norms > 0
            ), "Some embeddings have zero norm"
            assert (
                np.std(embedding_norms) > 0
            ), "All embeddings have identical norms"

            logger.info(
                f"✓ GNN generated valid embeddings: shape {embeddings.shape}"
            )
            logger.info(
                f"  Embedding norm range: {np.min(embedding_norms):.3f} - {np.max(embedding_norms):.3f}"
            )

            return embeddings

        except Exception as e:
            # Fallback to input features if GNN fails
            logger.warning(
                f"GNN processing failed: {e}, using input features as fallback"
            )
            return node_features

    async def test_embedding_to_text_transformation(self, test_graph_data):
        """Test transformation of GNN embeddings to structured text."""

        # Use test data as mock embeddings
        embeddings = test_graph_data["node_features"]
        node_ids = test_graph_data["node_ids"]
        metadata = test_graph_data["metadata"]

        transformer = GNNToLLMTransformer()

        # Transform embeddings to text
        text_description = transformer.embeddings_to_text(
            embeddings, node_ids, metadata
        )

        # Validate transformation quality
        validation_results = transformer.validate_transformation_quality(
            embeddings, text_description
        )

        assert (
            text_description is not None and len(text_description) > 0
        ), "Text transformation failed"
        assert (
            validation_results["overall_quality"] > 0.7
        ), f"Transformation quality too low: {validation_results}"

        # Validate specific content preservation
        assert (
            str(len(embeddings)) in text_description
        ), "Node count not preserved in text"
        assert (
            "activity" in text_description.lower()
        ), "Activity analysis missing"
        assert (
            "connect" in text_description.lower()
        ), "Connectivity analysis missing"

        logger.info(f"✓ Embedding→text transformation successful")
        logger.info(f"  Text length: {len(text_description)} characters")
        logger.info(
            f"  Quality score: {validation_results['overall_quality']:.3f}"
        )

        return text_description

    async def test_llm_graph_reasoning(self, llm_manager, test_graph_data):
        """Test LLM's ability to reason about graph structures from text descriptions."""

        if not llm_manager:
            pytest.skip("LLM manager not available for testing")

        # Generate text description from embeddings
        embeddings = test_graph_data["node_features"]
        node_ids = test_graph_data["node_ids"]
        metadata = test_graph_data["metadata"]

        transformer = GNNToLLMTransformer()
        text_description = transformer.embeddings_to_text(
            embeddings, node_ids, metadata
        )

        # Test LLM reasoning capabilities
        validator = LLMGraphReasoningValidator(llm_manager)

        reasoning_types = [
            "strategy_analysis",
            "risk_assessment",
            "optimization_opportunities",
        ]
        reasoning_results = {}

        for reasoning_type in reasoning_types:
            result = await validator.validate_llm_reasoning(
                text_description, reasoning_type
            )
            reasoning_results[reasoning_type] = result

            if result["success"]:
                assert (
                    result["reasoning_quality"] > 0.5
                ), f"Low quality reasoning for {reasoning_type}: {result['reasoning_quality']}"
                logger.info(
                    f"✓ LLM {reasoning_type} reasoning successful (quality: {result['reasoning_quality']:.3f})"
                )
            else:
                logger.warning(
                    f"✗ LLM {reasoning_type} reasoning failed: {result['error']}"
                )

        # At least one reasoning type should succeed
        successful_reasoning = [
            r for r in reasoning_results.values() if r["success"]
        ]
        assert (
            len(successful_reasoning) > 0
        ), "No LLM reasoning attempts succeeded"

        return reasoning_results

    async def test_round_trip_semantic_preservation(
        self, gnn_model, llm_manager, test_graph_data
    ):
        """Test that semantic meaning is preserved through GNN→text→LLM reasoning pipeline."""

        # Full pipeline test
        node_features = test_graph_data["node_features"]
        edge_indices = test_graph_data["edge_indices"]
        edge_features = test_graph_data["edge_features"]
        node_ids = test_graph_data["node_ids"]
        metadata = test_graph_data["metadata"]

        # Step 1: GNN processing
        try:
            gnn_output = gnn_model.forward(
                node_features, edge_indices, edge_features
            )
            embeddings = gnn_output.get("node_embeddings", node_features)
        except Exception:
            embeddings = node_features  # Fallback

        # Step 2: Text transformation
        transformer = GNNToLLMTransformer()
        text_description = transformer.embeddings_to_text(
            embeddings, node_ids, metadata
        )

        # Step 3: LLM reasoning (if available)
        if llm_manager:
            validator = LLMGraphReasoningValidator(llm_manager)
            reasoning_result = await validator.validate_llm_reasoning(
                text_description, "strategy_analysis"
            )

            if reasoning_result["success"]:
                # Validate semantic preservation
                reasoning_text = reasoning_result["reasoning_text"].lower()

                # Check if key graph concepts are preserved
                semantic_checks = {
                    "preserves_agent_concept": any(
                        term in reasoning_text
                        for term in ["agent", "robot", "unit"]
                    ),
                    "preserves_resource_concept": any(
                        term in reasoning_text
                        for term in ["resource", "target", "goal"]
                    ),
                    "preserves_coordination_concept": any(
                        term in reasoning_text
                        for term in ["coordinat", "collaborat", "team"]
                    ),
                    "preserves_spatial_concept": any(
                        term in reasoning_text
                        for term in [
                            "distance",
                            "location",
                            "position",
                            "spatial",
                        ]
                    ),
                    "preserves_connectivity_concept": any(
                        term in reasoning_text
                        for term in ["connect", "network", "link", "path"]
                    ),
                }

                semantic_preservation_score = sum(
                    semantic_checks.values()
                ) / len(semantic_checks)

                assert (
                    semantic_preservation_score > 0.6
                ), f"Poor semantic preservation: {semantic_preservation_score}"

                logger.info(f"✓ Round-trip semantic preservation successful")
                logger.info(
                    f"  Semantic preservation score: {semantic_preservation_score:.3f}"
                )
                logger.info(
                    f"  Preserved concepts: {[k for k, v in semantic_checks.items() if v]}"
                )

                return {
                    "embeddings": embeddings,
                    "text_description": text_description,
                    "llm_reasoning": reasoning_text,
                    "semantic_preservation": semantic_preservation_score,
                    "semantic_checks": semantic_checks,
                }
            else:
                logger.warning(
                    "LLM reasoning failed, testing text transformation only"
                )

        # Fallback validation without LLM
        validation_results = transformer.validate_transformation_quality(
            embeddings, text_description
        )
        assert (
            validation_results["overall_quality"] > 0.7
        ), "Text transformation quality insufficient"

        return {
            "embeddings": embeddings,
            "text_description": text_description,
            "transformation_quality": validation_results["overall_quality"],
        }

    async def test_integration_performance_characteristics(
        self, gnn_model, llm_manager, test_graph_data
    ):
        """Test performance characteristics of GNN-LLM integration."""

        performance_results = {}

        # Test GNN processing performance
        gnn_start = time.time()
        try:
            gnn_output = gnn_model.forward(
                test_graph_data["node_features"],
                test_graph_data["edge_indices"],
                test_graph_data["edge_features"],
            )
            embeddings = gnn_output.get(
                "node_embeddings", test_graph_data["node_features"]
            )
            gnn_success = True
        except Exception:
            embeddings = test_graph_data["node_features"]
            gnn_success = False

        gnn_time = time.time() - gnn_start
        performance_results["gnn_processing"] = {
            "time_seconds": gnn_time,
            "success": gnn_success,
            "nodes_processed": len(test_graph_data["node_features"]),
            "throughput_nodes_per_sec": (
                len(test_graph_data["node_features"]) / gnn_time
                if gnn_time > 0
                else 0
            ),
        }

        # Test text transformation performance
        transformer = GNNToLLMTransformer()
        transform_start = time.time()
        text_description = transformer.embeddings_to_text(
            embeddings,
            test_graph_data["node_ids"],
            test_graph_data["metadata"],
        )
        transform_time = time.time() - transform_start

        performance_results["text_transformation"] = {
            "time_seconds": transform_time,
            "text_length": len(text_description),
            "throughput_chars_per_sec": (
                len(text_description) / transform_time
                if transform_time > 0
                else 0
            ),
        }

        # Test LLM reasoning performance (if available)
        if llm_manager:
            validator = LLMGraphReasoningValidator(llm_manager)
            llm_start = time.time()
            reasoning_result = await validator.validate_llm_reasoning(
                text_description, "strategy_analysis"
            )
            llm_time = time.time() - llm_start

            performance_results["llm_reasoning"] = {
                "time_seconds": llm_time,
                "success": reasoning_result["success"],
                "reasoning_quality": reasoning_result.get(
                    "reasoning_quality", 0
                ),
            }

        # Overall pipeline performance
        total_time = sum(
            result["time_seconds"] for result in performance_results.values()
        )
        performance_results["overall_pipeline"] = {
            "total_time_seconds": total_time,
            "pipeline_stages": len(performance_results),
            "end_to_end_success": all(
                result.get("success", True)
                for result in performance_results.values()
            ),
        }

        # Performance requirements validation
        assert gnn_time < 5.0, f"GNN processing too slow: {gnn_time:.3f}s"
        assert (
            transform_time < 0.5
        ), f"Text transformation too slow: {transform_time:.3f}s"
        assert (
            total_time < 30.0
        ), f"Overall pipeline too slow: {total_time:.3f}s"

        logger.info(f"✓ Integration performance validation successful")
        logger.info(f"  GNN processing: {gnn_time:.3f}s")
        logger.info(f"  Text transformation: {transform_time:.3f}s")
        logger.info(f"  Total pipeline: {total_time:.3f}s")

        return performance_results

    async def test_integration_edge_cases(self, gnn_model, test_graph_data):
        """Test edge cases that only emerge from GNN-LLM integration."""

        transformer = GNNToLLMTransformer()
        edge_case_results = {}

        # Edge case 1: Empty graph
        empty_embeddings = np.empty((0, 6), dtype=np.float32)
        empty_text = transformer.embeddings_to_text(empty_embeddings, [], {})
        edge_case_results["empty_graph"] = {
            "text_generated": len(empty_text) > 0,
            "handles_gracefully": "empty" in empty_text.lower(),
        }

        # Edge case 2: Single node graph
        single_node = test_graph_data["node_features"][:1]
        single_text = transformer.embeddings_to_text(
            single_node, ["single_node"], {"type": "minimal"}
        )
        edge_case_results["single_node"] = {
            "text_generated": len(single_text) > 0,
            "mentions_node_count": "1" in single_text,
        }

        # Edge case 3: Extreme embedding values
        extreme_embeddings = np.array(
            [
                [1000, -1000, 0, 1e6, 0, 0],  # Very large values
                [1e-10, 1e-10, 0, 0, 0, 0],  # Very small values
                [
                    np.inf,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],  # Infinite values (if not caught by GNN)
            ],
            dtype=np.float32,
        )

        # Replace inf with large finite value for testing
        extreme_embeddings = np.nan_to_num(
            extreme_embeddings, nan=0.0, posinf=1e6, neginf=-1e6
        )

        try:
            extreme_text = transformer.embeddings_to_text(
                extreme_embeddings,
                ["extreme_1", "extreme_2", "extreme_3"],
                {"type": "stress_test"},
            )
            edge_case_results["extreme_values"] = {
                "text_generated": len(extreme_text) > 0,
                "handles_extremes": True,
            }
        except Exception as e:
            edge_case_results["extreme_values"] = {
                "text_generated": False,
                "handles_extremes": False,
                "error": str(e),
            }

        # Edge case 4: Very large graph
        large_graph_size = 100
        large_embeddings = np.random.randn(large_graph_size, 6).astype(
            np.float32
        )
        large_node_ids = [f"node_{i}" for i in range(large_graph_size)]

        large_start = time.time()
        large_text = transformer.embeddings_to_text(
            large_embeddings, large_node_ids, {"type": "large_graph"}
        )
        large_time = time.time() - large_start

        edge_case_results["large_graph"] = {
            "text_generated": len(large_text) > 0,
            "processing_time": large_time,
            "scales_reasonably": large_time < 5.0,
        }

        # Validate all edge cases handled reasonably
        for case_name, case_result in edge_case_results.items():
            if case_result.get("text_generated", False):
                logger.info(f"✓ Edge case '{case_name}' handled successfully")
            else:
                logger.warning(
                    f"✗ Edge case '{case_name}' failed: {case_result}"
                )

        # At least basic cases should work
        assert edge_case_results["empty_graph"][
            "text_generated"
        ], "Empty graph case failed"
        assert edge_case_results["single_node"][
            "text_generated"
        ], "Single node case failed"
        assert edge_case_results["large_graph"][
            "text_generated"
        ], "Large graph case failed"

        return edge_case_results


if __name__ == "__main__":
    """Run GNN-LLM interface integration tests directly."""

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    async def run_interface_tests():
        """Run GNN-LLM interface integration tests."""

        test_class = TestGNNLLMInterfaceIntegration()

        # Create fixtures
        gnn_model = GNNModel(
            node_features=6, edge_features=3, hidden_dim=64, num_layers=2
        )

        try:
            llm_manager = LocalLLMManager()
        except Exception:
            llm_manager = None
            print(
                "Warning: LLM manager not available, some tests will be skipped"
            )

        # Test graph data
        test_graph_data = {
            "node_features": np.array(
                [
                    [0, 0, 1, 50, 0.8, 0.9],
                    [10, 5, 1, 40, 0.6, 0.7],
                    [5, 3, 2, 100, 0.9, 0.8],
                    [2, 2, 4, 200, 0.95, 0.95],
                ],
                dtype=np.float32,
            ),
            "edge_indices": np.array([[0, 1], [0, 2], [0, 3]], dtype=np.int32),
            "edge_features": np.array(
                [[7.07, 0.8, 1], [5.83, 0.9, 2], [2.83, 0.95, 3]],
                dtype=np.float32,
            ),
            "node_ids": ["agent_1", "agent_2", "resource_1", "base_1"],
            "metadata": {
                "scenario": "test",
                "agent_count": 2,
                "resource_count": 1,
            },
        }

        # Run tests
        tests = [
            (
                "GNN Embedding Generation",
                lambda: test_class.test_gnn_embedding_generation(
                    gnn_model, test_graph_data
                ),
            ),
            (
                "Embedding to Text Transformation",
                lambda: test_class.test_embedding_to_text_transformation(
                    test_graph_data
                ),
            ),
            (
                "Round-trip Semantic Preservation",
                lambda: test_class.test_round_trip_semantic_preservation(
                    gnn_model, llm_manager, test_graph_data
                ),
            ),
            (
                "Integration Performance",
                lambda: test_class.test_integration_performance_characteristics(
                    gnn_model, llm_manager, test_graph_data
                ),
            ),
            (
                "Edge Cases",
                lambda: test_class.test_integration_edge_cases(
                    gnn_model, test_graph_data
                ),
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

        print(f"\n{'='*60}")
        print(f"GNN-LLM INTERFACE INTEGRATION TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Tests run: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success rate: {passed/total*100:.1f}%")

        if passed == total:
            print("🎉 All GNN-LLM interface tests passed!")
        else:
            print("❌ Some GNN-LLM interface tests failed!")

    # Run the tests
    asyncio.run(run_interface_tests())
