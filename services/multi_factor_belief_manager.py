"""Multi-Factor Belief Management Service for PyMDP Active Inference Engine.

This service implements advanced multi-factor belief management for complex state spaces
with hierarchical belief updates, cross-factor consistency validation, and optimized
performance for large-scale belief propagation.

Based on Task 44.5: Add Multi-Factor Beliefs and Complex State Support
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class FactorType(Enum):
    """Types of belief factors in complex state spaces."""

    INDEPENDENT = "independent"
    HIERARCHICAL = "hierarchical"
    CORRELATED = "correlated"
    DYNAMIC = "dynamic"


@dataclass
class FactorDependency:
    """Represents dependency between belief factors."""

    parent_factor: int
    child_factor: int
    dependency_type: str
    strength: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BeliefFactor:
    """Rich domain object representing a single belief factor."""

    index: int
    name: str
    beliefs: NDArray[np.floating]
    factor_type: FactorType
    dependencies: List[FactorDependency] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate factor consistency."""
        if not isinstance(self.beliefs, np.ndarray):
            raise TypeError(f"Factor {self.index} beliefs must be numpy array")

        if not np.isclose(self.beliefs.sum(), 1.0, rtol=1e-5):
            import warnings

            message = f"Factor {self.index} beliefs sum to {self.beliefs.sum():.6f}, normalizing"
            logger.warning(message)
            warnings.warn(message, UserWarning, stacklevel=2)
            self.beliefs = self.beliefs / self.beliefs.sum()

    @property
    def entropy(self) -> float:
        """Calculate entropy for this factor."""
        probs = self.beliefs + 1e-10
        probs = probs / probs.sum()
        return float(-np.sum(probs * np.log(probs)))

    @property
    def most_likely_state(self) -> int:
        """Get most likely state index."""
        return int(np.argmax(self.beliefs))

    @property
    def confidence(self) -> float:
        """Get confidence in most likely state."""
        return float(np.max(self.beliefs))


@dataclass
class CrossFactorCorrelation:
    """Represents correlation between belief factors."""

    factor_a: int
    factor_b: int
    correlation_matrix: NDArray[np.floating]
    correlation_strength: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Validate correlation matrix."""
        if not isinstance(self.correlation_matrix, np.ndarray):
            raise TypeError("Correlation matrix must be numpy array")

        if self.correlation_matrix.ndim != 2:
            raise ValueError(f"Correlation matrix must be 2D, got {self.correlation_matrix.ndim}D")

        rows, cols = self.correlation_matrix.shape
        if rows != cols:
            raise ValueError(
                f"Correlation matrix shape mismatch: {self.correlation_matrix.shape} (must be square)"
            )


@dataclass
class FactorizedBeliefState:
    """Advanced belief state representation for multi-factor systems."""

    factors: List[BeliefFactor]
    correlations: List[CrossFactorCorrelation] = field(default_factory=list)
    hierarchy: Dict[int, List[int]] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    agent_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate factorized belief state."""
        if not self.factors:
            raise ValueError("FactorizedBeliefState must have at least one factor")

        # Validate factor indices are consistent
        factor_indices = {f.index for f in self.factors}
        expected_indices = set(range(len(self.factors)))
        if factor_indices != expected_indices:
            raise ValueError(
                f"Factor indices {factor_indices} don't match expected {expected_indices}"
            )

    @property
    def num_factors(self) -> int:
        """Number of belief factors."""
        return len(self.factors)

    @property
    def overall_entropy(self) -> float:
        """Calculate overall belief entropy across all factors."""
        if not self.factors:
            return 0.0

        entropies = [factor.entropy for factor in self.factors]
        return float(np.mean(entropies))

    @property
    def overall_confidence(self) -> float:
        """Calculate overall confidence across all factors."""
        if not self.factors:
            return 0.0

        confidences = [factor.confidence for factor in self.factors]
        return float(np.mean(confidences))

    def get_factor(self, index: int) -> Optional[BeliefFactor]:
        """Get factor by index."""
        for factor in self.factors:
            if factor.index == index:
                return factor
        return None

    def get_correlated_factors(self, factor_index: int) -> List[int]:
        """Get list of factors correlated with given factor."""
        correlated = []
        for corr in self.correlations:
            if corr.factor_a == factor_index:
                correlated.append(corr.factor_b)
            elif corr.factor_b == factor_index:
                correlated.append(corr.factor_a)
        return correlated

    def get_hierarchical_children(self, parent_index: int) -> List[int]:
        """Get child factors in hierarchy."""
        return self.hierarchy.get(parent_index, [])

    def validate_consistency(self) -> Dict[str, Any]:
        """Validate cross-factor consistency."""
        validation_results = {
            "is_consistent": True,
            "violations": [],
            "warnings": [],
            "factor_scores": {},
        }

        # Check normalization for each factor
        for factor in self.factors:
            belief_sum = factor.beliefs.sum()
            if not np.isclose(belief_sum, 1.0, rtol=1e-5):
                validation_results["violations"].append(
                    f"Factor {factor.index} beliefs sum to {belief_sum:.6f}"
                )
                validation_results["is_consistent"] = False

        # Check correlation consistency
        for corr in self.correlations:
            if abs(corr.correlation_strength) > 1.0:
                validation_results["violations"].append(
                    f"Correlation between factors {corr.factor_a}-{corr.factor_b} "
                    f"has invalid strength {corr.correlation_strength}"
                )
                validation_results["is_consistent"] = False

        # Check hierarchical consistency
        for parent, children in self.hierarchy.items():
            parent_factor = self.get_factor(parent)
            if not parent_factor:
                validation_results["violations"].append(f"Hierarchical parent {parent} not found")
                validation_results["is_consistent"] = False
                continue

            for child in children:
                child_factor = self.get_factor(child)
                if not child_factor:
                    validation_results["violations"].append(f"Hierarchical child {child} not found")
                    validation_results["is_consistent"] = False

        return validation_results


class MultiFactorBeliefExtractor:
    """Extracts and manages multi-factor belief states from PyMDP agents."""

    def __init__(
        self,
        belief_threshold: float = 0.01,
        correlation_threshold: float = 0.1,
        enable_caching: bool = True,
    ):
        """Initialize multi-factor belief extractor.

        Args:
            belief_threshold: Minimum belief probability to consider significant
            correlation_threshold: Minimum correlation strength to track
            enable_caching: Whether to cache belief computations
        """
        self.belief_threshold = belief_threshold
        self.correlation_threshold = correlation_threshold
        self.enable_caching = enable_caching
        self._belief_cache: Dict[str, FactorizedBeliefState] = {}
        self._correlation_cache: Dict[str, List[CrossFactorCorrelation]] = {}

    async def extract_factorized_beliefs(
        self, agent: Any, agent_id: str, include_correlations: bool = True
    ) -> FactorizedBeliefState:
        """Extract factorized belief state from PyMDP agent.

        Args:
            agent: PyMDP agent instance
            agent_id: Unique agent identifier
            include_correlations: Whether to compute cross-factor correlations

        Returns:
            FactorizedBeliefState with rich factor information

        Raises:
            ValueError: If agent has no valid belief state
            RuntimeError: If belief extraction fails
        """
        try:
            logger.info(f"Extracting factorized beliefs from agent {agent_id}")

            # Check cache first
            cache_key = f"{agent_id}_{datetime.utcnow().isoformat()[:19]}"
            if self.enable_caching and cache_key in self._belief_cache:
                logger.debug(f"Using cached beliefs for agent {agent_id}")
                return self._belief_cache[cache_key]

            # Extract raw beliefs from agent
            raw_beliefs = await self._extract_raw_beliefs(agent)

            # Create belief factors
            factors = await self._create_belief_factors(raw_beliefs, agent)

            # Detect factor dependencies
            dependencies = await self._detect_factor_dependencies(factors, agent)

            # Update factors with dependencies
            for factor in factors:
                factor.dependencies = [
                    dep for dep in dependencies if dep.child_factor == factor.index
                ]

            # Compute correlations if requested
            correlations = []
            if include_correlations and len(factors) > 1:
                correlations = await self._compute_cross_factor_correlations(factors)

            # Build hierarchy information
            hierarchy = await self._build_factor_hierarchy(factors, dependencies)

            # Create factorized belief state
            belief_state = FactorizedBeliefState(
                factors=factors,
                correlations=correlations,
                hierarchy=hierarchy,
                agent_id=agent_id,
                metadata={
                    "extraction_method": "multi_factor",
                    "num_factors": len(factors),
                    "has_correlations": len(correlations) > 0,
                    "has_hierarchy": len(hierarchy) > 0,
                    "agent_class": type(agent).__name__,
                },
            )

            # Cache result
            if self.enable_caching:
                self._belief_cache[cache_key] = belief_state

            logger.info(f"Extracted {len(factors)} factors with {len(correlations)} correlations")
            return belief_state

        except Exception as e:
            logger.error(f"Failed to extract factorized beliefs from agent {agent_id}: {e}")
            raise RuntimeError(f"Belief extraction failed: {e}") from e

    async def _extract_raw_beliefs(self, agent: Any) -> List[NDArray[np.floating]]:
        """Extract raw belief arrays from PyMDP agent."""
        factor_beliefs = None

        # Try different attribute names for beliefs
        belief_attrs = ["qs", "beliefs", "posterior", "q_s"]
        for attr in belief_attrs:
            if hasattr(agent, attr):
                candidate_beliefs = getattr(agent, attr)
                if isinstance(candidate_beliefs, list) and len(candidate_beliefs) > 0:
                    factor_beliefs = candidate_beliefs
                    break
                elif isinstance(candidate_beliefs, np.ndarray) and candidate_beliefs.size > 0:
                    factor_beliefs = [candidate_beliefs]
                    break

        if factor_beliefs is None or len(factor_beliefs) == 0:
            raise ValueError("Agent has no valid belief state")

        # Ensure all beliefs are numpy arrays with proper dtype
        processed_beliefs = []
        for i, beliefs in enumerate(factor_beliefs):
            if not isinstance(beliefs, np.ndarray):
                beliefs = np.array(beliefs, dtype=np.float64)
            elif not np.issubdtype(beliefs.dtype, np.floating):
                beliefs = beliefs.astype(np.float64)

            # Validate beliefs are properly normalized
            if beliefs.size == 0:
                raise ValueError(f"Factor {i} has empty beliefs")

            belief_sum = beliefs.sum()
            if belief_sum <= 0:
                raise ValueError(f"Factor {i} has non-positive belief sum: {belief_sum}")

            # Normalize if needed
            if not np.isclose(belief_sum, 1.0, rtol=1e-5):
                beliefs = beliefs / belief_sum

            processed_beliefs.append(beliefs)

        return processed_beliefs

    async def _create_belief_factors(
        self, raw_beliefs: List[NDArray[np.floating]], agent: Any
    ) -> List[BeliefFactor]:
        """Create BeliefFactor objects from raw beliefs."""
        factors = []

        for i, beliefs in enumerate(raw_beliefs):
            # Determine factor type based on beliefs characteristics
            factor_type = self._classify_factor_type(beliefs, i, agent)

            # Extract factor metadata
            metadata = {
                "size": len(beliefs),
                "sparsity": float(np.sum(beliefs < self.belief_threshold) / len(beliefs)),
                "max_prob": float(np.max(beliefs)),
                "agent_context": getattr(agent, "agent_id", "unknown"),
            }

            # Add agent-specific metadata if available and valid
            if hasattr(agent, "action_precision"):
                try:
                    metadata["action_precision"] = float(agent.action_precision)
                except (TypeError, ValueError):
                    # Skip if can't convert to float
                    pass
            if hasattr(agent, "planning_horizon"):
                try:
                    metadata["planning_horizon"] = int(agent.planning_horizon)
                except (TypeError, ValueError):
                    # Skip if can't convert to int
                    pass

            factor = BeliefFactor(
                index=i,
                name=f"factor_{i}",
                beliefs=beliefs,
                factor_type=factor_type,
                metadata=metadata,
            )

            factors.append(factor)

        return factors

    def _classify_factor_type(
        self, beliefs: NDArray[np.floating], factor_index: int, agent: Any
    ) -> FactorType:
        """Classify the type of belief factor based on characteristics."""
        # Simple heuristic classification
        entropy = -np.sum((beliefs + 1e-10) * np.log(beliefs + 1e-10))
        max_prob = np.max(beliefs)
        sparsity = np.sum(beliefs < self.belief_threshold) / len(beliefs)

        if sparsity > 0.8:
            return FactorType.INDEPENDENT
        elif max_prob > 0.8:
            return FactorType.HIERARCHICAL
        elif entropy > 2.0:
            return FactorType.DYNAMIC
        else:
            return FactorType.CORRELATED

    async def _detect_factor_dependencies(
        self, factors: List[BeliefFactor], agent: Any
    ) -> List[FactorDependency]:
        """Detect dependencies between belief factors."""
        dependencies = []

        if len(factors) < 2:
            return dependencies

        # Check for hierarchical dependencies
        for i, parent in enumerate(factors):
            for j, child in enumerate(factors):
                if i == j:
                    continue

                # Simple dependency detection based on entropy relationship
                if parent.entropy < child.entropy and parent.confidence > 0.7:
                    dependency = FactorDependency(
                        parent_factor=i,
                        child_factor=j,
                        dependency_type="hierarchical",
                        strength=float(parent.confidence - child.confidence),
                        metadata={
                            "parent_entropy": parent.entropy,
                            "child_entropy": child.entropy,
                            "confidence_diff": parent.confidence - child.confidence,
                        },
                    )
                    dependencies.append(dependency)

        return dependencies

    async def _compute_cross_factor_correlations(
        self, factors: List[BeliefFactor]
    ) -> List[CrossFactorCorrelation]:
        """Compute correlations between belief factors."""
        correlations = []

        for i in range(len(factors)):
            for j in range(i + 1, len(factors)):
                factor_a = factors[i]
                factor_b = factors[j]

                # Compute correlation between belief distributions
                # Use minimum length for correlation computation
                min_len = min(len(factor_a.beliefs), len(factor_b.beliefs))
                beliefs_a = factor_a.beliefs[:min_len]
                beliefs_b = factor_b.beliefs[:min_len]

                correlation = float(np.corrcoef(beliefs_a, beliefs_b)[0, 1])

                # Only include significant correlations
                if abs(correlation) > self.correlation_threshold:
                    # Create correlation matrix (simplified for now)
                    corr_matrix = np.array([[1.0, correlation], [correlation, 1.0]])

                    cross_corr = CrossFactorCorrelation(
                        factor_a=i,
                        factor_b=j,
                        correlation_matrix=corr_matrix,
                        correlation_strength=correlation,
                    )
                    correlations.append(cross_corr)

        return correlations

    async def _build_factor_hierarchy(
        self, factors: List[BeliefFactor], dependencies: List[FactorDependency]
    ) -> Dict[int, List[int]]:
        """Build hierarchical structure from factor dependencies."""
        hierarchy = defaultdict(list)

        for dep in dependencies:
            if dep.dependency_type == "hierarchical":
                hierarchy[dep.parent_factor].append(dep.child_factor)

        return dict(hierarchy)


class HierarchicalBeliefUpdater:
    """Manages hierarchical belief updates in multi-factor systems."""

    def __init__(self, max_iterations: int = 10, convergence_threshold: float = 1e-6):
        """Initialize hierarchical belief updater.

        Args:
            max_iterations: Maximum iterations for belief propagation
            convergence_threshold: Convergence threshold for belief updates
        """
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    async def propagate_beliefs(
        self, belief_state: FactorizedBeliefState, observations: Optional[Dict[int, int]] = None
    ) -> FactorizedBeliefState:
        """Propagate beliefs through hierarchical factor structure.

        Args:
            belief_state: Current factorized belief state
            observations: Optional observations for specific factors

        Returns:
            Updated factorized belief state
        """
        logger.info(f"Propagating beliefs through {belief_state.num_factors} factors")

        updated_factors = [factor for factor in belief_state.factors]  # Copy factors
        observations_applied_count = 0

        # Apply observations if provided
        if observations:
            for factor_idx, obs_value in observations.items():
                factor = belief_state.get_factor(factor_idx)
                if factor and 0 <= obs_value < len(factor.beliefs):
                    # Update beliefs based on observation
                    new_beliefs = np.zeros_like(factor.beliefs)
                    new_beliefs[obs_value] = 1.0

                    # Update the factor in our copy
                    for i, f in enumerate(updated_factors):
                        if f.index == factor_idx:
                            updated_factors[i] = BeliefFactor(
                                index=f.index,
                                name=f.name,
                                beliefs=new_beliefs,
                                factor_type=f.factor_type,
                                dependencies=f.dependencies,
                                metadata={**f.metadata, "observation_applied": obs_value},
                            )
                            break

                    observations_applied_count += 1

        # Iterative belief propagation through hierarchy
        for iteration in range(self.max_iterations):
            prev_beliefs = [f.beliefs.copy() for f in updated_factors]
            converged = True

            # Update beliefs based on hierarchical dependencies
            for parent_idx, children in belief_state.hierarchy.items():
                parent_factor = next((f for f in updated_factors if f.index == parent_idx), None)
                if not parent_factor:
                    continue

                # Propagate parent beliefs to children
                for child_idx in children:
                    child_factor = next((f for f in updated_factors if f.index == child_idx), None)
                    if not child_factor:
                        continue

                    # Simple belief propagation (can be made more sophisticated)
                    influence_strength = 0.1  # Could be learned or configured
                    new_child_beliefs = (
                        1 - influence_strength
                    ) * child_factor.beliefs + influence_strength * parent_factor.beliefs[
                        : len(child_factor.beliefs)
                    ]

                    # Normalize
                    new_child_beliefs = new_child_beliefs / new_child_beliefs.sum()

                    # Update child factor
                    for i, f in enumerate(updated_factors):
                        if f.index == child_idx:
                            updated_factors[i] = BeliefFactor(
                                index=f.index,
                                name=f.name,
                                beliefs=new_child_beliefs,
                                factor_type=f.factor_type,
                                dependencies=f.dependencies,
                                metadata=f.metadata,
                            )
                            break

            # Check convergence
            for i, factor in enumerate(updated_factors):
                if np.max(np.abs(factor.beliefs - prev_beliefs[i])) > self.convergence_threshold:
                    converged = False
                    break

            if converged:
                logger.debug(f"Belief propagation converged after {iteration + 1} iterations")
                break

        # Create updated belief state
        updated_state = FactorizedBeliefState(
            factors=updated_factors,
            correlations=belief_state.correlations,
            hierarchy=belief_state.hierarchy,
            timestamp=datetime.utcnow(),
            agent_id=belief_state.agent_id,
            metadata={
                **belief_state.metadata,
                "propagation_iterations": iteration + 1,
                "converged": converged,
                "observations_applied": observations_applied_count,
            },
        )

        return updated_state


class BeliefConsistencyValidator:
    """Validates consistency across multi-factor belief systems."""

    def __init__(self, strict_validation: bool = True):
        """Initialize belief consistency validator.

        Args:
            strict_validation: Whether to enforce strict consistency checks
        """
        self.strict_validation = strict_validation

    async def validate_factorized_state(
        self, belief_state: FactorizedBeliefState
    ) -> Dict[str, Any]:
        """Comprehensive validation of factorized belief state.

        Args:
            belief_state: Factorized belief state to validate

        Returns:
            Validation results with detailed diagnostics
        """
        logger.debug(f"Validating factorized belief state with {belief_state.num_factors} factors")

        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "factor_diagnostics": {},
            "correlation_diagnostics": {},
            "hierarchy_diagnostics": {},
            "performance_metrics": {},
        }

        # Validate individual factors
        for factor in belief_state.factors:
            factor_results = await self._validate_factor(factor)
            validation_results["factor_diagnostics"][factor.index] = factor_results

            if not factor_results["is_valid"]:
                validation_results["is_valid"] = False
                validation_results["errors"].extend(factor_results["errors"])

            validation_results["warnings"].extend(factor_results["warnings"])

        # Validate correlations
        if belief_state.correlations:
            correlation_results = await self._validate_correlations(belief_state.correlations)
            validation_results["correlation_diagnostics"] = correlation_results

            if not correlation_results["is_valid"]:
                validation_results["is_valid"] = False
                validation_results["errors"].extend(correlation_results["errors"])

        # Validate hierarchy
        if belief_state.hierarchy:
            hierarchy_results = await self._validate_hierarchy(belief_state)
            validation_results["hierarchy_diagnostics"] = hierarchy_results

            if not hierarchy_results["is_valid"]:
                validation_results["is_valid"] = False
                validation_results["errors"].extend(hierarchy_results["errors"])

        # Performance metrics
        validation_results["performance_metrics"] = {
            "total_factors": belief_state.num_factors,
            "overall_entropy": belief_state.overall_entropy,
            "overall_confidence": belief_state.overall_confidence,
            "correlation_count": len(belief_state.correlations),
            "hierarchy_depth": self._calculate_hierarchy_depth(belief_state.hierarchy),
        }

        return validation_results

    async def _validate_factor(self, factor: BeliefFactor) -> Dict[str, Any]:
        """Validate individual belief factor."""
        results = {"is_valid": True, "errors": [], "warnings": [], "metrics": {}}

        # Check belief normalization
        belief_sum = factor.beliefs.sum()
        if not np.isclose(belief_sum, 1.0, rtol=1e-5):
            if self.strict_validation:
                results["is_valid"] = False
                results["errors"].append(f"Factor {factor.index} beliefs sum to {belief_sum:.6f}")
            else:
                results["warnings"].append(f"Factor {factor.index} beliefs poorly normalized")

        # Check for negative beliefs
        if np.any(factor.beliefs < 0):
            results["is_valid"] = False
            results["errors"].append(f"Factor {factor.index} has negative beliefs")

        # Check for NaN or infinite values
        if np.any(~np.isfinite(factor.beliefs)):
            results["is_valid"] = False
            results["errors"].append(f"Factor {factor.index} has non-finite beliefs")

        # Performance metrics
        results["metrics"] = {
            "entropy": factor.entropy,
            "confidence": factor.confidence,
            "sparsity": float(np.sum(factor.beliefs < 0.01) / len(factor.beliefs)),
            "effective_states": int(np.sum(factor.beliefs > 0.01)),
        }

        return results

    async def _validate_correlations(
        self, correlations: List[CrossFactorCorrelation]
    ) -> Dict[str, Any]:
        """Validate cross-factor correlations."""
        results = {"is_valid": True, "errors": [], "warnings": [], "metrics": {}}

        correlation_strengths = []

        for corr in correlations:
            # Check correlation strength bounds
            if abs(corr.correlation_strength) > 1.0:
                results["is_valid"] = False
                results["errors"].append(
                    f"Correlation between factors {corr.factor_a}-{corr.factor_b} "
                    f"has invalid strength {corr.correlation_strength}"
                )

            # Check correlation matrix validity
            if not np.allclose(corr.correlation_matrix, corr.correlation_matrix.T):
                results["warnings"].append(
                    f"Correlation matrix for {corr.factor_a}-{corr.factor_b} not symmetric"
                )

            correlation_strengths.append(abs(corr.correlation_strength))

        # Performance metrics
        if correlation_strengths:
            results["metrics"] = {
                "average_correlation": float(np.mean(correlation_strengths)),
                "max_correlation": float(np.max(correlation_strengths)),
                "correlation_count": len(correlations),
            }

        return results

    async def _validate_hierarchy(self, belief_state: FactorizedBeliefState) -> Dict[str, Any]:
        """Validate hierarchical factor structure."""
        results = {"is_valid": True, "errors": [], "warnings": [], "metrics": {}}

        # Check for circular dependencies
        visited = set()
        rec_stack = set()

        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)

            for child in belief_state.hierarchy.get(node, []):
                if child not in visited:
                    if has_cycle(child):
                        return True
                elif child in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for root in belief_state.hierarchy.keys():
            if root not in visited:
                if has_cycle(root):
                    results["is_valid"] = False
                    results["errors"].append(
                        f"Circular dependency detected in hierarchy at node {root}"
                    )

        # Check that all referenced factors exist
        all_factor_indices = {f.index for f in belief_state.factors}
        for parent, children in belief_state.hierarchy.items():
            if parent not in all_factor_indices:
                results["is_valid"] = False
                results["errors"].append(f"Hierarchy parent {parent} not found in factors")

            for child in children:
                if child not in all_factor_indices:
                    results["is_valid"] = False
                    results["errors"].append(f"Hierarchy child {child} not found in factors")

        # Performance metrics
        results["metrics"] = {
            "hierarchy_depth": self._calculate_hierarchy_depth(belief_state.hierarchy),
            "total_nodes": len(belief_state.hierarchy),
            "total_edges": sum(len(children) for children in belief_state.hierarchy.values()),
        }

        return results

    def _calculate_hierarchy_depth(self, hierarchy: Dict[int, List[int]]) -> int:
        """Calculate maximum depth of hierarchy."""
        if not hierarchy:
            return 0

        def get_depth(node, visited):
            if node in visited:
                return 0  # Avoid infinite recursion

            visited.add(node)
            children = hierarchy.get(node, [])
            if not children:
                visited.remove(node)
                return 1

            max_child_depth = max(get_depth(child, visited) for child in children)
            visited.remove(node)
            return 1 + max_child_depth

        max_depth = 0
        for root in hierarchy.keys():
            depth = get_depth(root, set())
            max_depth = max(max_depth, depth)

        return max_depth


# Factory function for dependency injection
def create_multi_factor_belief_manager() -> (
    Tuple[MultiFactorBeliefExtractor, HierarchicalBeliefUpdater, BeliefConsistencyValidator]
):
    """Create configured multi-factor belief management components.

    Returns:
        Tuple of (extractor, updater, validator) for multi-factor belief management
    """
    extractor = MultiFactorBeliefExtractor(
        belief_threshold=0.01, correlation_threshold=0.1, enable_caching=True
    )

    updater = HierarchicalBeliefUpdater(max_iterations=10, convergence_threshold=1e-6)

    validator = BeliefConsistencyValidator(strict_validation=True)

    return extractor, updater, validator
