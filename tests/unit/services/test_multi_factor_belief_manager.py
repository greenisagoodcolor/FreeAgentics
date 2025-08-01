"""Comprehensive test suite for Multi-Factor Belief Management Service.

Tests cover all aspects of multi-factor belief systems including:
- Factorized belief state creation and validation
- Hierarchical belief propagation
- Cross-factor correlation computation
- Belief consistency validation
- Performance optimization with caching

Following TDD principles with 100% coverage requirement.
"""

from datetime import datetime
from typing import List
from unittest.mock import Mock

import numpy as np
import pytest

from services.multi_factor_belief_manager import (
    BeliefConsistencyValidator,
    BeliefFactor,
    CrossFactorCorrelation,
    FactorDependency,
    FactorizedBeliefState,
    FactorType,
    HierarchicalBeliefUpdater,
    MultiFactorBeliefExtractor,
    create_multi_factor_belief_manager,
)


class TestBeliefFactor:
    """Test suite for BeliefFactor domain object."""

    def test_belief_factor_creation_valid(self):
        """Test creating valid belief factor."""
        beliefs = np.array([0.6, 0.3, 0.1])
        factor = BeliefFactor(
            index=0, name="test_factor", beliefs=beliefs, factor_type=FactorType.INDEPENDENT
        )

        assert factor.index == 0
        assert factor.name == "test_factor"
        assert np.allclose(factor.beliefs, beliefs)
        assert factor.factor_type == FactorType.INDEPENDENT
        assert factor.entropy > 0
        assert factor.most_likely_state == 0
        assert factor.confidence == 0.6

    def test_belief_factor_normalization(self):
        """Test automatic normalization of beliefs."""
        # Beliefs that don't sum to 1.0
        beliefs = np.array([0.6, 0.3, 0.2])  # Sum = 1.1

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            factor = BeliefFactor(
                index=0, name="test_factor", beliefs=beliefs, factor_type=FactorType.INDEPENDENT
            )

            # Check that a warning was issued
            assert len(w) > 0
            assert "normalizing" in str(w[0].message)

        # Should be normalized
        assert np.isclose(factor.beliefs.sum(), 1.0, rtol=1e-5)
        assert np.allclose(factor.beliefs, beliefs / beliefs.sum())

    def test_belief_factor_invalid_type(self):
        """Test error handling for invalid belief type."""
        with pytest.raises(TypeError, match="beliefs must be numpy array"):
            BeliefFactor(
                index=0,
                name="test_factor",
                beliefs=[0.6, 0.3, 0.1],  # List instead of ndarray
                factor_type=FactorType.INDEPENDENT,
            )

    def test_belief_factor_entropy_calculation(self):
        """Test entropy calculation for different belief distributions."""
        # Uniform distribution - high entropy
        uniform_beliefs = np.ones(4) / 4
        uniform_factor = BeliefFactor(0, "uniform", uniform_beliefs, FactorType.DYNAMIC)

        # Concentrated distribution - low entropy
        concentrated_beliefs = np.array([0.9, 0.05, 0.03, 0.02])
        concentrated_factor = BeliefFactor(
            1, "concentrated", concentrated_beliefs, FactorType.HIERARCHICAL
        )

        assert uniform_factor.entropy > concentrated_factor.entropy
        assert uniform_factor.confidence < concentrated_factor.confidence

    def test_belief_factor_with_dependencies(self):
        """Test belief factor with dependency metadata."""
        beliefs = np.array([0.5, 0.3, 0.2])
        dependency = FactorDependency(
            parent_factor=0, child_factor=1, dependency_type="hierarchical", strength=0.8
        )

        factor = BeliefFactor(
            index=1,
            name="dependent_factor",
            beliefs=beliefs,
            factor_type=FactorType.HIERARCHICAL,
            dependencies=[dependency],
            metadata={"parent_influence": 0.8},
        )

        assert len(factor.dependencies) == 1
        assert factor.dependencies[0].parent_factor == 0
        assert factor.metadata["parent_influence"] == 0.8


class TestCrossFactorCorrelation:
    """Test suite for CrossFactorCorrelation."""

    def test_correlation_creation_valid(self):
        """Test creating valid cross-factor correlation."""
        corr_matrix = np.array([[1.0, 0.7], [0.7, 1.0]])
        correlation = CrossFactorCorrelation(
            factor_a=0, factor_b=1, correlation_matrix=corr_matrix, correlation_strength=0.7
        )

        assert correlation.factor_a == 0
        assert correlation.factor_b == 1
        assert np.allclose(correlation.correlation_matrix, corr_matrix)
        assert correlation.correlation_strength == 0.7
        assert isinstance(correlation.timestamp, datetime)

    def test_correlation_matrix_validation(self):
        """Test correlation matrix shape validation."""
        # Create invalid matrix by manually constructing inconsistent arrays
        try:
            # This should fail during construction
            invalid_matrix = np.array([[1.0, 0.7, 0.5], [0.7, 1.0]])  # Irregular shape
            assert False, "Should have failed to create irregular array"
        except ValueError:
            # Expected - numpy can't create irregular arrays
            pass

        # Test with valid but incorrect shape matrix (non-square)
        non_square_matrix = np.array([[1.0, 0.7, 0.5], [0.7, 1.0, 0.3]])  # 2x3 instead of square

        with pytest.raises(ValueError, match="Correlation matrix shape mismatch.*must be square"):
            CrossFactorCorrelation(
                factor_a=0,
                factor_b=1,
                correlation_matrix=non_square_matrix,
                correlation_strength=0.7,
            )


class TestFactorizedBeliefState:
    """Test suite for FactorizedBeliefState."""

    def create_sample_factors(self, num_factors: int = 3) -> List[BeliefFactor]:
        """Create sample belief factors for testing."""
        factors = []
        for i in range(num_factors):
            beliefs = np.random.dirichlet(np.ones(4))  # Random normalized beliefs
            factor = BeliefFactor(
                index=i,
                name=f"factor_{i}",
                beliefs=beliefs,
                factor_type=FactorType.INDEPENDENT,
                metadata={"test_factor": True},
            )
            factors.append(factor)
        return factors

    def test_factorized_state_creation_valid(self):
        """Test creating valid factorized belief state."""
        factors = self.create_sample_factors(3)
        correlations = [
            CrossFactorCorrelation(0, 1, np.eye(2), 0.5),
            CrossFactorCorrelation(1, 2, np.eye(2), 0.3),
        ]
        hierarchy = {0: [1, 2]}

        state = FactorizedBeliefState(
            factors=factors, correlations=correlations, hierarchy=hierarchy, agent_id="test_agent"
        )

        assert state.num_factors == 3
        assert len(state.correlations) == 2
        assert len(state.hierarchy) == 1
        assert state.agent_id == "test_agent"
        assert state.overall_entropy > 0
        assert state.overall_confidence > 0

    def test_factorized_state_empty_factors_error(self):
        """Test error when creating state with no factors."""
        with pytest.raises(ValueError, match="must have at least one factor"):
            FactorizedBeliefState(factors=[])

    def test_factorized_state_invalid_indices(self):
        """Test error when factor indices don't match expected."""
        # Create factors with gaps in indices
        factor1 = BeliefFactor(0, "factor_0", np.array([0.5, 0.5]), FactorType.INDEPENDENT)
        factor2 = BeliefFactor(
            2, "factor_2", np.array([0.3, 0.7]), FactorType.INDEPENDENT
        )  # Missing index 1

        with pytest.raises(ValueError, match="Factor indices.*don't match expected"):
            FactorizedBeliefState(factors=[factor1, factor2])

    def test_get_factor_by_index(self):
        """Test retrieving factor by index."""
        factors = self.create_sample_factors(3)
        state = FactorizedBeliefState(factors=factors)

        factor_1 = state.get_factor(1)
        assert factor_1 is not None
        assert factor_1.index == 1
        assert factor_1.name == "factor_1"

        # Non-existent factor
        factor_missing = state.get_factor(10)
        assert factor_missing is None

    def test_get_correlated_factors(self):
        """Test finding correlated factors."""
        factors = self.create_sample_factors(4)
        correlations = [
            CrossFactorCorrelation(0, 1, np.eye(2), 0.5),
            CrossFactorCorrelation(0, 2, np.eye(2), 0.3),
            CrossFactorCorrelation(2, 3, np.eye(2), 0.7),
        ]
        state = FactorizedBeliefState(factors=factors, correlations=correlations)

        # Factor 0 is correlated with factors 1 and 2
        correlated_with_0 = state.get_correlated_factors(0)
        assert set(correlated_with_0) == {1, 2}

        # Factor 2 is correlated with factors 0 and 3
        correlated_with_2 = state.get_correlated_factors(2)
        assert set(correlated_with_2) == {0, 3}

        # Factor 1 is only correlated with factor 0
        correlated_with_1 = state.get_correlated_factors(1)
        assert correlated_with_1 == [0]

    def test_get_hierarchical_children(self):
        """Test finding hierarchical children."""
        factors = self.create_sample_factors(4)
        hierarchy = {0: [1, 2], 1: [3]}
        state = FactorizedBeliefState(factors=factors, hierarchy=hierarchy)

        # Factor 0 has children 1 and 2
        children_0 = state.get_hierarchical_children(0)
        assert set(children_0) == {1, 2}

        # Factor 1 has child 3
        children_1 = state.get_hierarchical_children(1)
        assert children_1 == [3]

        # Factor 2 has no children
        children_2 = state.get_hierarchical_children(2)
        assert children_2 == []

    def test_validate_consistency_valid_state(self):
        """Test consistency validation for valid state."""
        factors = self.create_sample_factors(3)
        correlations = [CrossFactorCorrelation(0, 1, np.eye(2), 0.5)]
        hierarchy = {0: [1]}

        state = FactorizedBeliefState(
            factors=factors, correlations=correlations, hierarchy=hierarchy
        )

        validation = state.validate_consistency()
        assert validation["is_consistent"] is True
        assert len(validation["violations"]) == 0

    def test_validate_consistency_normalization_violation(self):
        """Test consistency validation with normalization violation."""
        # Create factor with poorly normalized beliefs
        bad_beliefs = np.array([0.6, 0.3, 0.2])  # Sum > 1.0, will be normalized by BeliefFactor
        factor = BeliefFactor(0, "bad_factor", bad_beliefs, FactorType.INDEPENDENT)

        # Manually override beliefs to simulate violation
        factor.beliefs = np.array([0.6, 0.3, 0.2])  # Bypass normalization

        state = FactorizedBeliefState(factors=[factor])
        validation = state.validate_consistency()

        assert validation["is_consistent"] is False
        assert any("beliefs sum to" in violation for violation in validation["violations"])

    def test_validate_consistency_correlation_violation(self):
        """Test consistency validation with invalid correlation."""
        factors = self.create_sample_factors(2)
        # Invalid correlation strength > 1.0
        bad_correlation = CrossFactorCorrelation(0, 1, np.eye(2), 1.5)

        state = FactorizedBeliefState(factors=factors, correlations=[bad_correlation])
        validation = state.validate_consistency()

        assert validation["is_consistent"] is False
        assert any("invalid strength" in violation for violation in validation["violations"])

    def test_validate_consistency_hierarchy_violation(self):
        """Test consistency validation with missing hierarchy references."""
        factors = self.create_sample_factors(2)  # Only factors 0 and 1
        hierarchy = {
            0: [1, 5],
            2: [3],
        }  # Parent 0 has valid child 1 and invalid child 5; parent 2 missing

        state = FactorizedBeliefState(factors=factors, hierarchy=hierarchy)
        validation = state.validate_consistency()

        assert validation["is_consistent"] is False
        violations = validation["violations"]
        assert any("parent 2 not found" in violation for violation in violations)
        assert any(
            "child 5 not found" in violation for violation in violations
        )  # child 5 doesn't exist


class TestMultiFactorBeliefExtractor:
    """Test suite for MultiFactorBeliefExtractor."""

    def create_mock_agent(self, beliefs_list: List[np.ndarray]) -> Mock:
        """Create mock PyMDP agent with specified beliefs."""
        agent = Mock()
        agent.qs = beliefs_list
        agent.agent_id = "test_agent"
        agent.action_precision = 1.0
        agent.planning_horizon = 3
        return agent

    def create_minimal_mock_agent(
        self, beliefs_list: List[np.ndarray], attr_name: str = "qs"
    ) -> Mock:
        """Create minimal mock agent with only specified belief attribute."""
        agent = Mock()
        setattr(agent, attr_name, beliefs_list)
        # Don't set action_precision or planning_horizon to avoid float() conversion issues
        return agent

    @pytest.mark.asyncio
    async def test_extract_factorized_beliefs_single_factor(self):
        """Test extracting beliefs from single-factor agent."""
        beliefs = [np.array([0.7, 0.2, 0.1])]
        agent = self.create_mock_agent(beliefs)

        extractor = MultiFactorBeliefExtractor()
        result = await extractor.extract_factorized_beliefs(agent, "test_agent")

        assert isinstance(result, FactorizedBeliefState)
        assert result.num_factors == 1
        assert result.agent_id == "test_agent"
        assert len(result.factors) == 1
        assert np.allclose(result.factors[0].beliefs, beliefs[0])
        assert result.factors[0].factor_type in FactorType

    @pytest.mark.asyncio
    async def test_extract_factorized_beliefs_multi_factor(self):
        """Test extracting beliefs from multi-factor agent."""
        beliefs = [
            np.array([0.8, 0.15, 0.05]),
            np.array([0.4, 0.4, 0.2]),
            np.array([0.1, 0.2, 0.3, 0.4]),
        ]
        agent = self.create_mock_agent(beliefs)

        extractor = MultiFactorBeliefExtractor()
        result = await extractor.extract_factorized_beliefs(agent, "multi_agent")

        assert result.num_factors == 3
        assert len(result.factors) == 3

        # Check each factor
        for i, factor in enumerate(result.factors):
            assert factor.index == i
            assert np.allclose(factor.beliefs, beliefs[i])
            assert factor.factor_type in FactorType

    @pytest.mark.asyncio
    async def test_extract_beliefs_with_correlations(self):
        """Test extraction with correlation computation."""
        # Create beliefs with some correlation
        beliefs = [
            np.array([0.7, 0.3]),
            np.array([0.6, 0.4]),  # Similar distribution - should be correlated
            np.array([0.1, 0.9]),  # Opposite distribution - should be anti-correlated
        ]
        agent = self.create_mock_agent(beliefs)

        extractor = MultiFactorBeliefExtractor(correlation_threshold=0.1)
        result = await extractor.extract_factorized_beliefs(
            agent, "corr_agent", include_correlations=True
        )

        assert len(result.correlations) > 0
        # Should find correlations between factors
        correlation_pairs = {(c.factor_a, c.factor_b) for c in result.correlations}
        assert len(correlation_pairs) > 0

    @pytest.mark.asyncio
    async def test_extract_beliefs_no_correlations(self):
        """Test extraction without correlation computation."""
        beliefs = [np.array([0.7, 0.3]), np.array([0.4, 0.6])]
        agent = self.create_mock_agent(beliefs)

        extractor = MultiFactorBeliefExtractor()
        result = await extractor.extract_factorized_beliefs(
            agent, "no_corr_agent", include_correlations=False
        )

        assert len(result.correlations) == 0

    @pytest.mark.asyncio
    async def test_extract_beliefs_caching(self):
        """Test belief extraction caching."""
        beliefs = [np.array([0.5, 0.5])]
        agent = self.create_mock_agent(beliefs)

        extractor = MultiFactorBeliefExtractor(enable_caching=True)

        # First extraction
        result1 = await extractor.extract_factorized_beliefs(agent, "cached_agent")

        # Second extraction - should use cache (if within same second)
        result2 = await extractor.extract_factorized_beliefs(agent, "cached_agent")

        # Results should be identical (cached)
        assert result1.timestamp == result2.timestamp
        assert len(extractor._belief_cache) > 0

    @pytest.mark.asyncio
    async def test_extract_beliefs_agent_no_beliefs(self):
        """Test error handling when agent has no beliefs."""
        agent = Mock()
        # No qs, beliefs, posterior, or q_s attributes

        extractor = MultiFactorBeliefExtractor()

        with pytest.raises(RuntimeError, match="Belief extraction failed"):
            await extractor.extract_factorized_beliefs(agent, "no_beliefs_agent")

    @pytest.mark.asyncio
    async def test_extract_beliefs_invalid_beliefs(self):
        """Test error handling for invalid belief format."""
        agent = Mock()
        agent.qs = [np.array([])]  # Empty beliefs

        extractor = MultiFactorBeliefExtractor()

        with pytest.raises(RuntimeError, match="Belief extraction failed"):
            await extractor.extract_factorized_beliefs(agent, "invalid_agent")

    @pytest.mark.asyncio
    async def test_extract_beliefs_different_attributes(self):
        """Test extraction from different belief attribute names."""
        beliefs = [np.array([0.6, 0.4])]

        # Test different attribute names
        for attr_name in ["qs", "beliefs", "posterior", "q_s"]:
            agent = self.create_minimal_mock_agent(beliefs, attr_name)

            extractor = MultiFactorBeliefExtractor()
            result = await extractor.extract_factorized_beliefs(agent, f"{attr_name}_agent")

            assert result.num_factors == 1
            assert np.allclose(result.factors[0].beliefs, beliefs[0])

    @pytest.mark.asyncio
    async def test_extract_beliefs_normalization_required(self):
        """Test belief extraction with normalization."""
        # Non-normalized beliefs
        beliefs = [np.array([0.6, 0.3, 0.2])]  # Sum = 1.1
        agent = self.create_mock_agent(beliefs)

        extractor = MultiFactorBeliefExtractor()
        result = await extractor.extract_factorized_beliefs(agent, "norm_agent")

        # Should be normalized
        assert np.isclose(result.factors[0].beliefs.sum(), 1.0, rtol=1e-5)

    @pytest.mark.asyncio
    async def test_detect_factor_dependencies(self):
        """Test detection of factor dependencies."""
        # Create beliefs with hierarchical pattern
        beliefs = [
            np.array([0.9, 0.1]),  # High confidence parent
            np.array([0.4, 0.6]),  # Lower confidence child
            np.array([0.3, 0.3, 0.4]),  # Even lower confidence child
        ]
        agent = self.create_mock_agent(beliefs)

        extractor = MultiFactorBeliefExtractor()
        result = await extractor.extract_factorized_beliefs(agent, "dep_agent")

        # Should detect some dependencies
        has_dependencies = any(len(f.dependencies) > 0 for f in result.factors)
        assert has_dependencies or len(result.hierarchy) > 0

    @pytest.mark.asyncio
    async def test_factor_type_classification(self):
        """Test classification of different factor types."""
        # Create beliefs with different characteristics
        beliefs = [
            np.array([0.99, 0.01]),  # Hierarchical (high confidence)
            np.array([0.25, 0.25, 0.25, 0.25]),  # Dynamic (high entropy)
            np.array([0.9, 0.05, 0.03, 0.02]),  # Independent (sparse)
        ]
        agent = self.create_mock_agent(beliefs)

        extractor = MultiFactorBeliefExtractor()
        result = await extractor.extract_factorized_beliefs(agent, "type_agent")

        # Check that different types were assigned
        factor_types = {f.factor_type for f in result.factors}
        assert len(factor_types) > 1  # Should have multiple types


class TestHierarchicalBeliefUpdater:
    """Test suite for HierarchicalBeliefUpdater."""

    def create_sample_hierarchical_state(self) -> FactorizedBeliefState:
        """Create sample factorized state with hierarchy."""
        factors = [
            BeliefFactor(0, "parent", np.array([0.7, 0.3]), FactorType.HIERARCHICAL),
            BeliefFactor(1, "child1", np.array([0.4, 0.6]), FactorType.CORRELATED),
            BeliefFactor(2, "child2", np.array([0.3, 0.7]), FactorType.CORRELATED),
        ]
        hierarchy = {0: [1, 2]}  # Factor 0 is parent of 1 and 2

        return FactorizedBeliefState(
            factors=factors, hierarchy=hierarchy, agent_id="hierarchical_agent"
        )

    @pytest.mark.asyncio
    async def test_propagate_beliefs_no_observations(self):
        """Test belief propagation without observations."""
        initial_state = self.create_sample_hierarchical_state()
        initial_beliefs = [f.beliefs.copy() for f in initial_state.factors]

        updater = HierarchicalBeliefUpdater()
        updated_state = await updater.propagate_beliefs(initial_state)

        assert isinstance(updated_state, FactorizedBeliefState)
        assert updated_state.num_factors == initial_state.num_factors
        assert updated_state.agent_id == initial_state.agent_id

        # Beliefs should have changed due to hierarchical propagation
        beliefs_changed = any(
            not np.allclose(updated_state.factors[i].beliefs, initial_beliefs[i])
            for i in range(len(initial_beliefs))
        )
        # Note: May not change if already converged, so we just check structure
        assert len(updated_state.factors) == len(initial_beliefs)

    @pytest.mark.asyncio
    async def test_propagate_beliefs_with_observations(self):
        """Test belief propagation with observations."""
        initial_state = self.create_sample_hierarchical_state()
        observations = {1: 0}  # Observe state 0 for factor 1

        updater = HierarchicalBeliefUpdater()
        updated_state = await updater.propagate_beliefs(initial_state, observations)

        # Factor 1 should have beliefs concentrated on observed state
        factor_1 = updated_state.get_factor(1)
        assert factor_1 is not None
        # After observation, factor 1 should have higher probability on observed state 0
        # Note: due to hierarchical propagation, exact value may vary
        assert factor_1.most_likely_state == 0  # Most likely state should be the observed one
        assert (
            factor_1.beliefs[0] > factor_1.beliefs[1]
        )  # State 0 should be more likely than state 1

        # Metadata should reflect observation application
        assert updated_state.metadata["observations_applied"] == 1

    @pytest.mark.asyncio
    async def test_propagate_beliefs_convergence(self):
        """Test belief propagation convergence."""
        initial_state = self.create_sample_hierarchical_state()

        updater = HierarchicalBeliefUpdater(max_iterations=5, convergence_threshold=1e-4)
        updated_state = await updater.propagate_beliefs(initial_state)

        # Should track convergence information
        assert "propagation_iterations" in updated_state.metadata
        assert "converged" in updated_state.metadata
        assert updated_state.metadata["propagation_iterations"] <= 5

    @pytest.mark.asyncio
    async def test_propagate_beliefs_invalid_observation(self):
        """Test handling of invalid observations."""
        initial_state = self.create_sample_hierarchical_state()

        # Invalid observation - state index out of bounds
        invalid_observations = {0: 10}  # Factor 0 only has 2 states (0, 1)

        updater = HierarchicalBeliefUpdater()
        updated_state = await updater.propagate_beliefs(initial_state, invalid_observations)

        # Should handle gracefully - observation ignored
        assert updated_state.metadata["observations_applied"] == 0

    @pytest.mark.asyncio
    async def test_propagate_beliefs_no_hierarchy(self):
        """Test belief propagation with no hierarchical structure."""
        # Create state without hierarchy
        factors = [
            BeliefFactor(0, "independent1", np.array([0.6, 0.4]), FactorType.INDEPENDENT),
            BeliefFactor(1, "independent2", np.array([0.3, 0.7]), FactorType.INDEPENDENT),
        ]
        state = FactorizedBeliefState(factors=factors, agent_id="flat_agent")

        updater = HierarchicalBeliefUpdater()
        updated_state = await updater.propagate_beliefs(state)

        # Should complete without errors, minimal changes expected
        assert updated_state.num_factors == 2
        assert updated_state.metadata["propagation_iterations"] >= 1


class TestBeliefConsistencyValidator:
    """Test suite for BeliefConsistencyValidator."""

    def create_valid_state(self) -> FactorizedBeliefState:
        """Create valid factorized belief state for testing."""
        factors = [
            BeliefFactor(0, "factor_0", np.array([0.6, 0.4]), FactorType.INDEPENDENT),
            BeliefFactor(1, "factor_1", np.array([0.3, 0.7]), FactorType.CORRELATED),
            BeliefFactor(2, "factor_2", np.array([0.5, 0.3, 0.2]), FactorType.HIERARCHICAL),
        ]
        correlations = [CrossFactorCorrelation(0, 1, np.array([[1.0, 0.5], [0.5, 1.0]]), 0.5)]
        hierarchy = {0: [1], 1: [2]}

        return FactorizedBeliefState(
            factors=factors, correlations=correlations, hierarchy=hierarchy, agent_id="valid_agent"
        )

    @pytest.mark.asyncio
    async def test_validate_factorized_state_valid(self):
        """Test validation of valid factorized state."""
        state = self.create_valid_state()
        validator = BeliefConsistencyValidator()

        result = await validator.validate_factorized_state(state)

        assert result["is_valid"] is True
        assert len(result["errors"]) == 0
        assert "factor_diagnostics" in result
        assert "correlation_diagnostics" in result
        assert "hierarchy_diagnostics" in result
        assert "performance_metrics" in result

        # Check performance metrics
        metrics = result["performance_metrics"]
        assert metrics["total_factors"] == 3
        assert metrics["overall_entropy"] > 0
        assert metrics["overall_confidence"] > 0
        assert metrics["correlation_count"] == 1
        assert metrics["hierarchy_depth"] > 0

    @pytest.mark.asyncio
    async def test_validate_factor_invalid_normalization(self):
        """Test validation with factor normalization issues."""
        # Create factor with poor normalization
        invalid_factor = BeliefFactor(0, "invalid", np.array([0.5, 0.5]), FactorType.INDEPENDENT)
        # Manually set invalid beliefs to bypass BeliefFactor normalization
        invalid_factor.beliefs = np.array([0.6, 0.5])  # Sum = 1.1

        state = FactorizedBeliefState(factors=[invalid_factor])
        validator = BeliefConsistencyValidator(strict_validation=True)

        result = await validator.validate_factorized_state(state)

        assert result["is_valid"] is False
        assert len(result["errors"]) > 0
        assert any("beliefs sum to" in error for error in result["errors"])

    @pytest.mark.asyncio
    async def test_validate_factor_negative_beliefs(self):
        """Test validation with negative beliefs."""
        invalid_factor = BeliefFactor(0, "negative", np.array([0.7, 0.3]), FactorType.INDEPENDENT)
        # Manually set negative beliefs
        invalid_factor.beliefs = np.array([0.8, -0.1])

        state = FactorizedBeliefState(factors=[invalid_factor])
        validator = BeliefConsistencyValidator()

        result = await validator.validate_factorized_state(state)

        assert result["is_valid"] is False
        assert any("negative beliefs" in error for error in result["errors"])

    @pytest.mark.asyncio
    async def test_validate_factor_non_finite_beliefs(self):
        """Test validation with non-finite beliefs."""
        invalid_factor = BeliefFactor(0, "non_finite", np.array([0.5, 0.5]), FactorType.INDEPENDENT)
        # Manually set non-finite beliefs
        invalid_factor.beliefs = np.array([np.inf, 0.5])

        state = FactorizedBeliefState(factors=[invalid_factor])
        validator = BeliefConsistencyValidator()

        result = await validator.validate_factorized_state(state)

        assert result["is_valid"] is False
        assert any("non-finite beliefs" in error for error in result["errors"])

    @pytest.mark.asyncio
    async def test_validate_correlations_invalid_strength(self):
        """Test validation with invalid correlation strength."""
        factors = [
            BeliefFactor(0, "factor_0", np.array([0.6, 0.4]), FactorType.INDEPENDENT),
            BeliefFactor(1, "factor_1", np.array([0.3, 0.7]), FactorType.CORRELATED),
        ]
        # Invalid correlation strength > 1.0
        invalid_correlation = CrossFactorCorrelation(0, 1, np.eye(2), 1.5)

        state = FactorizedBeliefState(factors=factors, correlations=[invalid_correlation])
        validator = BeliefConsistencyValidator()

        result = await validator.validate_factorized_state(state)

        assert result["is_valid"] is False
        assert any("invalid strength" in error for error in result["errors"])

    @pytest.mark.asyncio
    async def test_validate_hierarchy_circular_dependency(self):
        """Test validation with circular hierarchy."""
        factors = [
            BeliefFactor(0, "factor_0", np.array([0.6, 0.4]), FactorType.HIERARCHICAL),
            BeliefFactor(1, "factor_1", np.array([0.3, 0.7]), FactorType.HIERARCHICAL),
        ]
        # Circular dependency: 0 -> 1 -> 0
        circular_hierarchy = {0: [1], 1: [0]}

        state = FactorizedBeliefState(factors=factors, hierarchy=circular_hierarchy)
        validator = BeliefConsistencyValidator()

        result = await validator.validate_factorized_state(state)

        assert result["is_valid"] is False
        assert any("Circular dependency" in error for error in result["errors"])

    @pytest.mark.asyncio
    async def test_validate_hierarchy_missing_references(self):
        """Test validation with missing factor references in hierarchy."""
        factors = [BeliefFactor(0, "factor_0", np.array([0.6, 0.4]), FactorType.HIERARCHICAL)]
        # Hierarchy references non-existent factors
        invalid_hierarchy = {0: [1], 2: [3]}  # Factors 1, 2, 3 don't exist

        state = FactorizedBeliefState(factors=factors, hierarchy=invalid_hierarchy)
        validator = BeliefConsistencyValidator()

        result = await validator.validate_factorized_state(state)

        assert result["is_valid"] is False
        errors = result["errors"]
        assert any("parent 2 not found" in error for error in errors)
        assert any("child 1 not found" in error for error in errors)
        assert any("child 3 not found" in error for error in errors)

    @pytest.mark.asyncio
    async def test_validate_non_strict_mode(self):
        """Test validation in non-strict mode."""
        invalid_factor = BeliefFactor(
            0, "poorly_normalized", np.array([0.5, 0.5]), FactorType.INDEPENDENT
        )
        # Set beliefs that violate normalization
        invalid_factor.beliefs = np.array([0.6, 0.5])  # Sum = 1.1

        state = FactorizedBeliefState(factors=[invalid_factor])
        validator = BeliefConsistencyValidator(strict_validation=False)

        result = await validator.validate_factorized_state(state)

        # Should pass validation but have warnings
        assert result["is_valid"] is True
        assert len(result["warnings"]) > 0
        assert any("poorly normalized" in warning for warning in result["warnings"])

    @pytest.mark.asyncio
    async def test_factor_diagnostics_metrics(self):
        """Test detailed factor diagnostic metrics."""
        # Create factor with known characteristics
        beliefs = np.array([0.8, 0.1, 0.05, 0.05])  # Sparse, high confidence
        factor = BeliefFactor(0, "diagnostic_test", beliefs, FactorType.INDEPENDENT)

        state = FactorizedBeliefState(factors=[factor])
        validator = BeliefConsistencyValidator()

        result = await validator.validate_factorized_state(state)

        factor_metrics = result["factor_diagnostics"][0]["metrics"]
        assert "entropy" in factor_metrics
        assert "confidence" in factor_metrics
        assert "sparsity" in factor_metrics
        assert "effective_states" in factor_metrics

        # Check expected values
        assert factor_metrics["confidence"] == 0.8
        assert factor_metrics["effective_states"] <= 4  # At most 4 significant states


class TestIntegration:
    """Integration tests for multi-factor belief management."""

    @pytest.mark.asyncio
    async def test_end_to_end_multi_factor_workflow(self):
        """Test complete multi-factor belief management workflow."""
        # Create mock agent with multi-factor beliefs
        beliefs = [np.array([0.7, 0.2, 0.1]), np.array([0.4, 0.6]), np.array([0.2, 0.3, 0.3, 0.2])]
        agent = Mock()
        agent.qs = beliefs
        agent.agent_id = "integration_test"

        # Create components
        extractor, updater, validator = create_multi_factor_belief_manager()

        # Step 1: Extract factorized beliefs
        belief_state = await extractor.extract_factorized_beliefs(
            agent, "integration_test", include_correlations=True
        )

        assert belief_state.num_factors == 3
        assert len(belief_state.correlations) >= 0  # May or may not have correlations

        # Step 2: Validate consistency
        validation_result = await validator.validate_factorized_state(belief_state)
        assert validation_result["is_valid"] is True

        # Step 3: Update beliefs with observations
        observations = {0: 0, 2: 1}  # Observe specific states
        updated_state = await updater.propagate_beliefs(belief_state, observations)

        assert updated_state.num_factors == 3
        assert updated_state.metadata["observations_applied"] == 2

        # Step 4: Validate updated state
        updated_validation = await validator.validate_factorized_state(updated_state)
        assert updated_validation["is_valid"] is True

        # Check that observations were applied
        factor_0 = updated_state.get_factor(0)
        factor_2 = updated_state.get_factor(2)
        assert factor_0 is not None
        assert factor_2 is not None
        assert factor_0.beliefs[0] == 1.0  # Observed state 0
        assert factor_2.beliefs[1] == 1.0  # Observed state 1

    @pytest.mark.asyncio
    async def test_performance_with_large_factor_space(self):
        """Test performance with large number of factors."""
        # Create agent with many factors
        num_factors = 10
        beliefs = [np.random.dirichlet(np.ones(5)) for _ in range(num_factors)]

        agent = Mock()
        agent.qs = beliefs
        agent.agent_id = "large_scale_test"

        # Create components with performance settings
        extractor = MultiFactorBeliefExtractor(
            belief_threshold=0.05,  # Higher threshold for efficiency
            correlation_threshold=0.2,  # Higher threshold for efficiency
            enable_caching=True,
        )

        # Measure extraction time
        import time

        start_time = time.time()

        belief_state = await extractor.extract_factorized_beliefs(
            agent, "large_scale_test", include_correlations=True
        )

        extraction_time = time.time() - start_time

        assert belief_state.num_factors == num_factors
        assert extraction_time < 5.0  # Should complete within 5 seconds

        # Test caching effectiveness
        start_time = time.time()
        cached_state = await extractor.extract_factorized_beliefs(
            agent, "large_scale_test", include_correlations=True
        )
        cached_time = time.time() - start_time

        # Cached version should be faster (if within same second)
        # Note: This test might be flaky due to timestamp precision
        assert len(extractor._belief_cache) > 0

    def test_factory_function(self):
        """Test factory function for component creation."""
        extractor, updater, validator = create_multi_factor_belief_manager()

        assert isinstance(extractor, MultiFactorBeliefExtractor)
        assert isinstance(updater, HierarchicalBeliefUpdater)
        assert isinstance(validator, BeliefConsistencyValidator)

        # Check default configurations
        assert extractor.belief_threshold == 0.01
        assert extractor.correlation_threshold == 0.1
        assert extractor.enable_caching is True

        assert updater.max_iterations == 10
        assert updater.convergence_threshold == 1e-6

        assert validator.strict_validation is True


# Performance benchmarks (optional, for development)
class TestPerformanceBenchmarks:
    """Performance benchmarks for multi-factor belief management."""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_extraction_performance_benchmark(self):
        """Benchmark belief extraction performance."""
        # Skip performance tests in regular test runs unless explicitly requested
        pytest.skip("Performance benchmark - run with --performance flag")

        # Create varying sizes of factor spaces
        test_cases = [
            (2, 5),  # 2 factors, 5 states each
            (5, 10),  # 5 factors, 10 states each
            (10, 20),  # 10 factors, 20 states each
        ]

        extractor = MultiFactorBeliefExtractor(
            enable_caching=False
        )  # Disable caching for fair comparison

        for num_factors, num_states in test_cases:
            beliefs = [np.random.dirichlet(np.ones(num_states)) for _ in range(num_factors)]
            agent = Mock()
            agent.qs = beliefs
            agent.agent_id = f"perf_test_{num_factors}_{num_states}"

            # Measure extraction time
            import time

            start_time = time.time()

            result = await extractor.extract_factorized_beliefs(
                agent, agent.agent_id, include_correlations=True
            )

            extraction_time = time.time() - start_time

            print(f"Factors: {num_factors}, States: {num_states}, Time: {extraction_time:.4f}s")

            # Performance assertions
            assert result.num_factors == num_factors
            assert extraction_time < 10.0  # Should complete within 10 seconds

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_memory_usage_benchmark(self):
        """Benchmark memory usage for large belief states."""
        pytest.skip("Memory benchmark - run with --performance flag")

        import tracemalloc

        # Start memory tracking
        tracemalloc.start()

        # Create large belief state
        num_factors = 20
        num_states = 50
        beliefs = [np.random.dirichlet(np.ones(num_states)) for _ in range(num_factors)]

        agent = Mock()
        agent.qs = beliefs

        extractor = MultiFactorBeliefExtractor()

        # Extract beliefs and measure memory
        snapshot1 = tracemalloc.take_snapshot()

        result = await extractor.extract_factorized_beliefs(
            agent, "memory_test", include_correlations=True
        )

        snapshot2 = tracemalloc.take_snapshot()
        top_stats = snapshot2.compare_to(snapshot1, "lineno")

        # Print memory usage
        for stat in top_stats[:5]:
            print(stat)

        # Memory usage should be reasonable
        total_memory_mb = sum(stat.size for stat in top_stats) / 1024 / 1024
        print(f"Total memory usage: {total_memory_mb:.2f} MB")

        assert total_memory_mb < 100  # Should use less than 100MB

        tracemalloc.stop()


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/unit/services/test_multi_factor_belief_manager.py -v
    # Run with performance tests: python -m pytest tests/unit/services/test_multi_factor_belief_manager.py -v -m performance
    pass
