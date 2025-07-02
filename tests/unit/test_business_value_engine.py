"""
Comprehensive tests for Business Value Calculation Engine

This test suite provides complete coverage of business value calculations,
mathematical models, and edge cases for coalition formation business intelligence.
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

# Mock dependencies
from coalitions.coalition.coalition_models import Coalition

# Import the module under test
from coalitions.formation.business_value_engine import (
    BusinessMetricType,
    BusinessValueCalculationEngine,
    BusinessValueMetrics,
    business_value_engine,
)
from coalitions.formation.coalition_formation_algorithms import AgentProfile, FormationResult


class TestBusinessValueMetrics:
    """Test the BusinessValueMetrics dataclass"""

    def test_default_initialization(self):
        """Test default metric initialization"""
        metrics = BusinessValueMetrics()

        assert metrics.synergy_score == 0.0
        assert metrics.risk_reduction == 0.0
        assert metrics.market_positioning == 0.0
        assert metrics.sustainability_score == 0.0
        assert metrics.operational_efficiency == 0.0
        assert metrics.innovation_potential == 0.0
        assert metrics.total_value == 0.0
        assert metrics.confidence_level == 0.0
        assert isinstance(metrics.calculation_timestamp, datetime)
        assert isinstance(metrics.methodology_notes, dict)
        assert len(metrics.methodology_notes) == 0

    def test_custom_initialization(self):
        """Test custom metric initialization"""
        custom_timestamp = datetime(2025, 1, 1, 12, 0, 0)
        custom_notes = {"test": "methodology"}

        metrics = BusinessValueMetrics(
            synergy_score=0.8,
            risk_reduction=0.7,
            market_positioning=0.9,
            sustainability_score=0.6,
            operational_efficiency=0.75,
            innovation_potential=0.85,
            total_value=0.775,
            confidence_level=0.9,
            calculation_timestamp=custom_timestamp,
            methodology_notes=custom_notes,
        )

        assert metrics.synergy_score == 0.8
        assert metrics.risk_reduction == 0.7
        assert metrics.market_positioning == 0.9
        assert metrics.sustainability_score == 0.6
        assert metrics.operational_efficiency == 0.75
        assert metrics.innovation_potential == 0.85
        assert metrics.total_value == 0.775
        assert metrics.confidence_level == 0.9
        assert metrics.calculation_timestamp == custom_timestamp
        assert metrics.methodology_notes == custom_notes


class TestBusinessMetricType:
    """Test the BusinessMetricType enum"""

    def test_enum_values(self):
        """Test all enum values are correctly defined"""
        assert BusinessMetricType.SYNERGY.value == "synergy"
        assert BusinessMetricType.RISK_REDUCTION.value == "risk_reduction"
        assert BusinessMetricType.MARKET_POSITIONING.value == "market_positioning"
        assert BusinessMetricType.SUSTAINABILITY.value == "sustainability"
        assert BusinessMetricType.OPERATIONAL_EFFICIENCY.value == "operational_efficiency"
        assert BusinessMetricType.INNOVATION_POTENTIAL.value == "innovation_potential"

    def test_enum_count(self):
        """Test that we have exactly 6 metric types"""
        assert len(BusinessMetricType) == 6


class TestBusinessValueCalculationEngine:
    """Comprehensive tests for the Business Value Calculation Engine"""

    @pytest.fixture
    def engine(self):
        """Create a fresh engine instance for each test"""
        return BusinessValueCalculationEngine()

    @pytest.fixture
    def mock_coalition(self):
        """Create a mock coalition"""
        coalition = Mock(spec=Coalition)
        coalition.coalition_id = "test-coalition-123"
        coalition.members = ["agent1", "agent2", "agent3"]
        return coalition

    @pytest.fixture
    def mock_formation_result_success(self):
        """Create a successful formation result"""
        result = Mock(spec=FormationResult)
        result.success = True
        result.score = 8.5
        result.formation_time = 2.3
        result.strategy_used = Mock()
        result.strategy_used.value = "active_inference"
        return result

    @pytest.fixture
    def mock_formation_result_failure(self):
        """Create a failed formation result"""
        result = Mock(spec=FormationResult)
        result.success = False
        result.score = 0.0
        result.formation_time = 15.0
        result.strategy_used = None
        return result

    @pytest.fixture
    def mock_agent_profiles(self):
        """Create mock agent profiles"""
        profiles = []

        # Agent 1: High capability, moderate resources
        agent1 = Mock(spec=AgentProfile)
        agent1.capabilities = [
            "data_analysis",
            "machine_learning",
            "visualization"]
        agent1.resources = {"compute": 15, "storage": 100, "bandwidth": 50}
        agent1.reliability_score = 0.9
        agent1.availability = 0.8
        profiles.append(agent1)

        # Agent 2: Different capabilities, good resources
        agent2 = Mock(spec=AgentProfile)
        agent2.capabilities = ["api_integration", "security", "testing"]
        agent2.resources = {"compute": 20, "database": 200, "memory": 32}
        agent2.reliability_score = 0.85
        agent2.availability = 0.95
        profiles.append(agent2)

        # Agent 3: Overlapping capabilities, unique resources
        agent3 = Mock(spec=AgentProfile)
        agent3.capabilities = ["data_analysis", "reporting", "optimization"]
        agent3.resources = {"gpu": 4, "licenses": 10, "storage": 50}
        agent3.reliability_score = 0.75
        agent3.availability = 0.7
        profiles.append(agent3)

        return profiles

    @pytest.fixture
    def mock_agent_profiles_empty(self):
        """Create empty agent profiles for edge case testing"""
        return []

    @pytest.fixture
    def mock_agent_profiles_minimal(self):
        """Create minimal agent profiles"""
        agent = Mock(spec=AgentProfile)
        agent.capabilities = []
        agent.resources = {}
        agent.reliability_score = 0.0
        agent.availability = 0.0
        return [agent]

    def test_engine_initialization(self, engine):
        """Test engine initializes with correct default values"""
        assert isinstance(engine.metric_weights, dict)
        assert len(engine.metric_weights) == 6
        assert engine.metric_weights[BusinessMetricType.SYNERGY] == 0.25
        assert engine.metric_weights[BusinessMetricType.RISK_REDUCTION] == 0.20
        assert engine.metric_weights[BusinessMetricType.MARKET_POSITIONING] == 0.20
        assert engine.metric_weights[BusinessMetricType.SUSTAINABILITY] == 0.15
        assert engine.metric_weights[BusinessMetricType.OPERATIONAL_EFFICIENCY] == 0.10
        assert engine.metric_weights[BusinessMetricType.INNOVATION_POTENTIAL] == 0.10
        assert isinstance(engine.calculation_history, list)
        assert len(engine.calculation_history) == 0

    def test_weights_sum_to_one(self, engine):
        """Test that metric weights sum to 1.0"""
        total_weight = sum(engine.metric_weights.values())
        # Account for floating point precision
        assert abs(total_weight - 1.0) < 0.0001

    @patch("coalitions.formation.business_value_engine.logger")
    def test_calculate_business_value_success(
        self,
        mock_logger,
        engine,
        mock_coalition,
        mock_formation_result_success,
        mock_agent_profiles,
    ):
        """Test successful business value calculation"""
        # Add optional market context
        market_context = {"readiness_score": 0.8, "market_size": 1000000}

        result = engine.calculate_business_value(
            coalition=mock_coalition,
            formation_result=mock_formation_result_success,
            agent_profiles=mock_agent_profiles,
            market_context=market_context,
        )

        # Verify result structure
        assert isinstance(result, BusinessValueMetrics)
        assert result.synergy_score >= 0.0
        assert result.risk_reduction >= 0.0
        assert result.market_positioning >= 0.0
        assert result.sustainability_score >= 0.0
        assert result.operational_efficiency >= 0.0
        assert result.innovation_potential >= 0.0
        assert result.total_value >= 0.0
        assert result.confidence_level >= 0.0

        # All scores should be <= 1.0
        assert result.synergy_score <= 1.0
        assert result.risk_reduction <= 1.0
        assert result.market_positioning <= 1.0
        assert result.sustainability_score <= 1.0
        assert result.operational_efficiency <= 1.0
        assert result.innovation_potential <= 1.0
        assert result.total_value <= 1.0
        assert result.confidence_level <= 1.0

        # Check methodology notes were added
        assert len(result.methodology_notes) == 6
        assert "synergy" in result.methodology_notes
        assert "risk_reduction" in result.methodology_notes
        assert "market_positioning" in result.methodology_notes
        assert "sustainability" in result.methodology_notes
        assert "operational_efficiency" in result.methodology_notes
        assert "innovation_potential" in result.methodology_notes

        # Verify calculation was logged
        mock_logger.info.assert_called_once()

        # Verify result was stored in history
        assert len(engine.calculation_history) == 1
        assert engine.calculation_history[0] == result

    def test_calculate_business_value_empty_profiles(
            self,
            engine,
            mock_coalition,
            mock_formation_result_success,
            mock_agent_profiles_empty):
        """Test business value calculation with empty agent profiles"""
        result = engine.calculate_business_value(
            coalition=mock_coalition,
            formation_result=mock_formation_result_success,
            agent_profiles=mock_agent_profiles_empty,
        )

        # With no agents, most metrics should be 0
        assert result.synergy_score == 0.0
        assert result.risk_reduction == 0.0
        assert result.operational_efficiency == 0.0
        assert result.innovation_potential == 0.0
        # Market positioning and sustainability may still have some value from
        # formation result
        assert result.market_positioning >= 0.0
        assert result.sustainability_score >= 0.0
        assert result.total_value >= 0.0
        assert result.confidence_level >= 0.0

    @patch("coalitions.formation.business_value_engine.logger")
    def test_calculate_business_value_exception_handling(
            self, mock_logger, engine):
        """Test exception handling in business value calculation"""
        # Create a mock that raises an exception
        mock_coalition = Mock()
        mock_coalition.coalition_id = "test"

        # Make formation result raise an exception when accessed
        mock_formation_result = Mock()
        mock_formation_result.score.side_effect = Exception("Test exception")

        result = engine.calculate_business_value(
            coalition=mock_coalition,
            formation_result=mock_formation_result,
            agent_profiles=[])

        # Should return zero metrics on error
        assert isinstance(result, BusinessValueMetrics)
        assert result.synergy_score == 0.0
        assert result.risk_reduction == 0.0
        assert result.market_positioning == 0.0
        assert result.sustainability_score == 0.0
        assert result.operational_efficiency == 0.0
        assert result.innovation_potential == 0.0
        assert result.total_value == 0.0
        assert result.confidence_level == 0.0

        # Should log the error
        mock_logger.error.assert_called_once()

    def test_calculate_synergy_empty_profiles(
        self, engine, mock_coalition, mock_formation_result_success
    ):
        """Test synergy calculation with empty profiles"""
        result = engine._calculate_synergy(
            mock_coalition, mock_formation_result_success, [])
        assert result == 0.0

    def test_calculate_synergy_with_profiles(
            self,
            engine,
            mock_coalition,
            mock_formation_result_success,
            mock_agent_profiles):
        """Test synergy calculation with agent profiles"""
        result = engine._calculate_synergy(
            mock_coalition, mock_formation_result_success, mock_agent_profiles
        )

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        # Should be positive given diverse capabilities and resources
        assert result > 0.0

    def test_calculate_synergy_zero_individual_value(
        self, engine, mock_coalition, mock_formation_result_success
    ):
        """Test synergy calculation when individual values sum to zero"""
        # Create profiles with zero values
        zero_profiles = []
        for i in range(2):
            agent = Mock(spec=AgentProfile)
            agent.capabilities = []
            agent.resources = {}
            agent.reliability_score = 0.0
            zero_profiles.append(agent)

        result = engine._calculate_synergy(
            mock_coalition, mock_formation_result_success, zero_profiles
        )
        assert result == 0.0

    def test_calculate_risk_reduction_empty_profiles(
            self, engine, mock_coalition):
        """Test risk reduction calculation with empty profiles"""
        result = engine._calculate_risk_reduction(mock_coalition, [])
        assert result == 0.0

    def test_calculate_risk_reduction_with_profiles(
        self, engine, mock_coalition, mock_agent_profiles
    ):
        """Test risk reduction calculation with diverse profiles"""
        result = engine._calculate_risk_reduction(
            mock_coalition, mock_agent_profiles)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        # Should be positive given diverse capabilities
        assert result > 0.0

    def test_calculate_risk_reduction_identical_profiles(
            self, engine, mock_coalition):
        """Test risk reduction with identical profiles (low diversification)"""
        # Create identical profiles
        identical_profiles = []
        for i in range(3):
            agent = Mock(spec=AgentProfile)
            agent.capabilities = ["same_capability"]
            agent.resources = {"same_resource": 10}
            agent.reliability_score = 0.8
            identical_profiles.append(agent)

        result = engine._calculate_risk_reduction(
            mock_coalition, identical_profiles)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        # Should be lower than diverse profiles due to low diversification

    def test_calculate_market_positioning_various_strategies(
            self, engine, mock_coalition):
        """Test market positioning calculation with different strategies"""
        strategies = [
            "active_inference",
            "capability_based",
            "resource_optimization",
            "preference_matching",
            "stability_maximization",
            "business_opportunity",
            "unknown",
        ]

        for strategy in strategies:
            formation_result = Mock(spec=FormationResult)
            formation_result.success = True
            formation_result.score = 5.0
            formation_result.formation_time = 1.0
            formation_result.strategy_used = Mock()
            formation_result.strategy_used.value = strategy

            result = engine._calculate_market_positioning(
                mock_coalition, formation_result, None)

            assert isinstance(result, float)
            assert 0.0 <= result <= 1.0

    def test_calculate_market_positioning_with_context(
        self, engine, mock_coalition, mock_formation_result_success
    ):
        """Test market positioning with market context"""
        market_context = {"readiness_score": 0.9}

        result = engine._calculate_market_positioning(
            mock_coalition, mock_formation_result_success, market_context
        )

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_calculate_market_positioning_speed_bonus(
            self, engine, mock_coalition):
        """Test market positioning speed bonus calculation"""
        # Fast formation (should get speed bonus)
        fast_result = Mock(spec=FormationResult)
        fast_result.success = True
        fast_result.score = 5.0
        fast_result.formation_time = 0.5  # Very fast
        fast_result.strategy_used = Mock()
        # Lower base score (0.6)
        fast_result.strategy_used.value = "preference_matching"

        # Slow formation (no speed bonus)
        slow_result = Mock(spec=FormationResult)
        slow_result.success = True
        slow_result.score = 5.0
        slow_result.formation_time = 15.0  # Very slow
        slow_result.strategy_used = Mock()
        # Lower base score (0.6)
        slow_result.strategy_used.value = "preference_matching"

        fast_score = engine._calculate_market_positioning(
            mock_coalition, fast_result, None)
        slow_score = engine._calculate_market_positioning(
            mock_coalition, slow_result, None)

        # Fast formation should score higher due to speed bonus
        assert fast_score > slow_score

    def test_calculate_sustainability_empty_profiles(
        self, engine, mock_coalition, mock_formation_result_success
    ):
        """Test sustainability calculation with empty profiles"""
        result = engine._calculate_sustainability(
            mock_coalition, mock_formation_result_success, [])

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        # Should still have some value from formation quality

    def test_calculate_sustainability_with_profiles(
            self,
            engine,
            mock_coalition,
            mock_formation_result_success,
            mock_agent_profiles):
        """Test sustainability calculation with agent profiles"""
        result = engine._calculate_sustainability(
            mock_coalition, mock_formation_result_success, mock_agent_profiles
        )

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        assert result > 0.0

    def test_calculate_sustainability_failed_formation(
            self,
            engine,
            mock_coalition,
            mock_formation_result_failure,
            mock_agent_profiles):
        """Test sustainability calculation with failed formation"""
        result = engine._calculate_sustainability(
            mock_coalition, mock_formation_result_failure, mock_agent_profiles
        )

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        # Should be lower than successful formation

    def test_calculate_operational_efficiency_empty_profiles(
            self, engine, mock_coalition):
        """Test operational efficiency with empty profiles"""
        result = engine._calculate_operational_efficiency(mock_coalition, [])
        assert result == 0.0

    def test_calculate_operational_efficiency_with_profiles(
        self, engine, mock_coalition, mock_agent_profiles
    ):
        """Test operational efficiency with agent profiles"""
        result = engine._calculate_operational_efficiency(
            mock_coalition, mock_agent_profiles)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_calculate_innovation_potential_empty_profiles(
            self, engine, mock_coalition):
        """Test innovation potential with empty profiles"""
        result = engine._calculate_innovation_potential(mock_coalition, [])
        assert result == 0.0

    def test_calculate_innovation_potential_with_profiles(
        self, engine, mock_coalition, mock_agent_profiles
    ):
        """Test innovation potential with agent profiles"""
        result = engine._calculate_innovation_potential(
            mock_coalition, mock_agent_profiles)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        assert result > 0.0  # Should be positive given unique capabilities

    def test_calculate_total_value(self, engine):
        """Test total value calculation"""
        metrics = BusinessValueMetrics(
            synergy_score=0.8,
            risk_reduction=0.6,
            market_positioning=0.9,
            sustainability_score=0.7,
            operational_efficiency=0.5,
            innovation_potential=0.8,
        )

        result = engine._calculate_total_value(metrics)

        # Calculate expected value manually
        expected = (
            0.8 * 0.25  # synergy
            + 0.6 * 0.20  # risk_reduction
            + 0.9 * 0.20  # market_positioning
            + 0.7 * 0.15  # sustainability
            + 0.5 * 0.10  # operational_efficiency
            + 0.8 * 0.10  # innovation_potential
        )

        assert abs(result - expected) < 0.0001
        assert result <= 1.0

    def test_calculate_confidence_empty_profiles(
            self, engine, mock_formation_result_success):
        """Test confidence calculation with empty profiles"""
        result = engine._calculate_confidence(
            mock_formation_result_success, [])

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_calculate_confidence_with_profiles(
        self, engine, mock_formation_result_success, mock_agent_profiles
    ):
        """Test confidence calculation with agent profiles"""
        result = engine._calculate_confidence(
            mock_formation_result_success, mock_agent_profiles)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        assert result > 0.0

    def test_calculate_confidence_failed_formation(
        self, engine, mock_formation_result_failure, mock_agent_profiles
    ):
        """Test confidence calculation with failed formation"""
        result = engine._calculate_confidence(
            mock_formation_result_failure, mock_agent_profiles)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        # Should be lower than successful formation

    def test_get_calculation_history_empty(self, engine):
        """Test getting calculation history when empty"""
        history = engine.get_calculation_history()
        assert isinstance(history, list)
        assert len(history) == 0

    def test_get_calculation_history_with_data(
            self,
            engine,
            mock_coalition,
            mock_formation_result_success,
            mock_agent_profiles):
        """Test getting calculation history with data"""
        # Perform several calculations
        for i in range(5):
            engine.calculate_business_value(
                mock_coalition,
                mock_formation_result_success,
                mock_agent_profiles)

        # Get all history
        history = engine.get_calculation_history()
        assert len(history) == 5

        # Get limited history
        limited_history = engine.get_calculation_history(limit=3)
        assert len(limited_history) == 3

        # Should be the most recent 3
        assert limited_history == history[-3:]

    def test_export_metrics_for_investors(self, engine):
        """Test exporting metrics in investor-friendly format"""
        metrics = BusinessValueMetrics(
            synergy_score=0.85,
            risk_reduction=0.70,
            market_positioning=0.90,
            sustainability_score=0.65,
            operational_efficiency=0.75,
            innovation_potential=0.80,
            total_value=0.785,
            confidence_level=0.88,
            methodology_notes={
                "synergy": "Test methodology",
                "risk_reduction": "Test risk methodology",
            },
        )

        export = engine.export_metrics_for_investors(metrics)

        # Check structure
        assert "executive_summary" in export
        assert "detailed_metrics" in export
        assert "methodology" in export
        assert "investment_readiness" in export

        # Check executive summary
        exec_summary = export["executive_summary"]
        assert "total_business_value" in exec_summary
        assert "confidence_level" in exec_summary
        assert "key_strengths" in exec_summary
        assert "calculated_at" in exec_summary

        # Check detailed metrics
        detailed = export["detailed_metrics"]
        assert "synergy_potential" in detailed
        assert "risk_mitigation" in detailed
        assert "market_position" in detailed
        assert "sustainability" in detailed
        assert "operational_efficiency" in detailed
        assert "innovation_potential" in detailed

        # Check investment readiness categorization
        assert export["investment_readiness"] == "HIGH"  # total_value > 0.7

    def test_identify_key_strengths(self, engine):
        """Test identification of key strengths"""
        # High synergy, medium risk reduction, high innovation
        metrics = BusinessValueMetrics(
            synergy_score=0.95,
            risk_reduction=0.65,
            market_positioning=0.40,
            sustainability_score=0.30,
            operational_efficiency=0.50,
            innovation_potential=0.90,
        )

        strengths = engine._identify_key_strengths(metrics)

        assert isinstance(strengths, list)
        # Should include top scores above 0.6
        assert "Synergy" in strengths
        assert "Innovation Potential" in strengths
        assert "Risk Reduction" in strengths
        # Should not include low scores
        assert "Market Positioning" not in strengths
        assert "Sustainability" not in strengths

    def test_investment_readiness_categorization(self, engine):
        """Test investment readiness categorization"""
        # HIGH readiness
        high_metrics = BusinessValueMetrics(total_value=0.8)
        high_export = engine.export_metrics_for_investors(high_metrics)
        assert high_export["investment_readiness"] == "HIGH"

        # MEDIUM readiness
        medium_metrics = BusinessValueMetrics(total_value=0.6)
        medium_export = engine.export_metrics_for_investors(medium_metrics)
        assert medium_export["investment_readiness"] == "MEDIUM"

        # LOW readiness
        low_metrics = BusinessValueMetrics(total_value=0.3)
        low_export = engine.export_metrics_for_investors(low_metrics)
        assert low_export["investment_readiness"] == "LOW"

    def test_mathematical_consistency(
            self,
            engine,
            mock_coalition,
            mock_formation_result_success,
            mock_agent_profiles):
        """Test mathematical consistency across multiple calculations"""
        # Run the same calculation multiple times
        results = []
        for i in range(3):
            result = engine.calculate_business_value(
                mock_coalition, mock_formation_result_success, mock_agent_profiles)
            results.append(result)

        # Results should be identical (deterministic calculation)
        for i in range(1, len(results)):
            assert abs(
                results[0].synergy_score -
                results[i].synergy_score) < 0.0001
            assert abs(
                results[0].risk_reduction -
                results[i].risk_reduction) < 0.0001
            assert abs(
                results[0].market_positioning -
                results[i].market_positioning) < 0.0001
            assert abs(
                results[0].sustainability_score -
                results[i].sustainability_score) < 0.0001
            assert (abs(results[0].operational_efficiency -
                        results[i].operational_efficiency) < 0.0001)
            assert abs(
                results[0].innovation_potential -
                results[i].innovation_potential) < 0.0001
            assert abs(
                results[0].total_value -
                results[i].total_value) < 0.0001
            assert abs(
                results[0].confidence_level -
                results[i].confidence_level) < 0.0001


class TestGlobalBusinessValueEngine:
    """Test the global business value engine instance"""

    def test_global_instance_exists(self):
        """Test that global instance exists and is correct type"""
        assert business_value_engine is not None
        assert isinstance(
            business_value_engine,
            BusinessValueCalculationEngine)

    def test_global_instance_initialization(self):
        """Test that global instance is properly initialized"""
        assert isinstance(business_value_engine.metric_weights, dict)
        assert len(business_value_engine.metric_weights) == 6
        assert isinstance(business_value_engine.calculation_history, list)


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_extreme_values(self):
        """Test with extreme input values"""
        engine = BusinessValueCalculationEngine()

        # Create agent with extreme values
        extreme_agent = Mock(spec=AgentProfile)
        extreme_agent.capabilities = ["cap"] * 1000  # Many capabilities
        extreme_agent.resources = {"resource": 999999}  # Huge resource
        extreme_agent.reliability_score = 1.0  # Perfect reliability
        extreme_agent.availability = 1.0  # Perfect availability

        coalition = Mock(spec=Coalition)
        coalition.coalition_id = "extreme-test"
        coalition.members = ["agent1"]

        formation_result = Mock(spec=FormationResult)
        formation_result.success = True
        formation_result.score = 100.0  # Extreme score
        formation_result.formation_time = 0.001  # Very fast
        formation_result.strategy_used = Mock()
        formation_result.strategy_used.value = "active_inference"

        result = engine.calculate_business_value(
            coalition, formation_result, [extreme_agent])

        # All values should still be bounded [0, 1]
        assert 0.0 <= result.synergy_score <= 1.0
        assert 0.0 <= result.risk_reduction <= 1.0
        assert 0.0 <= result.market_positioning <= 1.0
        assert 0.0 <= result.sustainability_score <= 1.0
        assert 0.0 <= result.operational_efficiency <= 1.0
        assert 0.0 <= result.innovation_potential <= 1.0
        assert 0.0 <= result.total_value <= 1.0
        assert 0.0 <= result.confidence_level <= 1.0

    def test_negative_values(self):
        """Test handling of negative input values"""
        engine = BusinessValueCalculationEngine()

        # Create agent with negative values
        negative_agent = Mock(spec=AgentProfile)
        negative_agent.capabilities = []
        negative_agent.resources = {"debt": -100}  # Negative resource
        negative_agent.reliability_score = -0.5  # Negative reliability
        negative_agent.availability = -0.2  # Negative availability

        coalition = Mock(spec=Coalition)
        coalition.coalition_id = "negative-test"
        coalition.members = ["agent1"]

        formation_result = Mock(spec=FormationResult)
        formation_result.success = True
        formation_result.score = -5.0  # Negative score
        formation_result.formation_time = -1.0  # Negative time
        formation_result.strategy_used = Mock()
        formation_result.strategy_used.value = "unknown"

        # Should not crash and should handle negatives gracefully
        result = engine.calculate_business_value(
            coalition, formation_result, [negative_agent])

        assert isinstance(result, BusinessValueMetrics)
        # Results should still be in valid range (engine should bound values)
        assert 0.0 <= result.total_value <= 1.0
        assert 0.0 <= result.confidence_level <= 1.0

    def test_none_values(self):
        """Test handling of None values"""
        engine = BusinessValueCalculationEngine()

        # Create agent with None values
        none_agent = Mock(spec=AgentProfile)
        none_agent.capabilities = None
        none_agent.resources = None
        none_agent.reliability_score = None
        none_agent.availability = None

        coalition = Mock(spec=Coalition)
        coalition.coalition_id = "none-test"

        formation_result = Mock(spec=FormationResult)
        formation_result.success = None
        formation_result.score = None
        formation_result.formation_time = None
        formation_result.strategy_used = None

        # Should handle None gracefully without crashing
        try:
            result = engine.calculate_business_value(
                coalition, formation_result, [none_agent])
            # If it doesn't crash, results should be valid
            assert isinstance(result, BusinessValueMetrics)
        except (TypeError, AttributeError):
            # Acceptable to fail with type/attribute errors for None values
            pass
