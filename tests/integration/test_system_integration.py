import logging
import time
from pathlib import Path

import numpy as np
import psutil
import pytest

from infrastructure.deployment.export_validator import ExportValidator
from knowledge.knowledge_graph import PatternExtractor
from world.simulation.engine import SimulationEngine

"""System Integration Tests

Tests the complete FreeAgentics system including:
- Full simulation scenarios
- Multi-agent interactions
- Performance under load
- System stability
- Edge cases and error conditions
"""
logger = logging.getLogger(__name__)


class TestFullSystemIntegration:
    """Test full system integration scenarios."""

    @pytest.fixture
    async def simulation_engine(self):
        """Create simulation engine."""
        config = {
            "world": {"resolution": 5, "size": 100},
            "agents": {
                "count": 10,
                "distribution": {
                    "explorer": 4,
                    "merchant": 3,
                    "scholar": 2,
                    "guardian": 1,
                },
            },
            "simulation": {"max_cycles": 100, "time_step": 1.0},
        }
        engine = SimulationEngine(config)
        engine.initialize()
        return engine

    @pytest.mark.asyncio
    async def test_complete_simulation_lifecycle(self, simulation_engine):
        """Test complete simulation lifecycle from start to finish."""
        engine = simulation_engine
        await engine.start()
        target_cycles = 50
        while engine.current_cycle < target_cycles:
            await engine.step()
            if engine.current_cycle % 10 == 0:
                health = await engine.get_system_health()
                assert health["status"] == "healthy"
                assert health["agent_count"] == 10
                assert health["message_queue_size"] < 1000
        await engine.stop()
        stats = engine.get_statistics()
        assert stats["cycles_completed"] == target_cycles
        assert stats["agents_alive"] == 10
        assert stats["total_messages"] > 0
        assert stats["total_trades"] > 0
        assert stats["knowledge_nodes_created"] > 0

    @pytest.mark.asyncio
    async def test_multi_agent_ecosystem_dynamics(self, simulation_engine):
        """Test ecosystem dynamics with multiple agent types."""
        engine = simulation_engine
        await engine.start()
        ecosystem_metrics = {
            "resource_distribution": [],
            "agent_wealth": [],
            "knowledge_density": [],
            "trade_volume": [],
            "exploration_coverage": [],
        }
        for cycle in range(30):
            await engine.step()
            metrics = await engine.get_ecosystem_metrics()
            ecosystem_metrics["resource_distribution"].append(metrics["resource_gini_coefficient"])
            ecosystem_metrics["agent_wealth"].append(metrics["average_agent_wealth"])
            ecosystem_metrics["knowledge_density"].append(metrics["knowledge_nodes_per_agent"])
            ecosystem_metrics["trade_volume"].append(metrics["trades_this_cycle"])
            ecosystem_metrics["exploration_coverage"].append(metrics["explored_cells_percentage"])
        resource_trend = np.polyfit(
            range(len(ecosystem_metrics["resource_distribution"])),
            ecosystem_metrics["resource_distribution"],
            1,
        )[0]
        assert resource_trend < 0
        knowledge_trend = np.polyfit(
            range(len(ecosystem_metrics["knowledge_density"])),
            ecosystem_metrics["knowledge_density"],
            1,
        )[0]
        assert knowledge_trend > 0
        final_coverage = ecosystem_metrics["exploration_coverage"][-1]
        assert final_coverage > ecosystem_metrics["exploration_coverage"][0]

    @pytest.mark.asyncio
    async def test_emergent_social_structures(self, simulation_engine):
        """Test emergence of social structures and relationships."""
        engine = simulation_engine
        await engine.start()
        for _ in range(50):
            await engine.step()
        social_network = await engine.get_social_network()
        trade_clusters = social_network.get_trade_clusters()
        assert len(trade_clusters) > 0
        merchant_centrality = []
        for agent_id, centrality in social_network.get_centrality_scores().items():
            agent = engine.get_agent(agent_id)
            if agent.agent_class == "merchant":
                merchant_centrality.append(centrality)
        avg_merchant_centrality = np.mean(merchant_centrality)
        avg_overall_centrality = np.mean(list(social_network.get_centrality_scores().values()))
        assert avg_merchant_centrality > avg_overall_centrality
        knowledge_network = social_network.get_knowledge_sharing_network()
        scholar_connections = []
        for agent_id, connections in knowledge_network.items():
            agent = engine.get_agent(agent_id)
            if agent.agent_class == "scholar":
                scholar_connections.append(len(connections))
        assert np.mean(scholar_connections) > 2
        protection_groups = social_network.get_protection_alliances()
        guardian_in_alliance = False
        for group in protection_groups:
            for agent_id in group:
                agent = engine.get_agent(agent_id)
                if agent.agent_class == "guardian":
                    guardian_in_alliance = True
                    break
        assert guardian_in_alliance

    @pytest.mark.asyncio
    async def test_system_performance_under_load(self):
        """Test system performance with large number of agents."""
        config = {
            "world": {"resolution": 6, "size": 500},
            "agents": {
                "count": 100,
                "distribution": {
                    "explorer": 40,
                    "merchant": 30,
                    "scholar": 20,
                    "guardian": 10,
                },
            },
            "simulation": {"max_cycles": 20, "time_step": 1.0},
        }
        engine = SimulationEngine(config)
        engine.initialize()
        performance_metrics = {
            "cycle_times": [],
            "memory_usage": [],
            "cpu_usage": [],
            "message_latency": [],
        }
        process = psutil.Process()
        await engine.start()
        for cycle in range(10):
            start_time = time.time()
            cpu_before = process.cpu_percent()
            await engine.step()
            cycle_time = time.time() - start_time
            performance_metrics["cycle_times"].append(cycle_time)
            memory_mb = process.memory_info().rss / 1024 / 1024
            performance_metrics["memory_usage"].append(memory_mb)
            cpu_after = process.cpu_percent()
            performance_metrics["cpu_usage"].append((cpu_before + cpu_after) / 2)
            latency = await engine.get_average_message_latency()
            performance_metrics["message_latency"].append(latency)
        avg_cycle_time = np.mean(performance_metrics["cycle_times"])
        assert avg_cycle_time < 5.0
        max_memory = max(performance_metrics["memory_usage"])
        assert max_memory < 2048
        avg_latency = np.mean(performance_metrics["message_latency"])
        assert avg_latency < 100
        memory_growth = (
            performance_metrics["memory_usage"][-1] - performance_metrics["memory_usage"][0]
        )
        assert memory_growth < 100

    @pytest.mark.asyncio
    async def test_fault_tolerance_and_recovery(self, simulation_engine):
        """Test system fault tolerance and recovery."""
        engine = simulation_engine
        await engine.start()
        for _ in range(10):
            await engine.step()
        initial_state = await engine.get_state_snapshot()
        agent_to_fail = engine.get_agents()[0]
        agent_id = agent_to_fail.agent_id
        await engine.simulate_agent_failure(agent_id)
        await engine.step()
        assert engine.get_agent_count() == 9
        assert engine.is_healthy()
        await engine.simulate_communication_failure(duration=5)
        for _ in range(5):
            await engine.step()
        await engine.step()
        comm_health = await engine.get_communication_health()
        assert comm_health["status"] == "recovered"
        await engine.simulate_resource_depletion(severity=0.8)
        for _ in range(10):
            await engine.step()
        survival_rate = engine.get_survival_rate()
        assert survival_rate > 0.5
        final_health = await engine.get_system_health()
        assert final_health["status"] in ["healthy", "degraded"]

    @pytest.mark.asyncio
    async def test_learning_and_adaptation(self, simulation_engine):
        """Test system-wide learning and adaptation."""
        engine = simulation_engine
        await engine.start()
        pattern_extractor = PatternExtractor()
        engine.attach_pattern_extractor(pattern_extractor)
        adaptation_metrics = {
            "collective_knowledge": [],
            "behavior_diversity": [],
            "success_rates": [],
            "pattern_count": [],
        }
        for cycle in range(40):
            await engine.step()
            if cycle % 5 == 0:
                patterns = pattern_extractor.extract_patterns(engine.get_event_history())
                metrics = await engine.get_adaptation_metrics()
                adaptation_metrics["collective_knowledge"].append(metrics["total_knowledge_nodes"])
                adaptation_metrics["behavior_diversity"].append(metrics["behavior_entropy"])
                adaptation_metrics["success_rates"].append(metrics["average_goal_achievement"])
                adaptation_metrics["pattern_count"].append(len(patterns))
        knowledge_growth = (
            adaptation_metrics["collective_knowledge"][-1]
            - adaptation_metrics["collective_knowledge"][0]
        )
        assert knowledge_growth > 0
        success_improvement = (
            adaptation_metrics["success_rates"][-1] - adaptation_metrics["success_rates"][0]
        )
        assert success_improvement > 0
        assert adaptation_metrics["pattern_count"][-1] > 5

    @pytest.mark.asyncio
    async def test_scalability_limits(self):
        """Test system scalability limits."""
        agent_counts = [10, 50, 100, 200]
        scalability_results = []
        for count in agent_counts:
            config = {
                "world": {"resolution": 6, "size": count * 10},
                "agents": {"count": count},
                "simulation": {"max_cycles": 5},
            }
            try:
                engine = SimulationEngine(config)
                engine.initialize()
                start_time = time.time()
                await engine.start()
                for _ in range(5):
                    await engine.step()
                total_time = time.time() - start_time
                scalability_results.append(
                    {
                        "agent_count": count,
                        "total_time": total_time,
                        "time_per_agent": total_time / count,
                        "success": True,
                    }
                )
                await engine.stop()
            except Exception as e:
                scalability_results.append(
                    {"agent_count": count, "success": False, "error": str(e)}
                )
        successful_tests = [r for r in scalability_results if r["success"]]
        max_successful = max(r["agent_count"] for r in successful_tests)
        assert max_successful >= 100
        if len(successful_tests) >= 2:
            time_per_agent_small = successful_tests[0]["time_per_agent"]
            time_per_agent_large = successful_tests[-1]["time_per_agent"]
            assert time_per_agent_large < time_per_agent_small * 2


class TestEdgeCasesAndStress:
    """Test edge cases and stress scenarios."""

    @pytest.mark.asyncio
    async def test_resource_scarcity_scenario(self):
        """Test system behavior under extreme resource scarcity."""
        config = {
            "world": {"resolution": 5, "size": 50, "resource_density": 0.1},
            "agents": {"count": 20},
            "simulation": {"max_cycles": 50},
        }
        engine = SimulationEngine(config)
        engine.initialize()
        await engine.start()
        survival_timeline = []
        cooperation_events = []
        for cycle in range(30):
            await engine.step()
            alive_count = engine.get_alive_agent_count()
            survival_timeline.append(alive_count)
            events = engine.get_events_this_cycle()
            cooperation_count = sum(
                1 for e in events if e["type"] in ["trade", "resource_share", "alliance_formed"]
            )
            cooperation_events.append(cooperation_count)
        early_cooperation = np.mean(cooperation_events[:10])
        late_cooperation = np.mean(cooperation_events[-10:])
        assert late_cooperation > early_cooperation
        final_survival = survival_timeline[-1]
        assert final_survival > 5

    @pytest.mark.asyncio
    async def test_information_overload(self):
        """Test system behavior with information overload."""
        config = {
            "world": {"resolution": 5, "size": 50},
            "agents": {"count": 10, "communication_rate": 10.0},
            "simulation": {"max_cycles": 20},
        }
        engine = SimulationEngine(config)
        engine.initialize()
        await engine.start()
        message_stats = {
            "dropped_messages": 0,
            "processing_delays": [],
            "queue_sizes": [],
        }
        for _ in range(10):
            await engine.step()
            stats = await engine.get_message_system_stats()
            message_stats["dropped_messages"] += stats["dropped_count"]
            message_stats["processing_delays"].append(stats["avg_delay"])
            message_stats["queue_sizes"].append(stats["queue_size"])
        drop_rate = message_stats["dropped_messages"] / (
            sum(message_stats["queue_sizes"]) + message_stats["dropped_messages"]
        )
        assert drop_rate < 0.1
        late_delays = message_stats["processing_delays"][-5:]
        assert np.std(late_delays) < np.mean(late_delays) * 0.5

    @pytest.mark.asyncio
    async def test_rapid_environmental_changes(self):
        """Test adaptation to rapid environmental changes."""
        engine = SimulationEngine(
            {
                "world": {"resolution": 5, "size": 100},
                "agents": {"count": 20},
                "simulation": {"max_cycles": 50},
            }
        )
        engine.initialize()
        await engine.start()
        adaptation_scores = []
        conditions = [
            {"resource_multiplier": 1.0, "hazard_level": 0.1},
            {"resource_multiplier": 0.3, "hazard_level": 0.5},
            {"resource_multiplier": 2.0, "hazard_level": 0.2},
            {"resource_multiplier": 0.5, "hazard_level": 0.8},
        ]
        for condition in conditions:
            await engine.set_environmental_conditions(condition)
            pre_change_performance = await engine.get_average_agent_performance()
            for _ in range(10):
                await engine.step()
            post_change_performance = await engine.get_average_agent_performance()
            adaptation = post_change_performance / pre_change_performance
            adaptation_scores.append(adaptation)
        avg_adaptation = np.mean(adaptation_scores)
        assert avg_adaptation > 0.7
        assert adaptation_scores[-1] > adaptation_scores[0]


class TestExportAndDeployment:
    """Test export and deployment functionality."""

    @pytest.mark.asyncio
    async def test_agent_export_validation(self):
        """Test agent export and validation."""
        engine = SimulationEngine(
            {
                "world": {"resolution": 5, "size": 50},
                "agents": {"count": 1},
                "simulation": {"max_cycles": 20},
            }
        )
        engine.initialize()
        await engine.start()
        for _ in range(20):
            await engine.step()
        agent = engine.get_agents()[0]
        export_path = Path("/tmp/test_agent_export")
        success = await engine.export_agent(agent.agent_id, export_path)
        assert success
        validator = ExportValidator()
        results = validator.validate_package(export_path)
        summary = next(r for r in results if r.check_name == "validation_summary")
        assert summary.status.value == "passed"
        assert (export_path / "manifest.json").exists()
        assert (export_path / "agent_config.json").exists()
        assert (export_path / "gnn_model.json").exists()
        assert (export_path / "knowledge_graph.json").exists()
        assert (export_path / "run.sh").exists()

    @pytest.mark.asyncio
    async def test_multi_agent_deployment_package(self):
        """Test creating deployment package for multiple agents."""
        engine = SimulationEngine(
            {
                "world": {"resolution": 5, "size": 100},
                "agents": {"count": 4},
                "simulation": {"max_cycles": 30},
            }
        )
        engine.initialize()
        await engine.start()
        for _ in range(30):
            await engine.step()
        deployment_path = Path("/tmp/test_deployment")
        success = await engine.export_deployment(
            deployment_path, include_world=True, include_history=True
        )
        assert success
        assert (deployment_path / "deployment.json").exists()
        assert (deployment_path / "agents").is_dir()
        assert (deployment_path / "world").is_dir()
        assert (deployment_path / "configs").is_dir()
        assert (deployment_path / "docker-compose.yml").exists()
        agent_exports = list((deployment_path / "agents").glob("*/"))
        assert len(agent_exports) == 4


def run_system_integration_tests():
    """Run all system integration tests."""
    pytest.main([__file__, "-v", "--asyncio-mode=auto", "-s"])
