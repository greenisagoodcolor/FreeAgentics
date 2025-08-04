"""Database Chaos Engineering Tests for Production Resilience.

Implements chaos engineering principles for database performance testing:
- Network latency simulation
- Connection failure scenarios
- Resource constraint testing
- Database failover simulation
- Degraded performance conditions
"""

import asyncio
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DisconnectionError, OperationalError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import Pool

from database.models import Agent, AgentStatus
from tests.db_infrastructure.factories import AgentFactory
from tests.db_infrastructure.fixtures import isolated_db_test
from tests.db_infrastructure.test_config import (
    create_test_engine,
    setup_test_database,
    teardown_test_database,
)

logger = logging.getLogger(__name__)


@dataclass
class ChaosScenario:
    """Configuration for a chaos engineering scenario."""
    
    name: str
    description: str
    failure_probability: float = 0.1  # 10% default failure rate
    latency_ms: int = 0  # Additional latency in milliseconds
    duration_seconds: int = 30  # How long to run the scenario
    recovery_time_seconds: int = 5  # Time to recover from failures


class NetworkLatencySimulator:
    """Simulate network latency in database connections."""
    
    def __init__(self, engine: Engine, base_latency_ms: int = 50):
        self.engine = engine
        self.base_latency_ms = base_latency_ms
        self.enabled = False
    
    def enable_latency_simulation(self):
        """Enable latency simulation on database connections."""
        self.enabled = True
        
        @event.listens_for(self.engine, "before_cursor_execute")
        def add_latency(conn, cursor, statement, parameters, context, executemany):
            if self.enabled:
                # Simulate network latency
                latency = self.base_latency_ms + random.randint(0, self.base_latency_ms // 2)
                time.sleep(latency / 1000.0)  # Convert to seconds
    
    def disable_latency_simulation(self):
        """Disable latency simulation."""
        self.enabled = False


class ConnectionFailureSimulator:
    """Simulate connection failures and recovery."""
    
    def __init__(self, engine: Engine, failure_probability: float = 0.1):
        self.engine = engine
        self.failure_probability = failure_probability
        self.enabled = False
        self.connection_attempts = 0
        self.failed_connections = 0
    
    def enable_connection_failures(self):
        """Enable connection failure simulation."""
        self.enabled = True
        
        @event.listens_for(Pool, "connect")
        def simulate_connection_failure(dbapi_connection, connection_record):
            if self.enabled:
                self.connection_attempts += 1
                
                if random.random() < self.failure_probability:
                    self.failed_connections += 1
                    logger.warning(f"Simulating connection failure (attempt {self.connection_attempts})")
                    raise OperationalError("Simulated connection failure", None, None)
    
    def disable_connection_failures(self):
        """Disable connection failure simulation."""
        self.enabled = False
    
    def get_failure_stats(self) -> Dict[str, int]:
        """Get connection failure statistics."""
        return {
            "total_attempts": self.connection_attempts,
            "failed_connections": self.failed_connections,
            "success_rate": (
                (self.connection_attempts - self.failed_connections) / self.connection_attempts
                if self.connection_attempts > 0
                else 0
            ),
        }


class DatabaseChaosEngine:
    """Orchestrates chaos engineering scenarios for database testing."""
    
    def __init__(self, engine: Engine):
        self.engine = engine
        self.latency_simulator = NetworkLatencySimulator(engine)
        self.connection_failure_simulator = ConnectionFailureSimulator(engine)
        self.chaos_results = {}
    
    @contextmanager
    def chaos_scenario(self, scenario: ChaosScenario):
        """Context manager for running chaos scenarios."""
        logger.info(f"🌀 Starting chaos scenario: {scenario.name}")
        logger.info(f"Description: {scenario.description}")
        logger.info(f"Duration: {scenario.duration_seconds}s, Failure rate: {scenario.failure_probability*100}%")
        
        start_time = time.time()
        
        try:
            # Enable chaos conditions
            if scenario.latency_ms > 0:
                self.latency_simulator.base_latency_ms = scenario.latency_ms
                self.latency_simulator.enable_latency_simulation()
            
            if scenario.failure_probability > 0:
                self.connection_failure_simulator.failure_probability = scenario.failure_probability
                self.connection_failure_simulator.enable_connection_failures()
            
            yield self
            
        finally:
            # Disable chaos conditions
            self.latency_simulator.disable_latency_simulation()
            self.connection_failure_simulator.disable_connection_failures()
            
            duration = time.time() - start_time
            
            # Record results
            self.chaos_results[scenario.name] = {
                "duration": duration,
                "latency_ms": scenario.latency_ms,
                "failure_probability": scenario.failure_probability,
                "connection_stats": self.connection_failure_simulator.get_failure_stats(),
            }
            
            logger.info(f"✅ Chaos scenario completed: {scenario.name} ({duration:.2f}s)")
    
    async def test_resilient_agent_operations(self, scenario: ChaosScenario, num_agents: int = 100) -> Dict[str, Any]:
        """Test agent operations under chaos conditions."""
        Session = sessionmaker(bind=self.engine)
        
        # Create initial agents
        with isolated_db_test(self.engine) as session:
            agents = AgentFactory.create_batch(
                session,
                count=num_agents,
                agent_type="chaos_test_agent",
                status=AgentStatus.ACTIVE,
            )
            agent_ids = [agent.agent_id for agent in agents]
            session.commit()
        
        results = {
            "successful_operations": 0,
            "failed_operations": 0,
            "connection_errors": 0,
            "timeout_errors": 0,
            "recovery_attempts": 0,
        }
        
        def resilient_operation(agent_id):
            """Perform database operation with resilience patterns."""
            max_retries = 3
            backoff_base = 0.1  # 100ms base backoff
            
            for attempt in range(max_retries):
                session = Session()
                
                try:
                    # Attempt agent update with exponential backoff
                    agent = session.query(Agent).filter(Agent.agent_id == agent_id).first()
                    
                    if agent:
                        # Update with chaos-resilient logic
                        agent.beliefs = {
                            "chaos_test": True,
                            "timestamp": datetime.utcnow().isoformat(),
                            "attempt": attempt + 1,
                            "resilience_score": random.uniform(0.5, 1.0),
                        }
                        agent.updated_at = datetime.utcnow()
                        
                        session.commit()
                        results["successful_operations"] += 1
                        return True
                
                except (OperationalError, DisconnectionError) as e:
                    session.rollback()
                    
                    if "connection" in str(e).lower():
                        results["connection_errors"] += 1
                    elif "timeout" in str(e).lower():
                        results["timeout_errors"] += 1
                    
                    if attempt < max_retries - 1:
                        results["recovery_attempts"] += 1
                        # Exponential backoff with jitter
                        backoff = backoff_base * (2 ** attempt) + random.uniform(0, 0.1)
                        time.sleep(backoff)
                    
                    logger.debug(f"Database operation failed (attempt {attempt + 1}): {e}")
                
                except Exception as e:
                    session.rollback()
                    logger.error(f"Unexpected error in chaos test: {e}")
                    break
                
                finally:
                    session.close()
            
            results["failed_operations"] += 1
            return False
        
        # Run operations concurrently under chaos conditions
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(resilient_operation, agent_id) for agent_id in agent_ids]
            
            # Process results as they complete
            for future in as_completed(futures):
                try:
                    future.result(timeout=30)  # 30 second timeout per operation
                except Exception as e:
                    results["failed_operations"] += 1
                    logger.error(f"Operation timeout or error: {e}")
        
        # Calculate resilience metrics
        total_ops = results["successful_operations"] + results["failed_operations"]
        results["success_rate"] = results["successful_operations"] / total_ops if total_ops > 0 else 0
        results["recovery_rate"] = (
            results["recovery_attempts"] / (results["connection_errors"] + results["timeout_errors"])
            if (results["connection_errors"] + results["timeout_errors"]) > 0
            else 0
        )
        
        return results
    
    async def test_database_failover_simulation(self) -> Dict[str, Any]:
        """Simulate database failover scenarios."""
        results = {"failover_scenarios": []}
        
        # Scenario 1: Gradual performance degradation
        degradation_scenario = ChaosScenario(
            name="gradual_degradation",
            description="Gradually increasing latency to simulate performance degradation",
            latency_ms=100,
            duration_seconds=20,
        )
        
        async with self.chaos_scenario(degradation_scenario):
            # Simulate increasing load
            for latency in [50, 100, 200, 500]:
                self.latency_simulator.base_latency_ms = latency
                
                # Test operations under increasing latency
                operation_results = await self.test_resilient_agent_operations(
                    degradation_scenario, num_agents=20
                )
                
                results["failover_scenarios"].append({
                    "latency_ms": latency,
                    "results": operation_results,
                })
                
                logger.info(f"Tested at {latency}ms latency: {operation_results['success_rate']:.2%} success rate")
        
        # Scenario 2: Connection instability
        instability_scenario = ChaosScenario(
            name="connection_instability",
            description="Intermittent connection failures with recovery",
            failure_probability=0.3,  # 30% failure rate
            duration_seconds=15,
        )
        
        async with self.chaos_scenario(instability_scenario):
            instability_results = await self.test_resilient_agent_operations(
                instability_scenario, num_agents=50
            )
            
            results["connection_instability"] = instability_results
        
        return results


@pytest.mark.performance
@pytest.mark.chaos
class TestDatabaseChaosEngineering:
    """Test suite for database chaos engineering scenarios."""
    
    @pytest.fixture
    def chaos_engine(self):
        """Create chaos engine for testing."""
        engine = create_test_engine(use_sqlite=False)  # PostgreSQL for realistic testing
        setup_test_database(engine)
        
        chaos_engine = DatabaseChaosEngine(engine)
        
        yield chaos_engine
        
        teardown_test_database(engine)
    
    @pytest.mark.asyncio
    async def test_network_latency_resilience(self, chaos_engine):
        """Test database operations under network latency."""
        scenario = ChaosScenario(
            name="network_latency_test",
            description="Test operations under 200ms network latency",
            latency_ms=200,
            duration_seconds=10,
        )
        
        with chaos_engine.chaos_scenario(scenario):
            results = await chaos_engine.test_resilient_agent_operations(scenario, num_agents=30)
        
        # Assert that operations still succeed despite latency
        assert results["success_rate"] > 0.8, f"Success rate too low under latency: {results['success_rate']}"
        assert results["successful_operations"] > 20, "Too few successful operations"
        
        logger.info(f"Latency resilience test: {results['success_rate']:.2%} success rate")
    
    @pytest.mark.asyncio
    async def test_connection_failure_recovery(self, chaos_engine):
        """Test recovery from connection failures."""
        scenario = ChaosScenario(
            name="connection_failure_test",
            description="Test recovery from 20% connection failures",
            failure_probability=0.2,
            duration_seconds=15,
        )
        
        with chaos_engine.chaos_scenario(scenario):
            results = await chaos_engine.test_resilient_agent_operations(scenario, num_agents=50)
        
        # Assert that recovery mechanisms work
        assert results["success_rate"] > 0.7, f"Success rate too low with failures: {results['success_rate']}"
        assert results["recovery_attempts"] > 0, "No recovery attempts detected"
        assert results["recovery_rate"] > 0.5, "Recovery rate too low"
        
        logger.info(f"Connection failure recovery: {results['recovery_rate']:.2%} recovery rate")
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_complete_failover_simulation(self, chaos_engine):
        """Test complete database failover simulation."""
        results = await chaos_engine.test_database_failover_simulation()
        
        # Validate failover scenario results
        assert "failover_scenarios" in results
        assert len(results["failover_scenarios"]) > 0
        
        # Check that system handles gradual degradation
        for scenario in results["failover_scenarios"]:
            latency = scenario["latency_ms"]
            success_rate = scenario["results"]["success_rate"]
            
            logger.info(f"Latency {latency}ms: {success_rate:.2%} success rate")
            
            # Success rate should degrade gracefully, not cliff-drop
            if latency <= 200:
                assert success_rate > 0.6, f"Success rate dropped too quickly at {latency}ms latency"
        
        # Check connection instability handling
        assert "connection_instability" in results
        instability_result = results["connection_instability"]
        assert instability_result["success_rate"] > 0.6, "System not resilient to connection instability"
    
    @pytest.mark.asyncio
    async def test_chaos_metrics_collection(self, chaos_engine):
        """Test that chaos engineering metrics are properly collected."""
        scenario = ChaosScenario(
            name="metrics_test",
            description="Test metrics collection during chaos",
            latency_ms=100,
            failure_probability=0.1,
            duration_seconds=5,
        )
        
        with chaos_engine.chaos_scenario(scenario):
            await chaos_engine.test_resilient_agent_operations(scenario, num_agents=20)
        
        # Verify chaos results are recorded
        assert scenario.name in chaos_engine.chaos_results
        result = chaos_engine.chaos_results[scenario.name]
        
        assert "duration" in result
        assert "latency_ms" in result
        assert "failure_probability" in result
        assert "connection_stats" in result
        
        # Validate connection stats
        conn_stats = result["connection_stats"]
        assert "total_attempts" in conn_stats
        assert "failed_connections" in conn_stats
        assert "success_rate" in conn_stats
        
        logger.info(f"Chaos metrics collected: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    import asyncio
    import json
    
    async def run_chaos_engineering_demo():
        """Run chaos engineering demonstration."""
        print("Running Database Chaos Engineering Demo...\n")
        
        # Create chaos engine with PostgreSQL
        engine = create_test_engine(use_sqlite=False)
        setup_test_database(engine)
        
        chaos_engine = DatabaseChaosEngine(engine)
        
        try:
            # Demo 1: Network latency impact
            print("🌀 Demo 1: Network Latency Impact")
            latency_scenario = ChaosScenario(
                name="demo_latency",
                description="Demonstrate impact of network latency on database operations",
                latency_ms=300,
                duration_seconds=10,
            )
            
            with chaos_engine.chaos_scenario(latency_scenario):
                latency_results = await chaos_engine.test_resilient_agent_operations(
                    latency_scenario, num_agents=25
                )
            
            print(f"Latency Results: {latency_results['success_rate']:.2%} success rate")
            print(f"Recovery attempts: {latency_results['recovery_attempts']}")
            
            # Demo 2: Connection failures
            print("\n🌀 Demo 2: Connection Failure Resilience")
            failure_scenario = ChaosScenario(
                name="demo_failures",
                description="Test resilience to connection failures",
                failure_probability=0.25,  # 25% failure rate
                duration_seconds=12,
            )
            
            with chaos_engine.chaos_scenario(failure_scenario):
                failure_results = await chaos_engine.test_resilient_agent_operations(
                    failure_scenario, num_agents=40
                )
            
            print(f"Failure Results: {failure_results['success_rate']:.2%} success rate")
            print(f"Connection errors: {failure_results['connection_errors']}")
            print(f"Recovery rate: {failure_results['recovery_rate']:.2%}")
            
            # Demo 3: Complete failover simulation
            print("\n🌀 Demo 3: Complete Failover Simulation")
            failover_results = await chaos_engine.test_database_failover_simulation()
            
            print("Gradual degradation results:")
            for scenario in failover_results["failover_scenarios"]:
                latency = scenario["latency_ms"]
                success_rate = scenario["results"]["success_rate"]
                print(f"  {latency}ms latency: {success_rate:.2%} success")
            
            print(f"\nConnection instability: {failover_results['connection_instability']['success_rate']:.2%} success")
            
            # Summary
            print("\n" + "=" * 60)
            print("CHAOS ENGINEERING SUMMARY")
            print("=" * 60)
            print(f"Total scenarios executed: {len(chaos_engine.chaos_results)}")
            
            for name, result in chaos_engine.chaos_results.items():
                print(f"\n{name}:")
                print(f"  Duration: {result['duration']:.2f}s")
                print(f"  Latency: {result['latency_ms']}ms")
                print(f"  Failure rate: {result['failure_probability']*100:.1f}%")
                
                conn_stats = result['connection_stats']
                if conn_stats['total_attempts'] > 0:
                    print(f"  Connection success: {conn_stats['success_rate']:.2%}")
            
            print("\n✅ Chaos engineering demonstration completed successfully!")
            
        except Exception as e:
            print(f"❌ Chaos engineering demo failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            teardown_test_database(engine)
    
    asyncio.run(run_chaos_engineering_demo())
