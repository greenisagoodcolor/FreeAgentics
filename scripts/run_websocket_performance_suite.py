#!/usr/bin/env python3
"""
WebSocket Performance Testing Suite Runner
==========================================

Comprehensive script to run WebSocket performance tests with multiple modes:
- Mocked tests (fast, for CI/CD)
- Integration tests (with real server)
- Benchmark mode (detailed performance analysis)
- Regression mode (compare against baselines)

Usage:
    python scripts/run_websocket_performance_suite.py --mode=mocked
    python scripts/run_websocket_performance_suite.py --mode=integration
    python scripts/run_websocket_performance_suite.py --mode=benchmark --output=performance_report.json
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.performance.test_websocket_throughput_mocked import MockedWebSocketTester
from tests.performance.test_websocket_integrated_throughput import TestServerManager, IntegratedWebSocketTester

logger = logging.getLogger(__name__)


class WebSocketPerformanceSuite:
    """Comprehensive WebSocket performance testing suite."""
    
    def __init__(self, mode: str = "mocked", output_file: Optional[str] = None):
        self.mode = mode
        self.output_file = output_file
        self.results = {}
        self.start_time = datetime.now()
        
    async def run_mocked_performance_tests(self) -> Dict:
        """Run fast mocked performance tests."""
        logger.info("Running mocked WebSocket performance tests...")
        
        tester = MockedWebSocketTester()
        results = {}
        
        # Test 1: Multi-agent conversation throughput
        print("\n📊 Multi-Agent Conversation Throughput Test")
        print("-" * 50)
        
        conv_metrics = await tester.test_multi_agent_conversation_throughput(
            agent_count=5,
            conversation_turns=5,
            turn_duration_seconds=0.5,
            kg_updates_per_turn=2,
        )
        
        results["multi_agent_conversation"] = {
            "connections_established": conv_metrics.connections_established,
            "throughput_mps": conv_metrics.throughput_mps,
            "avg_latency_ms": conv_metrics.average_latency_ms,
            "p95_latency_ms": conv_metrics.p95_latency_ms,
            "conversations_completed": conv_metrics.agent_conversations_completed,
            "kg_updates_processed": conv_metrics.kg_updates_processed,
            "coordination_messages": conv_metrics.coordination_messages_sent,
            "memory_per_connection_mb": conv_metrics.memory_per_connection_mb,
            "business_impact_score": conv_metrics.business_impact_score,
            "sla_violations": conv_metrics.sla_violations,
        }
        
        print(f"✓ Agents Connected: {conv_metrics.connections_established}")
        print(f"✓ Throughput: {conv_metrics.throughput_mps:.1f} msg/s")
        print(f"✓ Avg Latency: {conv_metrics.average_latency_ms:.1f}ms")
        print(f"✓ P95 Latency: {conv_metrics.p95_latency_ms:.1f}ms")
        print(f"✓ Conversations: {conv_metrics.agent_conversations_completed}")
        print(f"✓ KG Updates: {conv_metrics.kg_updates_processed}")
        print(f"✓ Memory/Connection: {conv_metrics.memory_per_connection_mb:.1f}MB")
        print(f"✓ Business Impact: {conv_metrics.business_impact_score:.1f}/100")
        
        if conv_metrics.sla_violations:
            print(f"⚠️  SLA Violations: {len(conv_metrics.sla_violations)}")
            for violation in conv_metrics.sla_violations:
                print(f"   - {violation}")
        else:
            print("✅ All SLA requirements met")
            
        # Test 2: Connection stability
        print("\n🔗 Connection Stability Test")
        print("-" * 50)
        
        stab_metrics = await tester.test_connection_stability_patterns(
            connection_count=15,
            test_duration_seconds=10,
            dropout_probability=0.03,
        )
        
        results["connection_stability"] = {
            "connections_established": stab_metrics.connections_established,
            "connection_dropouts": stab_metrics.connection_dropouts,
            "reconnections_successful": stab_metrics.reconnections_successful,
            "messages_sent": stab_metrics.messages_sent,
            "throughput_mps": stab_metrics.throughput_mps,
            "business_impact_score": stab_metrics.business_impact_score,
            "sla_violations": stab_metrics.sla_violations,
        }
        
        reconnection_rate = (stab_metrics.reconnections_successful / max(stab_metrics.connection_dropouts, 1)) * 100
        
        print(f"✓ Connections: {stab_metrics.connections_established}")
        print(f"✓ Dropouts: {stab_metrics.connection_dropouts}")
        print(f"✓ Reconnections: {stab_metrics.reconnections_successful}")
        print(f"✓ Reconnection Rate: {reconnection_rate:.1f}%")
        print(f"✓ Messages: {stab_metrics.messages_sent}")
        print(f"✓ Business Impact: {stab_metrics.business_impact_score:.1f}/100")
        
        # Test 3: High throughput burst
        print("\n⚡ High Throughput Burst Test")
        print("-" * 50)
        
        burst_metrics = await tester.test_high_throughput_burst_patterns(
            connection_count=8,
            burst_duration_seconds=5,
            messages_per_second_per_connection=15,
        )
        
        results["high_throughput_burst"] = {
            "connections_established": burst_metrics.connections_established,
            "peak_throughput_mps": burst_metrics.throughput_mps,
            "avg_latency_ms": burst_metrics.average_latency_ms,
            "p95_latency_ms": burst_metrics.p95_latency_ms,
            "p99_latency_ms": burst_metrics.p99_latency_ms,
            "memory_per_connection_mb": burst_metrics.memory_per_connection_mb,
            "memory_budget_violations": burst_metrics.memory_budget_violations,
            "business_impact_score": burst_metrics.business_impact_score,
            "sla_violations": burst_metrics.sla_violations,
        }
        
        print(f"✓ Peak Throughput: {burst_metrics.throughput_mps:.1f} msg/s")
        print(f"✓ Avg Latency: {burst_metrics.average_latency_ms:.1f}ms")
        print(f"✓ P95 Latency: {burst_metrics.p95_latency_ms:.1f}ms")
        print(f"✓ P99 Latency: {burst_metrics.p99_latency_ms:.1f}ms")
        print(f"✓ Memory/Connection: {burst_metrics.memory_per_connection_mb:.1f}MB")
        print(f"✓ Memory Violations: {burst_metrics.memory_budget_violations}")
        print(f"✓ Business Impact: {burst_metrics.business_impact_score:.1f}/100")
        
        return results
        
    async def run_integration_performance_tests(self) -> Dict:
        """Run integration performance tests with real server."""
        logger.info("Running integration WebSocket performance tests...")
        
        try:
            # Import required for server management
            import aiohttp
        except ImportError:
            logger.error("aiohttp required for integration tests. Install with: pip install aiohttp")
            return {"error": "Missing aiohttp dependency"}
            
        results = {}
        
        # Start test server
        server_manager = TestServerManager()
        server_started = await server_manager.start_server()
        
        if not server_started:
            logger.error("Failed to start test server for integration tests")
            return {"error": "Failed to start test server"}
            
        try:
            tester = IntegratedWebSocketTester(server_manager)
            
            # Test 1: Realistic agent conversation
            print("\n🤖 Realistic Agent Conversation Test")
            print("-" * 50)
            
            conv_metrics = await tester.test_realistic_agent_conversation_throughput(
                agent_count=3,
                conversation_turns=5,
                turn_duration_ms=800,
            )
            
            results["realistic_conversation"] = conv_metrics
            
            throughput = conv_metrics["messages_sent"] / conv_metrics["test_duration_seconds"]
            
            print(f"✓ Agents: {conv_metrics['connections_established']}")
            print(f"✓ Messages: {conv_metrics['messages_sent']}")
            print(f"✓ Throughput: {throughput:.1f} msg/s")
            print(f"✓ Avg Latency: {conv_metrics['avg_latency_ms']:.1f}ms")
            print(f"✓ P95 Latency: {conv_metrics['p95_latency_ms']:.1f}ms")
            print(f"✓ Conversations: {conv_metrics['conversations_completed']}")
            print(f"✓ KG Updates: {conv_metrics['kg_updates_sent']}")
            print(f"✓ Coordination Messages: {conv_metrics['coordination_messages']}")
            
            # Test 2: Connection stability with server
            print("\n🔄 Connection Stability with Server")
            print("-" * 50)
            
            stab_metrics = await tester.test_connection_stability_with_server(
                connection_count=5,
                test_duration_seconds=20,
                message_interval_seconds=1.0,
            )
            
            results["server_stability"] = stab_metrics
            
            print(f"✓ Connections: {stab_metrics['connections_established']}")
            print(f"✓ Messages: {stab_metrics['messages_sent']}")
            print(f"✓ Dropouts: {stab_metrics['connection_dropouts']}")
            print(f"✓ Reconnections: {stab_metrics['reconnections_successful']}")
            
        finally:
            await server_manager.stop_server()
            
        return results
        
    async def run_benchmark_mode(self) -> Dict:
        """Run comprehensive benchmark analysis."""
        logger.info("Running comprehensive WebSocket benchmarks...")
        
        results = {
            "mocked_tests": await self.run_mocked_performance_tests(),
        }
        
        # Add integration tests if possible
        try:
            integration_results = await self.run_integration_performance_tests()
            if "error" not in integration_results:
                results["integration_tests"] = integration_results
        except Exception as e:
            logger.warning(f"Integration tests skipped: {e}")
            results["integration_tests"] = {"error": str(e)}
            
        # Generate comprehensive analysis
        results["benchmark_analysis"] = self._generate_benchmark_analysis(results)
        
        return results
        
    def _generate_benchmark_analysis(self, results: Dict) -> Dict:
        """Generate comprehensive benchmark analysis."""
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "test_environment": {
                "mode": self.mode,
                "python_version": sys.version,
                "platform": sys.platform,
            },
            "performance_summary": {},
            "sla_compliance": {},
            "recommendations": [],
        }
        
        # Analyze mocked test results
        if "mocked_tests" in results:
            mocked = results["mocked_tests"]
            
            # Performance summary
            if "multi_agent_conversation" in mocked:
                conv = mocked["multi_agent_conversation"]
                analysis["performance_summary"]["conversation_throughput_mps"] = conv.get("throughput_mps", 0)
                analysis["performance_summary"]["conversation_latency_p95_ms"] = conv.get("p95_latency_ms", 0)
                analysis["performance_summary"]["memory_per_connection_mb"] = conv.get("memory_per_connection_mb", 0)
                
            if "high_throughput_burst" in mocked:
                burst = mocked["high_throughput_burst"]
                analysis["performance_summary"]["peak_throughput_mps"] = burst.get("peak_throughput_mps", 0)
                analysis["performance_summary"]["burst_latency_p99_ms"] = burst.get("p99_latency_ms", 0)
                
            # SLA compliance analysis
            all_violations = []
            total_business_impact = 0
            test_count = 0
            
            for test_name, test_results in mocked.items():
                if isinstance(test_results, dict):
                    violations = test_results.get("sla_violations", [])
                    if violations:
                        all_violations.extend([f"{test_name}: {v}" for v in violations])
                        
                    impact = test_results.get("business_impact_score", 0)
                    if impact > 0:
                        total_business_impact += impact
                        test_count += 1
                        
            analysis["sla_compliance"]["violations"] = all_violations
            analysis["sla_compliance"]["requirements_met"] = len(all_violations) == 0
            analysis["sla_compliance"]["avg_business_impact"] = total_business_impact / max(test_count, 1)
            
            # Generate recommendations
            if analysis["sla_compliance"]["avg_business_impact"] > 50:
                analysis["recommendations"].append("High business impact detected. Consider optimizing WebSocket performance.")
                
            if analysis["performance_summary"].get("conversation_latency_p95_ms", 0) > 200:
                analysis["recommendations"].append("P95 conversation latency exceeds 200ms. Optimize message processing.")
                
            if analysis["performance_summary"].get("memory_per_connection_mb", 0) > 34.5:
                analysis["recommendations"].append("Memory usage per connection exceeds 34.5MB budget. Investigate memory leaks.")
                
            if not analysis["recommendations"]:
                analysis["recommendations"].append("All performance metrics within acceptable ranges. System performing well.")
                
        return analysis
        
    def save_results(self, results: Dict):
        """Save results to file."""
        if not self.output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_file = f"websocket_performance_{self.mode}_{timestamp}.json"
            
        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        print(f"\n📄 Results saved to: {output_path.absolute()}")
        
    def print_summary(self, results: Dict):
        """Print comprehensive test summary."""
        print("\n" + "=" * 80)
        print("WEBSOCKET PERFORMANCE TESTING SUMMARY")
        print("=" * 80)
        
        if "benchmark_analysis" in results:
            analysis = results["benchmark_analysis"]
            
            print("\n📊 Performance Metrics:")
            summary = analysis.get("performance_summary", {})
            for metric, value in summary.items():
                if isinstance(value, float):
                    print(f"   {metric}: {value:.2f}")
                else:
                    print(f"   {metric}: {value}")
                    
            print("\n✅ SLA Compliance:")
            sla = analysis.get("sla_compliance", {})
            print(f"   Requirements Met: {'✓' if sla.get('requirements_met', False) else '✗'}")
            print(f"   Business Impact: {sla.get('avg_business_impact', 0):.1f}/100")
            
            if sla.get("violations"):
                print(f"   Violations ({len(sla['violations'])}):")
                for violation in sla["violations"]:
                    print(f"     - {violation}")
                    
            print("\n💡 Recommendations:")
            for rec in analysis.get("recommendations", []):
                print(f"   - {rec}")
                
        duration = (datetime.now() - self.start_time).total_seconds()
        print(f"\n⏱️  Total execution time: {duration:.1f} seconds")
        print("=" * 80)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="WebSocket Performance Testing Suite")
    parser.add_argument("--mode", choices=["mocked", "integration", "benchmark"], 
                       default="mocked", help="Test mode to run")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif args.quiet:
        logging.basicConfig(level=logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO)
        
    suite = WebSocketPerformanceSuite(mode=args.mode, output_file=args.output)
    
    print("🚀 Starting WebSocket Performance Testing Suite")
    print(f"Mode: {args.mode}")
    print(f"Time: {suite.start_time.isoformat()}")
    
    try:
        if args.mode == "mocked":
            results = await suite.run_mocked_performance_tests()
        elif args.mode == "integration":
            results = await suite.run_integration_performance_tests()
        elif args.mode == "benchmark":
            results = await suite.run_benchmark_mode()
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
            
        # Save and display results
        suite.save_results(results)
        suite.print_summary(results)
        
        # Exit with error code if there are SLA violations
        if "benchmark_analysis" in results:
            sla = results["benchmark_analysis"].get("sla_compliance", {})
            if not sla.get("requirements_met", True):
                print("\n❌ SLA violations detected. Exiting with error code 1.")
                sys.exit(1)
                
        print("\n✅ WebSocket performance testing completed successfully!")
        
    except Exception as e:
        logger.error(f"Performance testing failed: {e}")
        print(f"\n❌ Performance testing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())