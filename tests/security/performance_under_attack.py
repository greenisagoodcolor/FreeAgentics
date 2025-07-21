"""
Performance Under Attack Testing

This module tests system performance and resilience under various attack conditions
to ensure the platform maintains functionality even during security incidents.
"""

import asyncio
import json
import statistics
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import httpx
import psutil


@dataclass
class AttackMetrics:
    """Metrics collected during attack simulation"""

    response_times: List[float]
    success_rate: float
    error_rate: float
    throughput: float
    cpu_usage: List[float]
    memory_usage: List[float]
    network_usage: List[float]
    timestamp: datetime


class PerformanceUnderAttackTester:
    """Test system performance under various attack scenarios"""

    def __init__(self, target_url: str = "http://localhost:8000"):
        self.target_url = target_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.metrics = {}
        self.baseline_metrics = None

        # Performance thresholds
        self.thresholds = {
            "max_response_time": 5.0,  # 5 seconds max
            "min_success_rate": 0.95,  # 95% success rate
            "max_cpu_usage": 80.0,  # 80% CPU max
            "max_memory_usage": 85.0,  # 85% memory max
            "min_throughput_ratio": 0.7,  # 70% of baseline throughput
        }

    async def run_all_attack_scenarios(self) -> Dict:
        """Run all attack scenarios and measure performance"""

        print("Starting Performance Under Attack Testing...")

        # Get baseline performance
        await self._measure_baseline_performance()

        # Test scenarios
        scenarios = [
            ("DDoS Attack", self._simulate_ddos_attack),
            ("Brute Force Attack", self._simulate_brute_force_attack),
            ("SQL Injection Flood", self._simulate_sql_injection_flood),
            ("XSS Attack Wave", self._simulate_xss_attack_wave),
            ("Resource Exhaustion", self._simulate_resource_exhaustion),
            (
                "Authentication Bypass Attempts",
                self._simulate_auth_bypass_flood,
            ),
            ("Rate Limit Testing", self._simulate_rate_limit_stress),
            ("Large Payload Attack", self._simulate_large_payload_attack),
        ]

        results = {}

        for scenario_name, scenario_func in scenarios:
            print(f"\n[*] Running {scenario_name}...")

            try:
                metrics = await scenario_func()
                results[scenario_name] = {
                    "metrics": metrics,
                    "performance_impact": self._calculate_performance_impact(metrics),
                    "passed": self._evaluate_performance_threshold(metrics),
                    "recommendations": self._generate_recommendations(scenario_name, metrics),
                }

                # Brief recovery time between scenarios
                await asyncio.sleep(5)

            except Exception as e:
                results[scenario_name] = {
                    "error": str(e),
                    "passed": False,
                    "recommendations": ["Fix test execution error"],
                }

        # Generate comprehensive report
        report = self._generate_attack_performance_report(results)

        await self.client.aclose()
        return report

    async def _measure_baseline_performance(self):
        """Measure baseline system performance"""

        print("Measuring baseline performance...")

        # Monitor system metrics
        system_monitor = SystemMonitor()
        system_monitor.start()

        # Run normal load test
        response_times = []
        successful_requests = 0
        total_requests = 100

        start_time = time.time()

        for i in range(total_requests):
            try:
                response = await self.client.get(f"{self.target_url}/api/v1/health")
                response_times.append(response.elapsed.total_seconds())

                if response.status_code == 200:
                    successful_requests += 1

            except Exception:
                pass

        end_time = time.time()
        duration = end_time - start_time

        # Stop monitoring
        system_metrics = system_monitor.stop()

        self.baseline_metrics = AttackMetrics(
            response_times=response_times,
            success_rate=successful_requests / total_requests,
            error_rate=1 - (successful_requests / total_requests),
            throughput=successful_requests / duration,
            cpu_usage=system_metrics["cpu"],
            memory_usage=system_metrics["memory"],
            network_usage=system_metrics["network"],
            timestamp=datetime.now(),
        )

        print("Baseline established:")
        print(f"  - Average response time: {statistics.mean(response_times):.2f}s")
        print(f"  - Success rate: {successful_requests/total_requests:.2%}")
        print(f"  - Throughput: {successful_requests/duration:.2f} req/s")

    async def _simulate_ddos_attack(self) -> AttackMetrics:
        """Simulate DDoS attack with high concurrent requests"""

        # System monitoring
        system_monitor = SystemMonitor()
        system_monitor.start()

        # Attack parameters
        concurrent_requests = 1000
        attack_duration = 60  # seconds

        response_times = []
        successful_requests = 0
        total_requests = 0

        start_time = time.time()

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(concurrent_requests)

        async def make_request():
            nonlocal successful_requests, total_requests

            async with semaphore:
                try:
                    response = await self.client.get(f"{self.target_url}/api/v1/health")
                    response_times.append(response.elapsed.total_seconds())
                    total_requests += 1

                    if response.status_code == 200:
                        successful_requests += 1

                except Exception:
                    total_requests += 1

        # Launch attack
        tasks = []
        end_time = start_time + attack_duration

        while time.time() < end_time:
            task = asyncio.create_task(make_request())
            tasks.append(task)

            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.01)

        # Wait for all requests to complete
        await asyncio.gather(*tasks, return_exceptions=True)

        # Stop monitoring
        system_metrics = system_monitor.stop()

        return AttackMetrics(
            response_times=response_times,
            success_rate=successful_requests / max(total_requests, 1),
            error_rate=1 - (successful_requests / max(total_requests, 1)),
            throughput=successful_requests / attack_duration,
            cpu_usage=system_metrics["cpu"],
            memory_usage=system_metrics["memory"],
            network_usage=system_metrics["network"],
            timestamp=datetime.now(),
        )

    async def _simulate_brute_force_attack(self) -> AttackMetrics:
        """Simulate brute force login attack"""

        system_monitor = SystemMonitor()
        system_monitor.start()

        # Attack parameters
        attack_duration = 30
        credentials_list = [
            ("admin", "password"),
            ("admin", "123456"),
            ("admin", "admin"),
            ("user", "password"),
            ("test", "test"),
            ("root", "root"),
        ]

        response_times = []
        successful_requests = 0
        total_requests = 0

        start_time = time.time()
        end_time = start_time + attack_duration

        while time.time() < end_time:
            for username, password in credentials_list:
                try:
                    response = await self.client.post(
                        f"{self.target_url}/api/v1/auth/login",
                        json={"username": username, "password": password},
                    )

                    response_times.append(response.elapsed.total_seconds())
                    total_requests += 1

                    if response.status_code in [
                        200,
                        401,
                    ]:  # Both are valid responses
                        successful_requests += 1

                except Exception:
                    total_requests += 1

        system_metrics = system_monitor.stop()

        return AttackMetrics(
            response_times=response_times,
            success_rate=successful_requests / max(total_requests, 1),
            error_rate=1 - (successful_requests / max(total_requests, 1)),
            throughput=successful_requests / attack_duration,
            cpu_usage=system_metrics["cpu"],
            memory_usage=system_metrics["memory"],
            network_usage=system_metrics["network"],
            timestamp=datetime.now(),
        )

    async def _simulate_sql_injection_flood(self) -> AttackMetrics:
        """Simulate flood of SQL injection attempts"""

        system_monitor = SystemMonitor()
        system_monitor.start()

        # SQL injection payloads
        payloads = [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "' UNION SELECT * FROM users --",
            "admin'--",
            "' OR 1=1--",
            "' OR SLEEP(5)--",
        ]

        response_times = []
        successful_requests = 0
        total_requests = 0

        start_time = time.time()
        attack_duration = 30

        for _ in range(1000):  # 1000 injection attempts
            for payload in payloads:
                try:
                    response = await self.client.get(
                        f"{self.target_url}/api/v1/users/search",
                        params={"q": payload},
                    )

                    response_times.append(response.elapsed.total_seconds())
                    total_requests += 1

                    if response.status_code in [
                        200,
                        400,
                        422,
                    ]:  # Valid responses
                        successful_requests += 1

                except Exception:
                    total_requests += 1

                # Check if duration exceeded
                if time.time() - start_time > attack_duration:
                    break

        system_metrics = system_monitor.stop()

        return AttackMetrics(
            response_times=response_times,
            success_rate=successful_requests / max(total_requests, 1),
            error_rate=1 - (successful_requests / max(total_requests, 1)),
            throughput=successful_requests / (time.time() - start_time),
            cpu_usage=system_metrics["cpu"],
            memory_usage=system_metrics["memory"],
            network_usage=system_metrics["network"],
            timestamp=datetime.now(),
        )

    async def _simulate_xss_attack_wave(self) -> AttackMetrics:
        """Simulate wave of XSS attacks"""

        system_monitor = SystemMonitor()
        system_monitor.start()

        # XSS payloads
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "javascript:alert('XSS')",
            "<iframe src='javascript:alert(\"XSS\")'></iframe>",
        ]

        response_times = []
        successful_requests = 0
        total_requests = 0

        start_time = time.time()
        attack_duration = 30

        for _ in range(500):  # 500 XSS attempts
            for payload in xss_payloads:
                try:
                    # Test in search parameter
                    response = await self.client.get(
                        f"{self.target_url}/api/v1/search",
                        params={"q": payload},
                    )

                    response_times.append(response.elapsed.total_seconds())
                    total_requests += 1

                    if response.status_code in [200, 400, 422]:
                        successful_requests += 1

                except Exception:
                    total_requests += 1

                if time.time() - start_time > attack_duration:
                    break

        system_metrics = system_monitor.stop()

        return AttackMetrics(
            response_times=response_times,
            success_rate=successful_requests / max(total_requests, 1),
            error_rate=1 - (successful_requests / max(total_requests, 1)),
            throughput=successful_requests / (time.time() - start_time),
            cpu_usage=system_metrics["cpu"],
            memory_usage=system_metrics["memory"],
            network_usage=system_metrics["network"],
            timestamp=datetime.now(),
        )

    async def _simulate_resource_exhaustion(self) -> AttackMetrics:
        """Simulate resource exhaustion attacks"""

        system_monitor = SystemMonitor()
        system_monitor.start()

        response_times = []
        successful_requests = 0
        total_requests = 0

        start_time = time.time()
        attack_duration = 30

        # Create many large requests
        large_data = "A" * 10000  # 10KB payload

        for _ in range(100):
            try:
                response = await self.client.post(
                    f"{self.target_url}/api/v1/resources",
                    json={
                        "name": f"resource_{total_requests}",
                        "data": large_data,
                        "metadata": {"size": len(large_data)},
                    },
                )

                response_times.append(response.elapsed.total_seconds())
                total_requests += 1

                if response.status_code in [200, 201, 400, 422]:
                    successful_requests += 1

            except Exception:
                total_requests += 1

            if time.time() - start_time > attack_duration:
                break

        system_metrics = system_monitor.stop()

        return AttackMetrics(
            response_times=response_times,
            success_rate=successful_requests / max(total_requests, 1),
            error_rate=1 - (successful_requests / max(total_requests, 1)),
            throughput=successful_requests / (time.time() - start_time),
            cpu_usage=system_metrics["cpu"],
            memory_usage=system_metrics["memory"],
            network_usage=system_metrics["network"],
            timestamp=datetime.now(),
        )

    async def _simulate_auth_bypass_flood(self) -> AttackMetrics:
        """Simulate flood of authentication bypass attempts"""

        system_monitor = SystemMonitor()
        system_monitor.start()

        # Various bypass techniques
        bypass_headers = [
            {"X-Forwarded-For": "127.0.0.1"},
            {"X-Real-IP": "127.0.0.1"},
            {"X-Originating-IP": "127.0.0.1"},
            {"X-Original-URL": "/api/v1/public"},
            {"Authorization": "Bearer null"},
            {"Authorization": "Bearer undefined"},
            {"Authorization": "Basic YWRtaW46YWRtaW4="},
        ]

        response_times = []
        successful_requests = 0
        total_requests = 0

        start_time = time.time()
        attack_duration = 30

        protected_endpoints = [
            "/api/v1/admin/users",
            "/api/v1/resources/protected",
            "/api/v1/users/profile",
            "/api/v1/agents/create",
        ]

        for _ in range(200):  # 200 bypass attempts
            for endpoint in protected_endpoints:
                for headers in bypass_headers:
                    try:
                        response = await self.client.get(
                            f"{self.target_url}{endpoint}", headers=headers
                        )

                        response_times.append(response.elapsed.total_seconds())
                        total_requests += 1

                        if response.status_code in [200, 401, 403]:
                            successful_requests += 1

                    except Exception:
                        total_requests += 1

                    if time.time() - start_time > attack_duration:
                        break

        system_metrics = system_monitor.stop()

        return AttackMetrics(
            response_times=response_times,
            success_rate=successful_requests / max(total_requests, 1),
            error_rate=1 - (successful_requests / max(total_requests, 1)),
            throughput=successful_requests / (time.time() - start_time),
            cpu_usage=system_metrics["cpu"],
            memory_usage=system_metrics["memory"],
            network_usage=system_metrics["network"],
            timestamp=datetime.now(),
        )

    async def _simulate_rate_limit_stress(self) -> AttackMetrics:
        """Simulate rate limit stress testing"""

        system_monitor = SystemMonitor()
        system_monitor.start()

        response_times = []
        successful_requests = 0
        total_requests = 0

        start_time = time.time()
        attack_duration = 30

        # Rapid requests to trigger rate limiting
        for _ in range(1000):
            try:
                response = await self.client.post(
                    f"{self.target_url}/api/v1/auth/login",
                    json={"username": "test", "password": "test"},
                )

                response_times.append(response.elapsed.total_seconds())
                total_requests += 1

                if response.status_code in [
                    200,
                    401,
                    429,
                ]:  # Including rate limit response
                    successful_requests += 1

            except Exception:
                total_requests += 1

            if time.time() - start_time > attack_duration:
                break

        system_metrics = system_monitor.stop()

        return AttackMetrics(
            response_times=response_times,
            success_rate=successful_requests / max(total_requests, 1),
            error_rate=1 - (successful_requests / max(total_requests, 1)),
            throughput=successful_requests / (time.time() - start_time),
            cpu_usage=system_metrics["cpu"],
            memory_usage=system_metrics["memory"],
            network_usage=system_metrics["network"],
            timestamp=datetime.now(),
        )

    async def _simulate_large_payload_attack(self) -> AttackMetrics:
        """Simulate large payload attacks"""

        system_monitor = SystemMonitor()
        system_monitor.start()

        response_times = []
        successful_requests = 0
        total_requests = 0

        start_time = time.time()
        attack_duration = 30

        # Create increasingly large payloads
        for size in [1024, 10240, 102400, 1048576]:  # 1KB to 1MB
            payload = "A" * size

            try:
                response = await self.client.post(
                    f"{self.target_url}/api/v1/resources",
                    json={"name": f"large_payload_{size}", "data": payload},
                )

                response_times.append(response.elapsed.total_seconds())
                total_requests += 1

                if response.status_code in [200, 201, 400, 413, 422]:
                    successful_requests += 1

            except Exception:
                total_requests += 1

            if time.time() - start_time > attack_duration:
                break

        system_metrics = system_monitor.stop()

        return AttackMetrics(
            response_times=response_times,
            success_rate=successful_requests / max(total_requests, 1),
            error_rate=1 - (successful_requests / max(total_requests, 1)),
            throughput=successful_requests / (time.time() - start_time),
            cpu_usage=system_metrics["cpu"],
            memory_usage=system_metrics["memory"],
            network_usage=system_metrics["network"],
            timestamp=datetime.now(),
        )

    def _calculate_performance_impact(self, metrics: AttackMetrics) -> Dict:
        """Calculate performance impact compared to baseline"""

        if not self.baseline_metrics:
            return {"error": "No baseline metrics available"}

        baseline_avg_response = statistics.mean(self.baseline_metrics.response_times)
        attack_avg_response = (
            statistics.mean(metrics.response_times) if metrics.response_times else 0
        )

        baseline_cpu = statistics.mean(self.baseline_metrics.cpu_usage)
        attack_cpu = statistics.mean(metrics.cpu_usage) if metrics.cpu_usage else 0

        baseline_memory = statistics.mean(self.baseline_metrics.memory_usage)
        attack_memory = statistics.mean(metrics.memory_usage) if metrics.memory_usage else 0

        return {
            "response_time_increase": (
                (attack_avg_response - baseline_avg_response) / baseline_avg_response
            )
            * 100,
            "success_rate_decrease": (
                (self.baseline_metrics.success_rate - metrics.success_rate)
                / self.baseline_metrics.success_rate
            )
            * 100,
            "throughput_decrease": (
                (self.baseline_metrics.throughput - metrics.throughput)
                / self.baseline_metrics.throughput
            )
            * 100,
            "cpu_usage_increase": ((attack_cpu - baseline_cpu) / baseline_cpu) * 100,
            "memory_usage_increase": ((attack_memory - baseline_memory) / baseline_memory) * 100,
        }

    def _evaluate_performance_threshold(self, metrics: AttackMetrics) -> bool:
        """Evaluate if performance meets thresholds during attack"""

        if not metrics.response_times:
            return False

        avg_response_time = statistics.mean(metrics.response_times)
        max_cpu = max(metrics.cpu_usage) if metrics.cpu_usage else 0
        max_memory = max(metrics.memory_usage) if metrics.memory_usage else 0

        # Check thresholds
        checks = [
            avg_response_time <= self.thresholds["max_response_time"],
            metrics.success_rate >= self.thresholds["min_success_rate"],
            max_cpu <= self.thresholds["max_cpu_usage"],
            max_memory <= self.thresholds["max_memory_usage"],
        ]

        # Check throughput ratio if baseline exists
        if self.baseline_metrics:
            throughput_ratio = metrics.throughput / self.baseline_metrics.throughput
            checks.append(throughput_ratio >= self.thresholds["min_throughput_ratio"])

        return all(checks)

    def _generate_recommendations(self, scenario: str, metrics: AttackMetrics) -> List[str]:
        """Generate performance recommendations based on attack results"""

        recommendations = []

        if metrics.response_times:
            avg_response = statistics.mean(metrics.response_times)
            if avg_response > self.thresholds["max_response_time"]:
                recommendations.append(
                    f"Response time ({avg_response:.2f}s) exceeds threshold. Consider implementing caching or optimizing queries."
                )

        if metrics.success_rate < self.thresholds["min_success_rate"]:
            recommendations.append(
                f"Success rate ({metrics.success_rate:.2%}) below threshold. Implement better error handling and rate limiting."
            )

        if metrics.cpu_usage and max(metrics.cpu_usage) > self.thresholds["max_cpu_usage"]:
            recommendations.append(
                "High CPU usage detected. Consider horizontal scaling or CPU optimization."
            )

        if metrics.memory_usage and max(metrics.memory_usage) > self.thresholds["max_memory_usage"]:
            recommendations.append(
                "High memory usage detected. Implement memory optimization and garbage collection tuning."
            )

        # Scenario-specific recommendations
        if "DDoS" in scenario:
            recommendations.append("Implement DDoS protection (rate limiting, IP filtering, CDN).")

        if "Brute Force" in scenario:
            recommendations.append("Implement account lockout and progressive delays.")

        if "SQL Injection" in scenario:
            recommendations.append(
                "Ensure parameterized queries and input validation are properly implemented."
            )

        if "Resource Exhaustion" in scenario:
            recommendations.append("Implement request size limits and resource quotas.")

        return recommendations

    def _generate_attack_performance_report(self, results: Dict) -> Dict:
        """Generate comprehensive attack performance report"""

        total_scenarios = len(results)
        passed_scenarios = sum(1 for r in results.values() if r.get("passed", False))

        report = {
            "summary": {
                "total_scenarios": total_scenarios,
                "passed_scenarios": passed_scenarios,
                "failed_scenarios": total_scenarios - passed_scenarios,
                "overall_resilience_score": (passed_scenarios / total_scenarios) * 100,
                "test_date": datetime.now().isoformat(),
            },
            "baseline_metrics": {
                "avg_response_time": (
                    statistics.mean(self.baseline_metrics.response_times)
                    if self.baseline_metrics
                    else 0
                ),
                "success_rate": self.baseline_metrics.success_rate if self.baseline_metrics else 0,
                "throughput": self.baseline_metrics.throughput if self.baseline_metrics else 0,
            },
            "attack_scenarios": results,
            "overall_recommendations": self._generate_overall_recommendations(results),
            "performance_thresholds": self.thresholds,
        }

        # Save report
        with open("performance_under_attack_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

        return report

    def _generate_overall_recommendations(self, results: Dict) -> List[str]:
        """Generate overall recommendations based on all attack results"""

        recommendations = []

        failed_scenarios = [
            name for name, result in results.items() if not result.get("passed", False)
        ]

        if len(failed_scenarios) > 3:
            recommendations.append(
                "Multiple attack scenarios failed. Consider comprehensive security hardening."
            )

        if any("DDoS" in scenario for scenario in failed_scenarios):
            recommendations.append(
                "Implement DDoS protection measures (CloudFlare, AWS Shield, etc.)."
            )

        if any("Brute Force" in scenario for scenario in failed_scenarios):
            recommendations.append(
                "Strengthen authentication mechanisms and implement account protection."
            )

        if any("Resource Exhaustion" in scenario for scenario in failed_scenarios):
            recommendations.append("Implement resource limits and monitoring.")

        recommendations.extend(
            [
                "Consider implementing a Web Application Firewall (WAF)",
                "Set up real-time monitoring and alerting for attacks",
                "Implement auto-scaling to handle traffic spikes",
                "Regular performance testing under attack conditions",
                "Create incident response procedures for security events",
            ]
        )

        return recommendations


class SystemMonitor:
    """Monitor system resources during attacks"""

    def __init__(self):
        self.monitoring = False
        self.metrics = {"cpu": [], "memory": [], "network": []}
        self.monitor_thread = None

    def start(self):
        """Start system monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()

    def stop(self) -> Dict:
        """Stop monitoring and return metrics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        return self.metrics

    def _monitor_loop(self):
        """Monitor system resources in a loop"""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.metrics["cpu"].append(cpu_percent)

                # Memory usage
                memory = psutil.virtual_memory()
                self.metrics["memory"].append(memory.percent)

                # Network usage (simplified)
                network = psutil.net_io_counters()
                self.metrics["network"].append(network.bytes_sent + network.bytes_recv)

            except Exception:
                pass

            time.sleep(1)


if __name__ == "__main__":

    async def main():
        tester = PerformanceUnderAttackTester()
        report = await tester.run_all_attack_scenarios()

        print("\n" + "=" * 80)
        print("PERFORMANCE UNDER ATTACK REPORT")
        print("=" * 80)
        print(f"Overall Resilience Score: {report['summary']['overall_resilience_score']:.1f}%")
        print(
            f"Passed Scenarios: {report['summary']['passed_scenarios']}/{report['summary']['total_scenarios']}"
        )

        print("\nFailed Scenarios:")
        for name, result in report["attack_scenarios"].items():
            if not result.get("passed", False):
                print(f"  - {name}")

        print("\nOverall Recommendations:")
        for rec in report["overall_recommendations"]:
            print(f"  - {rec}")

        print("\nDetailed report saved to: performance_under_attack_report.json")

    asyncio.run(main())
