."""
Deployment Verification

Verifies that deployed agents are functioning correctly on target hardware.
Includes health checks, functional tests, and performance validation.
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import psutil

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ServiceStatus(Enum):
    """Service status."""

    RUNNING = "running"
    STOPPED = "stopped"
    STARTING = "starting"
    STOPPING = "stopping"
    CRASHED = "crashed"


@dataclass
class HealthCheck:
    """Health check configuration."""

    name: str
    endpoint: str
    method: str = "GET"
    timeout: int = 5
    expected_status: int = 200
    interval: int = 30
    retries: int = 3


@dataclass
class ServiceInfo:
    """Service information."""

    name: str
    pid: Optional[int] = None
    status: ServiceStatus = ServiceStatus.STOPPED
    uptime: float = 0
    memory_mb: float = 0
    cpu_percent: float = 0
    port: Optional[int] = None
    health_status: HealthStatus = HealthStatus.UNKNOWN
    last_health_check: Optional[float] = None
    errors: List[str] = field(default_factory=list)


@dataclass
class DeploymentConfig:
    """Deployment configuration."""

    agent_name: str
    base_url: str
    health_checks: List[HealthCheck]
    pid_file: Path
    log_file: Path
    metrics_file: Optional[Path] = None
    startup_timeout: int = 120
    shutdown_timeout: int = 30


class HealthMonitor:
    """
    Monitors agent health and performance.
    """

    def __init__(self, config: DeploymentConfig) -> None:
        """Initialize health monitor."""
        self.config = config
        self.running = False
        self.health_history = []
        self.max_history = 100

    async def check_health(self) -> HealthStatus:
        """
        Perform health checks.

        Returns:
            Overall health status
        """
        results = []

        async with aiohttp.ClientSession() as session:
            for check in self.config.health_checks:
                result = await self._perform_health_check(session, check)
                results.append(result)

        # Determine overall status
        if all(r for r in results):
            return HealthStatus.HEALTHY
        elif any(r for r in results):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNHEALTHY

    async def _perform_health_check(
        self, session: aiohttp.ClientSession, check: HealthCheck
    ) -> bool:
        """Perform a single health check."""
        url = f"{self.config.base_url}{check.endpoint}"

        for attempt in range(check.retries):
            try:
                async with session.request(
                    check.method,
                    url,
                    timeout=aiohttp.ClientTimeout(total=check.timeout),
                ) as response:
                    if response.status == check.expected_status:
                        logger.debug(f"Health check {check.name} passed")
                        return True
                    else:
                        logger.warning(
                            f"Health check {check.name} failed: " f"status {response.status}"
                        )

            except asyncio.TimeoutError:
                logger.warning(f"Health check {check.name} timed out")
            except Exception as e:
                logger.error(f"Health check {check.name} error: {e}")

            if attempt < check.retries - 1:
                await asyncio.sleep(1)

        return False

    async def monitor_continuous(self, callback: Optional[callable] = None):
        """
        Continuously monitor health.

        Args:
            callback: Optional callback for health status changes
        """
        self.running = True
        last_status = HealthStatus.UNKNOWN

        while self.running:
            try:
                status = await self.check_health()

                # Record history
                self.health_history.append({"timestamp": time.time(),
                    "status": status.value})

                # Keep history bounded
                if len(self.health_history) > self.max_history:
                    self.health_history.pop(0)

                # Notify on status change
                if status != last_status and callback:
                    callback(status, last_status)

                last_status = status

                # Wait for next check
                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(10)

    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.running = False

    def get_health_metrics(self) -> Dict[str, Any]:
        """Get health metrics."""
        if not self.health_history:
            return {
                "status": HealthStatus.UNKNOWN.value,
                "uptime_percent": 0,
                "last_check": None,
            }

        # Calculate uptime percentage
        healthy_count = sum(
            1 for h in self.health_history if h["status"] == HealthStatus.HEALTHY.value
        )
        uptime_percent = (healthy_count / len(self.health_history)) * 100

        # Get current status
        current_status = self.health_history[-1]["status"]
        last_check = self.health_history[-1]["timestamp"]

        return {
            "status": current_status,
            "uptime_percent": uptime_percent,
            "last_check": last_check,
            "history_size": len(self.health_history),
        }


class ServiceManager:
    """
    Manages agent service lifecycle.
    """

    def __init__(self, config: DeploymentConfig) -> None:
        """Initialize service manager."""
        self.config = config
        self.service_info = ServiceInfo(name=config.agent_name)

    def start_service(self, command: List[str], env: Optional[Dict[str,
        str]] = None) -> bool:
        """
        Start the agent service.

        Args:
            command: Command to start service
            env: Environment variables

        Returns:
            True if started successfully
        """
        if self.is_running():
            logger.warning(f"Service {self.config.agent_name} already running")
            return True

        try:
            # Start process
            process = subprocess.Popen(
                command,
                env=env or os.environ.copy(),
                stdout=open(self.config.log_file, "a"),
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid if os.name != "nt" else None,
            )

            # Wait for startup
            self.service_info.status = ServiceStatus.STARTING
            start_time = time.time()

            while time.time() - start_time < self.config.startup_timeout:
                if process.poll() is not None:
                    # Process exited
                    self.service_info.status = ServiceStatus.CRASHED
                    self.service_info.errors.append(
                        f"Process exited with code {process.returncode}"
                    )
                    return False

                # Check if PID file created
                if self.config.pid_file.exists():
                    with open(self.config.pid_file) as f:
                        pid = int(f.read().strip())

                    self.service_info.pid = pid
                    self.service_info.status = ServiceStatus.RUNNING
                    self.service_info.uptime = 0

                    logger.info(f"Service {self.config.agent_name} started (PID: {pid})")
                    return True

                time.sleep(1)

            # Startup timeout
            process.terminate()
            self.service_info.status = ServiceStatus.STOPPED
            self.service_info.errors.append("Startup timeout")
            return False

        except Exception as e:
            logger.error(f"Failed to start service: {e}")
            self.service_info.status = ServiceStatus.STOPPED
            self.service_info.errors.append(str(e))
            return False

    def stop_service(self) -> bool:
        """Stop the agent service."""
        if not self.is_running():
            logger.warning(f"Service {self.config.agent_name} not running")
            return True

        try:
            self.service_info.status = ServiceStatus.STOPPING

            # Send SIGTERM
            os.kill(self.service_info.pid, 15)

            # Wait for graceful shutdown
            start_time = time.time()
            while time.time() - start_time < self.config.shutdown_timeout:
                try:
                    os.kill(self.service_info.pid, 0)  # Check if still running
                    time.sleep(1)
                except ProcessLookupError:
                    # Process stopped
                    break
            else:
                # Force kill if still running
                try:
                    os.kill(self.service_info.pid, 9)
                except ProcessLookupError:
                    pass

            # Clean up PID file
            if self.config.pid_file.exists():
                self.config.pid_file.unlink()

            self.service_info.pid = None
            self.service_info.status = ServiceStatus.STOPPED
            self.service_info.uptime = 0

            logger.info(f"Service {self.config.agent_name} stopped")
            return True

        except Exception as e:
            logger.error(f"Failed to stop service: {e}")
            self.service_info.errors.append(str(e))
            return False

    def restart_service(self, command: List[str], env: Optional[Dict[str,
        str]] = None) -> bool:
        """Restart the agent service."""
        logger.info(f"Restarting service {self.config.agent_name}")

        if self.is_running():
            if not self.stop_service():
                return False

            # Wait a bit before restarting
            time.sleep(2)

        return self.start_service(command, env)

    def is_running(self) -> bool:
        """Check if service is running."""
        if not self.service_info.pid:
            return False

        try:
            # Check if process exists
            os.kill(self.service_info.pid, 0)
            return True
        except ProcessLookupError:
            self.service_info.pid = None
            self.service_info.status = ServiceStatus.STOPPED
            return False

    def get_service_metrics(self) -> Dict[str, Any]:
        """Get service metrics."""
        if not self.is_running():
            return {
                "status": self.service_info.status.value,
                "uptime": 0,
                "memory_mb": 0,
                "cpu_percent": 0,
            }

        try:
            process = psutil.Process(self.service_info.pid)

            # Update metrics
            self.service_info.memory_mb = (
                process.memory_info().rss / (1024 * 1024))
            self.service_info.cpu_percent = process.cpu_percent(interval=1)
            self.service_info.uptime = time.time() - process.create_time()

            return {
                "status": self.service_info.status.value,
                "pid": self.service_info.pid,
                "uptime": self.service_info.uptime,
                "memory_mb": self.service_info.memory_mb,
                "cpu_percent": self.service_info.cpu_percent,
                "threads": process.num_threads(),
                "open_files": len(process.open_files()),
            }

        except Exception as e:
            logger.error(f"Failed to get service metrics: {e}")
            return {"status": self.service_info.status.value, "error": str(e)}


class FunctionalTester:
    """
    Runs functional tests on deployed agent.
    """

    def __init__(self, base_url: str) -> None:
        """Initialize functional tester."""
        self.base_url = base_url

    async def run_tests(self) -> Dict[str, Any]:
        """
        Run functional tests.

        Returns:
            Test results
        """
        tests = [
            self._test_agent_info,
            self._test_movement,
            self._test_perception,
            self._test_communication,
            self._test_decision_making,
        ]

        results = []

        async with aiohttp.ClientSession() as session:
            for test_func in tests:
                result = await test_func(session)
                results.append(result)

        # Calculate summary
        passed = sum(1 for r in results if r["passed"])
        total = len(results)

        return {
            "results": results,
            "summary": {
                "passed": passed,
                "failed": total - passed,
                "total": total,
                "success_rate": (passed / total) * 100 if total > 0 else 0,
            },
        }

    async def _test_agent_info(self, session: aiohttp.ClientSession) -> Dict[str,
        Any]:
        """Test agent info endpoint."""
        test_name = "agent_info"

        try:
            async with session.get(
                f"{self.base_url}/api/agent/info",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status == 200:
                    data = await response.json()

                    # Validate response structure
                    required_fields = ["id", "name", "class", "status"]
                    missing_fields = (
                        [f for f in required_fields if f not in data])

                    if missing_fields:
                        return {
                            "test": test_name,
                            "passed": False,
                            "message": f"Missing fields: {missing_fields}",
                        }

                    return {
                        "test": test_name,
                        "passed": True,
                        "message": "Agent info retrieved successfully",
                        "data": data,
                    }
                else:
                    return {
                        "test": test_name,
                        "passed": False,
                        "message": f"Unexpected status: {response.status}",
                    }

        except Exception as e:
            return {
                "test": test_name,
                "passed": False,
                "message": f"Test failed: {str(e)}",
            }

    async def _test_movement(self, session: aiohttp.ClientSession) -> Dict[str,
        Any]:
        """Test agent movement."""
        test_name = "movement"

        try:
            # Get current position
            async with session.get(f"{self.base_url}/api/agent/position") as response:
                if response.status != 200:
                    return {
                        "test": test_name,
                        "passed": False,
                        "message": "Failed to get current position",
                    }

                current_pos = await response.json()

            # Attempt movement
            move_data = {"direction": "north", "distance": 1}

            async with session.post(f"{self.base_url}/api/agent/move",
                json=move_data) as response:
                if response.status == 200:
                    new_pos = await response.json()

                    # Verify movement occurred
                    if new_pos != current_pos:
                        return {
                            "test": test_name,
                            "passed": True,
                            "message": "Movement successful",
                            "data": {"from": current_pos, "to": new_pos},
                        }
                    else:
                        return {
                            "test": test_name,
                            "passed": False,
                            "message": "Movement did not change position",
                        }
                else:
                    return {
                        "test": test_name,
                        "passed": False,
                        "message": f"Movement failed: {response.status}",
                    }

        except Exception as e:
            return {
                "test": test_name,
                "passed": False,
                "message": f"Test failed: {str(e)}",
            }

    async def _test_perception(self, session: aiohttp.ClientSession) -> Dict[str,
        Any]:
        """Test agent perception."""
        test_name = "perception"

        try:
            async with session.get(f"{self.base_url}/api/agent/perceive") as response:
                if response.status == 200:
                    perception_data = await response.json()

                    # Validate perception data
                    if "surroundings" in perception_data:
                        return {
                            "test": test_name,
                            "passed": True,
                            "message": "Perception working",
                            "data": perception_data,
                        }
                    else:
                        return {
                            "test": test_name,
                            "passed": False,
                            "message": "Invalid perception data structure",
                        }
                else:
                    return {
                        "test": test_name,
                        "passed": False,
                        "message": f"Perception failed: {response.status}",
                    }

        except Exception as e:
            return {
                "test": test_name,
                "passed": False,
                "message": f"Test failed: {str(e)}",
            }

    async def _test_communication(self, session: aiohttp.ClientSession) -> Dict[str,
        Any]:
        """Test agent communication."""
        test_name = "communication"

        try:
            # Send test message
            message_data = {
                "type": "greeting",
                "content": "Hello from functional test",
                "recipient": "broadcast",
            }

            async with session.post(
                f"{self.base_url}/api/agent/communicate", json=message_data
            ) as response:
                if response.status == 200:
                    result = await response.json()

                    if result.get("success"):
                        return {
                            "test": test_name,
                            "passed": True,
                            "message": "Communication successful",
                            "data": result,
                        }
                    else:
                        return {
                            "test": test_name,
                            "passed": False,
                            "message": "Communication reported failure",
                        }
                else:
                    return {
                        "test": test_name,
                        "passed": False,
                        "message": f"Communication failed: {response.status}",
                    }

        except Exception as e:
            return {
                "test": test_name,
                "passed": False,
                "message": f"Test failed: {str(e)}",
            }

    async def _test_decision_making(self, session: aiohttp.ClientSession) -> Dict[str,
        Any]:
        """Test agent decision making."""
        test_name = "decision_making"

        try:
            # Request decision for scenario
            scenario_data = {
                "situation": "low_resources",
                "options": ["explore", "trade", "conserve"],
                "context": {"food": 5, "water": 3, "energy": 10},
            }

            async with session.post(
                f"{self.base_url}/api/agent/decide", json=scenario_data
            ) as response:
                if response.status == 200:
                    decision = await response.json()

                    if "action" in decision and "reasoning" in decision:
                        return {
                            "test": test_name,
                            "passed": True,
                            "message": "Decision making functional",
                            "data": decision,
                        }
                    else:
                        return {
                            "test": test_name,
                            "passed": False,
                            "message": "Invalid decision structure",
                        }
                else:
                    return {
                        "test": test_name,
                        "passed": False,
                        "message": f"Decision making failed: {response.status}",
                    }

        except Exception as e:
            return {
                "test": test_name,
                "passed": False,
                "message": f"Test failed: {str(e)}",
            }


class DeploymentVerifier:
    """
    Comprehensive deployment verification.
    """

    def __init__(self, deployment_dir: Path) -> None:
        """Initialize deployment verifier."""
        self.deployment_dir = deployment_dir
        self.config = self._load_config()

        self.health_monitor = HealthMonitor(self.config)
        self.service_manager = ServiceManager(self.config)
        self.functional_tester = FunctionalTester(self.config.base_url)

    def _load_config(self) -> DeploymentConfig:
        """Load deployment configuration."""
        config_file = self.deployment_dir / "deployment_config.json"

        if config_file.exists():
            with open(config_file) as f:
                config_data = json.load(f)
        else:
            # Default configuration
            config_data = {
                "agent_name": "freeagentics_agent",
                "base_url": "http://localhost:8080",
                "health_checks": [
                    {
                        "name": "basic",
                        "endpoint": "/health",
                        "method": "GET",
                        "expected_status": 200,
                    }
                ],
            }

        # Create health check objects
        health_checks = (
            [HealthCheck(**hc) for hc in config_data.get("health_checks", [])])

        return DeploymentConfig(
            agent_name=config_data.get("agent_name", "agent"),
            base_url=config_data.get("base_url", "http://localhost:8080"),
            health_checks=health_checks,
            pid_file=self.deployment_dir / "agent.pid",
            log_file=self.deployment_dir / "agent.log",
            metrics_file=self.deployment_dir / "metrics.json",
            startup_timeout=config_data.get("startup_timeout", 120),
            shutdown_timeout=config_data.get("shutdown_timeout", 30),
        )

    async def verify_deployment(self) -> Dict[str, Any]:
        """
        Perform comprehensive deployment verification.

        Returns:
            Verification results
        """
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "deployment_dir": str(self.deployment_dir),
            "checks": {},
        }

        # 1. Check service status
        service_metrics = self.service_manager.get_service_metrics()
        results["checks"]["service"] = {
            "running": self.service_manager.is_running(),
            "metrics": service_metrics,
        }

        # 2. Perform health checks
        if self.service_manager.is_running():
            health_status = await self.health_monitor.check_health()
            health_metrics = self.health_monitor.get_health_metrics()

            results["checks"]["health"] = {
                "status": health_status.value,
                "metrics": health_metrics,
            }

            # 3. Run functional tests
            if health_status != HealthStatus.UNHEALTHY:
                test_results = await self.functional_tester.run_tests()
                results["checks"]["functional"] = test_results
            else:
                results["checks"]["functional"] = {
                    "skipped": True,
                    "reason": "Health check failed",
                }
        else:
            results["checks"]["health"] = {
                "status": HealthStatus.UNKNOWN.value,
                "reason": "Service not running",
            }
            results["checks"]["functional"] = {
                "skipped": True,
                "reason": "Service not running",
            }

        # 4. Check logs for errors
        log_errors = self._check_logs()
        results["checks"]["logs"] = log_errors

        # 5. Check resource usage
        resource_usage = self._check_resources()
        results["checks"]["resources"] = resource_usage

        # Generate overall status
        results["overall_status"] = (
            self._determine_overall_status(results["checks"]))

        # Save results
        results_file = self.deployment_dir / "verification_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        return results

    def _check_logs(self) -> Dict[str, Any]:
        """Check logs for errors."""
        if not self.config.log_file.exists():
            return {"checked": False, "reason": "Log file not found"}

        try:
            with open(self.config.log_file) as f:
                # Read last 1000 lines
                lines = f.readlines()[-1000:]

            error_count = 0
            warning_count = 0
            recent_errors = []

            for line in lines:
                line_lower = line.lower()
                if "error" in line_lower or "exception" in line_lower:
                    error_count += 1
                    if len(recent_errors) < 5:
                        recent_errors.append(line.strip())
                elif "warning" in line_lower or "warn" in line_lower:
                    warning_count += 1

            return {
                "checked": True,
                "error_count": error_count,
                "warning_count": warning_count,
                "recent_errors": recent_errors,
            }

        except Exception as e:
            return {"checked": False, "error": str(e)}

    def _check_resources(self) -> Dict[str, Any]:
        """Check system resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage(str(self.deployment_dir))

            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available / (1024 * 1024),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3),
            }

        except Exception as e:
            return {"error": str(e)}

    def _determine_overall_status(self, checks: Dict[str, Any]) -> str:
        """Determine overall deployment status."""
        # Service must be running
        if not checks.get("service", {}).get("running"):
            return "failed"

        # Health must be at least degraded
        health_status = checks.get("health", {}).get("status")
        if health_status == "unhealthy":
            return "unhealthy"

        # Functional tests should mostly pass
        functional = checks.get("functional", {})
        if not functional.get("skipped"):
            success_rate = functional.get("summary", {}).get("success_rate", 0)
            if success_rate < 50:
                return "degraded"

        # Check for errors in logs
        log_errors = checks.get("logs", {}).get("error_count", 0)
        if log_errors > 10:
            return "degraded"

        # Check resource usage
        resources = checks.get("resources", {})
        if resources.get("cpu_percent", 0) > 90 or resources.get("memory_percent",
            0) > 90:
            return "degraded"

        return "healthy"


async def verify_deployment(deployment_path: str) -> bool:
    """
    Verify a deployment.

    Args:
        deployment_path: Path to deployment directory

    Returns:
        True if deployment is healthy
    """
    verifier = DeploymentVerifier(Path(deployment_path))

    print(f"\n=== Verifying Deployment: {deployment_path} ===")

    results = await verifier.verify_deployment()

    # Print results
    print(f"\nService Status: {results['checks']['service']['running']}")

    if "health" in results["checks"]:
        print(f"Health Status: {results['checks']['health']['status']}")

    if "functional" in results["checks"] and not results["checks"]["functional"].get("skipped"):
        summary = results["checks"]["functional"]["summary"]
        print(f"Functional Tests: {summary['passed']}/{summary['total']} passed")

    if "logs" in results["checks"] and results["checks"]["logs"].get("checked"):
        logs = results["checks"]["logs"]
        print(f"Log Errors: {logs['error_count']} errors, {logs['warning_count']} warnings")

    if "resources" in results["checks"]:
        resources = results["checks"]["resources"]
        print(
            f"Resources: CPU {resources.get('cpu_percent', 0):.1f}%, "
            f"Memory {resources.get('memory_percent', 0):.1f}%"
        )

    print(f"\nOverall Status: {results['overall_status']}")
    print(f"Detailed results saved to: {deployment_path}/verification_results.json")

    return results["overall_status"] in ["healthy", "degraded"]
