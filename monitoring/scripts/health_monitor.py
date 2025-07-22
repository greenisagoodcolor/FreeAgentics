#!/usr/bin/env python3
"""
Health monitoring script for FreeAgentics.

Continuously monitors system health and triggers alerts when thresholds are exceeded.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict

import aiohttp
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class HealthMonitor:
    """Monitors FreeAgentics health and triggers alerts."""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.alert_history = []
        self.check_results = {}
        self.session = None

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load monitoring configuration."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    async def start(self):
        """Start health monitoring."""
        self.session = aiohttp.ClientSession()
        logger.info("Starting FreeAgentics health monitoring...")

        try:
            while True:
                await self.run_health_checks()
                await asyncio.sleep(self.config["global"]["evaluation_interval"])

        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Cleanup resources."""
        if self.session:
            await self.session.close()

    async def run_health_checks(self):
        """Run all configured health checks."""
        tasks = []

        # API health checks
        for endpoint_name, endpoint_config in self.config["health_checks"]["endpoints"].items():
            task = self.check_endpoint(endpoint_name, endpoint_config)
            tasks.append(task)

        # Synthetic monitoring
        for journey in (
            self.config["health_checks"].get("synthetic_monitoring", {}).get("user_journey", [])
        ):
            task = self.check_synthetic_journey(journey)
            tasks.append(task)

        # Run all checks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Health check failed: {result}")
            else:
                await self.process_check_result(result)

    async def check_endpoint(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check a single endpoint."""
        start_time = time.time()

        try:
            async with self.session.get(
                config["url"],
                timeout=aiohttp.ClientTimeout(total=config.get("timeout", 5)),
            ) as response:
                latency = (time.time() - start_time) * 1000

                result = {
                    "name": name,
                    "type": "endpoint",
                    "timestamp": datetime.utcnow().isoformat(),
                    "status_code": response.status,
                    "latency_ms": round(latency, 2),
                    "success": response.status == 200,
                }

                # Parse response body
                try:
                    body = await response.json()
                    result["response"] = body
                    result["health_status"] = body.get("status", "unknown")
                except (json.JSONDecodeError, ValueError):
                    result["response"] = await response.text()

                return result

        except asyncio.TimeoutError:
            return {
                "name": name,
                "type": "endpoint",
                "timestamp": datetime.utcnow().isoformat(),
                "success": False,
                "error": "timeout",
                "latency_ms": config.get("timeout", 5) * 1000,
            }
        except Exception as e:
            return {
                "name": name,
                "type": "endpoint",
                "timestamp": datetime.utcnow().isoformat(),
                "success": False,
                "error": str(e),
                "latency_ms": (time.time() - start_time) * 1000,
            }

    async def check_synthetic_journey(self, journey: Dict[str, Any]) -> Dict[str, Any]:
        """Run a synthetic user journey test."""
        start_time = time.time()
        journey_name = journey["name"]

        try:
            # Prepare request
            method = journey.get("method", "GET")
            url = f"{self.config['health_checks']['endpoints']['api']['url']}{journey['endpoint']}"

            # Add test data if needed
            data = None
            if method == "POST":
                data = self._get_test_data(journey_name)

            async with self.session.request(
                method=method,
                url=url,
                json=data,
                timeout=aiohttp.ClientTimeout(total=journey.get("timeout", 30)),
            ) as response:
                latency = (time.time() - start_time) * 1000

                return {
                    "name": journey_name,
                    "type": "synthetic",
                    "timestamp": datetime.utcnow().isoformat(),
                    "status_code": response.status,
                    "latency_ms": round(latency, 2),
                    "success": 200 <= response.status < 300,
                    "response": await response.json() if response.status == 200 else None,
                }

        except Exception as e:
            return {
                "name": journey_name,
                "type": "synthetic",
                "timestamp": datetime.utcnow().isoformat(),
                "success": False,
                "error": str(e),
                "latency_ms": (time.time() - start_time) * 1000,
            }

    def _get_test_data(self, journey_name: str) -> Dict[str, Any]:
        """Get test data for synthetic monitoring."""
        test_data = {
            "Create Agent": {
                "name": f"test-agent-{int(time.time())}",
                "type": "explorer",
                "config": {"exploration_rate": 0.1},
            },
            "Agent Inference": {
                "agent_id": "test-agent-1",
                "observation": [0, 1, 0, 0],
                "context": {"test": True},
            },
            "Coalition Formation": {
                "name": f"test-coalition-{int(time.time())}",
                "agents": ["test-agent-1", "test-agent-2"],
                "strategy": "collaborative",
            },
        }

        return test_data.get(journey_name, {})

    async def process_check_result(self, result: Dict[str, Any]):
        """Process health check result and trigger alerts if needed."""
        check_name = result["name"]

        # Store result
        self.check_results[check_name] = result

        # Check against thresholds
        if not result["success"]:
            await self.trigger_alert(
                level="critical",
                check_name=check_name,
                message=f"Health check failed: {result.get('error', 'Unknown error')}",
                details=result,
            )

        # Check latency thresholds
        elif result.get("latency_ms", 0) > 1000:
            await self.trigger_alert(
                level="warning",
                check_name=check_name,
                message=f"High latency detected: {result['latency_ms']}ms",
                details=result,
            )

        # Check specific health status
        elif result.get("health_status") == "unhealthy":
            await self.trigger_alert(
                level="critical",
                check_name=check_name,
                message="Service reported unhealthy status",
                details=result,
            )

    async def trigger_alert(
        self,
        level: str,
        check_name: str,
        message: str,
        details: Dict[str, Any],
    ):
        """Trigger an alert."""
        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "check_name": check_name,
            "message": message,
            "details": details,
        }

        # Log alert
        if level == "critical":
            logger.error(f"CRITICAL ALERT: {check_name} - {message}")
        else:
            logger.warning(f"WARNING: {check_name} - {message}")

        # Store alert
        self.alert_history.append(alert)

        # Send notifications
        await self.send_notifications(alert)

    async def send_notifications(self, alert: Dict[str, Any]):
        """Send alert notifications to configured channels."""
        # Slack notification
        if "slack" in self.config.get("integrations", {}):
            await self.send_slack_notification(alert)

        # PagerDuty notification
        if alert["level"] == "critical" and "pagerduty" in self.config.get("integrations", {}):
            await self.send_pagerduty_notification(alert)

    async def send_slack_notification(self, alert: Dict[str, Any]):
        """Send Slack notification."""
        webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        if not webhook_url:
            return

        color = "#ff0000" if alert["level"] == "critical" else "#ffaa00"

        payload = {
            "attachments": [
                {
                    "color": color,
                    "title": f"{alert['level'].upper()}: {alert['check_name']}",
                    "text": alert["message"],
                    "fields": [
                        {
                            "title": "Timestamp",
                            "value": alert["timestamp"],
                            "short": True,
                        },
                        {
                            "title": "Environment",
                            "value": os.getenv("ENVIRONMENT", "production"),
                            "short": True,
                        },
                    ],
                    "footer": "FreeAgentics Health Monitor",
                }
            ]
        }

        try:
            async with self.session.post(webhook_url, json=payload) as response:
                if response.status != 200:
                    logger.error(f"Failed to send Slack notification: {response.status}")
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")

    async def send_pagerduty_notification(self, alert: Dict[str, Any]):
        """Send PagerDuty notification."""
        routing_key = os.getenv("PAGERDUTY_ROUTING_KEY")
        if not routing_key:
            return

        payload = {
            "routing_key": routing_key,
            "event_action": "trigger",
            "payload": {
                "summary": f"{alert['check_name']}: {alert['message']}",
                "severity": "critical",
                "source": "freeagentics-health-monitor",
                "timestamp": alert["timestamp"],
                "custom_details": alert["details"],
            },
        }

        try:
            async with self.session.post(
                "https://events.pagerduty.com/v2/enqueue", json=payload
            ) as response:
                if response.status != 202:
                    logger.error(f"Failed to send PagerDuty notification: {response.status}")
        except Exception as e:
            logger.error(f"Error sending PagerDuty notification: {e}")

    def print_summary(self):
        """Print monitoring summary."""
        print("\n" + "=" * 80)
        print("FREEAGENTICS HEALTH MONITORING SUMMARY")
        print("=" * 80)
        print(f"Timestamp: {datetime.utcnow().isoformat()}")
        print(f"Environment: {os.getenv('ENVIRONMENT', 'production')}")
        print()

        # Check results
        print("Health Check Results:")
        print("-" * 80)
        for check_name, result in self.check_results.items():
            status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
            latency = result.get("latency_ms", "N/A")
            print(f"{check_name:<30} {status:<10} Latency: {latency}ms")

        # Recent alerts
        if self.alert_history:
            print("\nRecent Alerts:")
            print("-" * 80)
            for alert in self.alert_history[-5:]:
                print(f"[{alert['timestamp']}] {alert['level'].upper()}: {alert['message']}")

        print("=" * 80 + "\n")


async def main():
    """Main entry point."""
    config_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "/home/green/FreeAgentics/monitoring/config/monitoring_config.yaml"
    )

    monitor = HealthMonitor(config_path)

    # Print summary periodically
    async def print_summary_task():
        while True:
            await asyncio.sleep(60)  # Every minute
            monitor.print_summary()

    # Start summary task
    asyncio.create_task(print_summary_task())

    # Start monitoring
    await monitor.start()


if __name__ == "__main__":
    asyncio.run(main())
