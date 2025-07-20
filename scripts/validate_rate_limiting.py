#!/usr/bin/env python3
"""Rate Limiting Validation Script.

This script validates the rate limiting and DDoS protection implementation
by testing various scenarios and edge cases.
"""

import asyncio
import json
import sys
from typing import Any, Dict, List, Union

import aiohttp
import redis.asyncio as aioredis
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


class RateLimitingValidator:
    """Validates rate limiting implementation."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        redis_url: str = "redis://localhost:6379",
    ):
        self.base_url = base_url
        self.redis_url = redis_url
        self.redis_client: Any = None
        self.results: List[Dict[str, Any]] = []

    async def setup(self):
        """Setup connections for validation."""
        try:
            # Setup Redis connection
            self.redis_client = aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            console.print("[green]✓[/green] Connected to Redis")

            # Test API connection
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health") as response:
                    if response.status == 200:
                        console.print(
                            "[green]✓[/green] API server is accessible"
                        )
                    else:
                        console.print(
                            f"[red]✗[/red] API server returned {response.status}"
                        )

        except Exception as e:
            console.print(f"[red]✗[/red] Setup failed: {e}")
            return False

        return True

    async def cleanup(self):
        """Cleanup connections."""
        if self.redis_client:
            await self.redis_client.close()

    async def test_basic_rate_limiting(self) -> Dict[str, Any]:
        """Test basic rate limiting functionality."""
        console.print("\n[bold]Testing Basic Rate Limiting[/bold]")

        results = {
            "test_name": "Basic Rate Limiting",
            "passed": False,
            "details": [],
        }

        try:
            async with aiohttp.ClientSession() as session:
                # Make requests within limit
                success_count = 0
                for i in range(5):
                    async with session.get(
                        f"{self.base_url}/health"
                    ) as response:
                        if response.status == 200:
                            success_count += 1
                            headers = dict(response.headers)
                            if "X-RateLimit-Remaining-Minute" in headers:
                                remaining = int(
                                    headers["X-RateLimit-Remaining-Minute"]
                                )
                                results["details"].append(
                                    f"Request {i+1}: Success, {remaining} remaining"
                                )
                        await asyncio.sleep(0.1)

                if success_count == 5:
                    results["passed"] = True
                    results["details"].append(
                        "All requests within limit succeeded"
                    )
                else:
                    results["details"].append(
                        f"Only {success_count}/5 requests succeeded"
                    )

        except Exception as e:
            results["details"].append(f"Error: {e}")

        self.results.append(results)
        return results

    async def test_rate_limit_enforcement(self) -> Dict[str, Any]:
        """Test rate limit enforcement."""
        console.print("\n[bold]Testing Rate Limit Enforcement[/bold]")

        results = {
            "test_name": "Rate Limit Enforcement",
            "passed": False,
            "details": [],
        }

        try:
            async with aiohttp.ClientSession() as session:
                # Make many requests rapidly to trigger rate limiting
                rate_limited_count = 0
                success_count = 0

                # Test auth endpoint (stricter limits)
                tasks = []
                for i in range(20):
                    task = self._make_request(
                        session, f"{self.base_url}/api/v1/auth/me"
                    )
                    tasks.append(task)

                responses = await asyncio.gather(
                    *tasks, return_exceptions=True
                )

                for i, response in enumerate(responses):
                    if isinstance(response, Exception):
                        results["details"].append(
                            f"Request {i+1}: Error - {response}"
                        )
                        continue

                    if response["status"] == 429:
                        rate_limited_count += 1
                        results["details"].append(
                            f"Request {i+1}: Rate limited (429)"
                        )
                    elif response["status"] == 200:
                        success_count += 1
                        results["details"].append(
                            f"Request {i+1}: Success (200)"
                        )
                    else:
                        results["details"].append(
                            f"Request {i+1}: Status {response['status']}"
                        )

                if rate_limited_count > 0:
                    results["passed"] = True
                    results["details"].append(
                        f"Rate limiting triggered: {rate_limited_count} requests blocked"
                    )
                else:
                    results["details"].append(
                        "Rate limiting not triggered - this may indicate a problem"
                    )

        except Exception as e:
            results["details"].append(f"Error: {e}")

        self.results.append(results)
        return results

    async def test_different_endpoint_limits(self) -> Dict[str, Any]:
        """Test different rate limits for different endpoints."""
        console.print("\n[bold]Testing Different Endpoint Limits[/bold]")

        results = {
            "test_name": "Different Endpoint Limits",
            "passed": False,
            "details": [],
        }

        try:
            endpoints = [
                ("/health", "Static endpoint"),
                ("/api/v1/auth/me", "Auth endpoint"),
                ("/api/v1/system/info", "API endpoint"),
            ]

            endpoint_results = {}

            async with aiohttp.ClientSession() as session:
                for endpoint, description in endpoints:
                    # Make requests to each endpoint
                    tasks = []
                    for i in range(10):
                        task = self._make_request(
                            session, f"{self.base_url}{endpoint}"
                        )
                        tasks.append(task)

                    responses = await asyncio.gather(
                        *tasks, return_exceptions=True
                    )

                    success_count = 0
                    rate_limited_count = 0

                    for response in responses:
                        if isinstance(response, Exception):
                            continue

                        if response["status"] == 200:
                            success_count += 1
                        elif response["status"] == 429:
                            rate_limited_count += 1

                    endpoint_results[endpoint] = {
                        "description": description,
                        "success": success_count,
                        "rate_limited": rate_limited_count,
                    }

                    results["details"].append(
                        f"{description}: {success_count} success, {rate_limited_count} rate limited"
                    )

                # Check if different endpoints have different behaviors
                if (
                    len(
                        set(
                            r["rate_limited"]
                            for r in endpoint_results.values()
                        )
                    )
                    > 1
                ):
                    results["passed"] = True
                    results["details"].append(
                        "Different endpoints show different rate limiting behavior"
                    )
                else:
                    results["details"].append(
                        "All endpoints show similar rate limiting behavior"
                    )

        except Exception as e:
            results["details"].append(f"Error: {e}")

        self.results.append(results)
        return results

    async def test_redis_storage(self) -> Dict[str, Any]:
        """Test Redis storage for rate limiting."""
        console.print("\n[bold]Testing Redis Storage[/bold]")

        results = {
            "test_name": "Redis Storage",
            "passed": False,
            "details": [],
        }

        try:
            if not self.redis_client:
                results["details"].append("Redis client not available")
                self.results.append(results)
                return results

            # Check if rate limiting keys exist in Redis
            keys = await self.redis_client.keys("rate_limit:*")
            if keys:
                results["details"].append(
                    f"Found {len(keys)} rate limiting keys in Redis"
                )
                for key in keys[:5]:  # Show first 5 keys
                    key_str = key.decode() if isinstance(key, bytes) else key
                    results["details"].append(f"Key: {key_str}")
                results["passed"] = True
            else:
                results["details"].append(
                    "No rate limiting keys found in Redis"
                )

            # Check blocked keys
            blocked_keys = await self.redis_client.keys("blocked:*")
            if blocked_keys:
                results["details"].append(
                    f"Found {len(blocked_keys)} blocked keys in Redis"
                )
                for key in blocked_keys[:3]:  # Show first 3 keys
                    key_str = key.decode() if isinstance(key, bytes) else key
                    value = await self.redis_client.get(key)
                    if value:
                        try:
                            block_data = json.loads(value)
                            results["details"].append(
                                f"Blocked: {key_str} - {block_data.get('reason', 'Unknown')}"
                            )
                        except json.JSONDecodeError:
                            results["details"].append(
                                f"Blocked: {key_str} - Invalid JSON"
                            )

            # Check DDoS blocked keys
            ddos_keys = await self.redis_client.keys("ddos_blocked:*")
            if ddos_keys:
                results["details"].append(
                    f"Found {len(ddos_keys)} DDoS blocked keys in Redis"
                )

        except Exception as e:
            results["details"].append(f"Error: {e}")

        self.results.append(results)
        return results

    async def test_rate_limit_headers(self) -> Dict[str, Any]:
        """Test rate limiting headers."""
        console.print("\n[bold]Testing Rate Limit Headers[/bold]")

        results = {
            "test_name": "Rate Limit Headers",
            "passed": False,
            "details": [],
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health") as response:
                    headers = dict(response.headers)

                    expected_headers = [
                        "X-RateLimit-Limit-Minute",
                        "X-RateLimit-Limit-Hour",
                        "X-RateLimit-Remaining-Minute",
                        "X-RateLimit-Remaining-Hour",
                        "X-RateLimit-Reset",
                    ]

                    found_headers = 0
                    for header in expected_headers:
                        if header in headers:
                            found_headers += 1
                            results["details"].append(
                                f"✓ {header}: {headers[header]}"
                            )
                        else:
                            results["details"].append(f"✗ {header}: Missing")

                    if found_headers == len(expected_headers):
                        results["passed"] = True
                        results["details"].append(
                            "All expected rate limit headers present"
                        )
                    else:
                        results["details"].append(
                            f"Only {found_headers}/{len(expected_headers)} headers present"
                        )

        except Exception as e:
            results["details"].append(f"Error: {e}")

        self.results.append(results)
        return results

    async def test_websocket_rate_limiting(self) -> Dict[str, Any]:
        """Test WebSocket rate limiting."""
        console.print("\n[bold]Testing WebSocket Rate Limiting[/bold]")

        results = {
            "test_name": "WebSocket Rate Limiting",
            "passed": False,
            "details": [],
        }

        try:
            # This is a placeholder - would need actual WebSocket testing
            results["details"].append(
                "WebSocket rate limiting test not implemented"
            )
            results["details"].append(
                "Manual testing recommended for WebSocket endpoints"
            )

        except Exception as e:
            results["details"].append(f"Error: {e}")

        self.results.append(results)
        return results

    async def _make_request(
        self, session: aiohttp.ClientSession, url: str
    ) -> Dict[str, Any]:
        """Make a request and return response info."""
        try:
            async with session.get(url) as response:
                return {
                    "status": response.status,
                    "headers": dict(response.headers),
                    "body": await response.text()
                    if response.status != 200
                    else "Success",
                }
        except Exception as e:
            return {"status": -1, "headers": {}, "body": f"Error: {e}"}

    def generate_report(self) -> None:
        """Generate a validation report."""
        console.print(
            "\n[bold blue]Rate Limiting Validation Report[/bold blue]"
        )
        console.print("=" * 60)

        # Summary table
        table = Table(title="Test Results Summary")
        table.add_column("Test Name", style="cyan", no_wrap=True)
        table.add_column("Status", style="magenta")
        table.add_column("Details Count", justify="right", style="green")

        passed_count = 0
        total_count = len(self.results)

        for result in self.results:
            status = "✓ PASS" if result["passed"] else "✗ FAIL"
            status_color = "green" if result["passed"] else "red"

            table.add_row(
                result["test_name"],
                f"[{status_color}]{status}[/{status_color}]",
                str(len(result["details"])),
            )

            if result["passed"]:
                passed_count += 1

        console.print(table)

        # Overall summary
        overall_status = "PASS" if passed_count == total_count else "FAIL"
        status_color = "green" if passed_count == total_count else "red"

        console.print(
            f"\n[bold]Overall Status: [{status_color}]{overall_status}[/{status_color}][/bold]"
        )
        console.print(f"Tests Passed: {passed_count}/{total_count}")

        # Detailed results
        for result in self.results:
            if result["details"]:
                console.print(f"\n[bold]{result['test_name']}[/bold]")
                for detail in result["details"]:
                    console.print(f"  • {detail}")

        return passed_count == total_count

    async def run_all_tests(self) -> bool:
        """Run all validation tests."""
        console.print(
            "[bold green]Starting Rate Limiting Validation[/bold green]"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Setup
            task = progress.add_task("Setting up connections...", total=None)
            if not await self.setup():
                console.print("[red]Setup failed. Exiting.[/red]")
                return False
            progress.update(task, description="Setup complete")

            # Run tests
            tests = [
                (
                    "Testing basic rate limiting...",
                    self.test_basic_rate_limiting,
                ),
                (
                    "Testing rate limit enforcement...",
                    self.test_rate_limit_enforcement,
                ),
                (
                    "Testing different endpoint limits...",
                    self.test_different_endpoint_limits,
                ),
                ("Testing Redis storage...", self.test_redis_storage),
                (
                    "Testing rate limit headers...",
                    self.test_rate_limit_headers,
                ),
                (
                    "Testing WebSocket rate limiting...",
                    self.test_websocket_rate_limiting,
                ),
            ]

            for description, test_func in tests:
                progress.update(task, description=description)
                await test_func()
                await asyncio.sleep(0.5)  # Small delay between tests

            progress.update(task, description="Cleaning up...")
            await self.cleanup()
            progress.update(task, description="Validation complete")

        # Generate report
        return self.generate_report()


async def main():
    """Main validation function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate rate limiting implementation"
    )
    parser.add_argument(
        "--base-url", default="http://localhost:8000", help="Base URL for API"
    )
    parser.add_argument(
        "--redis-url", default="redis://localhost:6379", help="Redis URL"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick validation only"
    )

    args = parser.parse_args()

    validator = RateLimitingValidator(args.base_url, args.redis_url)

    try:
        success = await validator.run_all_tests()

        if success:
            console.print(
                "\n[bold green]✓ All tests passed! Rate limiting is working correctly.[/bold green]"
            )
            sys.exit(0)
        else:
            console.print(
                "\n[bold red]✗ Some tests failed. Please check the implementation.[/bold red]"
            )
            sys.exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Validation interrupted by user.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Validation failed with error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
