#!/usr/bin/env python3
"""
Script to test rate limiting functionality.
This validates that rate limiting is properly configured and working.
"""

import asyncio
import os
import sys
import time

import httpx

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RateLimitTester:
    """Test rate limiting functionality."""

    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.results = {
            "total_requests": 0,
            "successful_requests": 0,
            "rate_limited_requests": 0,
            "errors": 0,
            "response_times": [],
            "rate_limit_headers": [],
        }

    async def make_request(self, endpoint, headers=None):
        """Make a single request and record results."""
        start_time = time.time()

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}{endpoint}",
                    headers=headers or {},
                    timeout=5.0,
                )

                elapsed_time = time.time() - start_time
                self.results["total_requests"] += 1
                self.results["response_times"].append(elapsed_time)

                # Check for rate limit headers
                rate_limit_headers = {
                    "X-RateLimit-Limit": response.headers.get("X-RateLimit-Limit"),
                    "X-RateLimit-Remaining": response.headers.get("X-RateLimit-Remaining"),
                    "X-RateLimit-Reset": response.headers.get("X-RateLimit-Reset"),
                    "Retry-After": response.headers.get("Retry-After"),
                }

                if any(rate_limit_headers.values()):
                    self.results["rate_limit_headers"].append(rate_limit_headers)

                if response.status_code == 429:
                    self.results["rate_limited_requests"] += 1
                    return False, response.status_code, rate_limit_headers
                elif response.status_code == 200:
                    self.results["successful_requests"] += 1
                    return True, response.status_code, rate_limit_headers
                else:
                    self.results["errors"] += 1
                    return False, response.status_code, rate_limit_headers

            except Exception as e:
                self.results["errors"] += 1
                self.results["total_requests"] += 1
                return False, None, {"error": str(e)}

    async def test_endpoint_rate_limit(
        self, endpoint, requests_per_test=10, delay_between_requests=0.1
    ):
        """Test rate limiting for a specific endpoint."""
        print(f"\nTesting endpoint: {endpoint}")
        print(f"Making {requests_per_test} requests with {delay_between_requests}s delay...")

        for i in range(requests_per_test):
            success, status, headers = await self.make_request(endpoint)

            if status == 429:
                print(
                    f"  Request {i + 1}: Rate limited! Retry-After: {headers.get('Retry-After', 'N/A')}s"
                )
            elif success:
                remaining = headers.get("X-RateLimit-Remaining", "N/A")
                print(f"  Request {i + 1}: Success. Remaining: {remaining}")
            else:
                print(f"  Request {i + 1}: Error. Status: {status}")

            if i < requests_per_test - 1:
                await asyncio.sleep(delay_between_requests)

    async def test_burst_requests(self, endpoint, burst_size=20):
        """Test burst request handling."""
        print(f"\nTesting burst requests on endpoint: {endpoint}")
        print(f"Sending {burst_size} concurrent requests...")

        tasks = []
        for i in range(burst_size):
            task = self.make_request(endpoint)
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        successful = sum(1 for r in results if r[0])
        rate_limited = sum(1 for r in results if r[1] == 429)

        print(f"  Successful: {successful}/{burst_size}")
        print(f"  Rate limited: {rate_limited}/{burst_size}")

        return successful, rate_limited

    async def test_authenticated_vs_anonymous(self, endpoint, auth_token=None):
        """Test different rate limits for authenticated vs anonymous users."""
        print(f"\nTesting authenticated vs anonymous rate limits on: {endpoint}")

        # Test anonymous requests
        print("\nAnonymous requests:")
        self.reset_results()
        await self.test_endpoint_rate_limit(
            endpoint, requests_per_test=10, delay_between_requests=0.1
        )
        anonymous_results = self.results.copy()

        if auth_token:
            # Test authenticated requests
            print("\nAuthenticated requests:")
            self.reset_results()
            headers = {"Authorization": f"Bearer {auth_token}"}

            for i in range(10):
                success, status, rate_headers = await self.make_request(endpoint, headers)
                if status == 429:
                    print(f"  Request {i + 1}: Rate limited!")
                else:
                    print(
                        f"  Request {i + 1}: Success. Remaining: {rate_headers.get('X-RateLimit-Remaining', 'N/A')}"
                    )
                await asyncio.sleep(0.1)

            authenticated_results = self.results.copy()

            # Compare results
            print("\nComparison:")
            print(
                f"  Anonymous - Successful: {anonymous_results['successful_requests']}, Rate limited: {anonymous_results['rate_limited_requests']}"
            )
            print(
                f"  Authenticated - Successful: {authenticated_results['successful_requests']}, Rate limited: {authenticated_results['rate_limited_requests']}"
            )

    def reset_results(self):
        """Reset results for new test."""
        self.results = {
            "total_requests": 0,
            "successful_requests": 0,
            "rate_limited_requests": 0,
            "errors": 0,
            "response_times": [],
            "rate_limit_headers": [],
        }

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 50)
        print("RATE LIMITING TEST SUMMARY")
        print("=" * 50)
        print(f"Total requests made: {self.results['total_requests']}")
        print(f"Successful requests: {self.results['successful_requests']}")
        print(f"Rate limited requests: {self.results['rate_limited_requests']}")
        print(f"Errors: {self.results['errors']}")

        if self.results["response_times"]:
            avg_response_time = sum(self.results["response_times"]) / len(
                self.results["response_times"]
            )
            print(f"Average response time: {avg_response_time:.3f}s")

        if self.results["rate_limit_headers"]:
            print("\nRate limit headers detected:")
            sample_headers = self.results["rate_limit_headers"][0]
            for key, value in sample_headers.items():
                if value:
                    print(f"  {key}: {value}")


async def main():
    """Run rate limiting tests."""
    tester = RateLimitTester()

    # Test different endpoints
    endpoints_to_test = [
        "/health",
        "/api/v1/auth/login",
        "/api/v1/agents",
        "/docs",
    ]

    print("Starting rate limiting tests...")
    print("Make sure the FastAPI application is running on http://localhost:8000")
    print("Make sure Redis is running for distributed rate limiting")

    # Test each endpoint
    for endpoint in endpoints_to_test:
        await tester.test_endpoint_rate_limit(endpoint)
        tester.reset_results()

    # Test burst requests on auth endpoint
    await tester.test_burst_requests("/api/v1/auth/login", burst_size=10)

    # Test authenticated vs anonymous (if you have a valid token)
    # await tester.test_authenticated_vs_anonymous("/api/v1/agents", auth_token="your_token_here")

    # Print final summary
    tester.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
