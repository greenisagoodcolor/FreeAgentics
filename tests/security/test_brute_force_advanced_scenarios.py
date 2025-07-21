"""Advanced Brute Force Protection Test Scenarios.

This module contains specialized tests for complex brute force scenarios including:
- Timing attack prevention
- Distributed coordinated attacks
- Account takeover protection
- Zero-day pattern detection
- Adaptive protection mechanisms
"""

import asyncio
import random
import time
from collections import defaultdict

import numpy as np
import pytest
from httpx import AsyncClient


class TestTimingAttackPrevention:
    """Test prevention of timing-based attacks."""

    @pytest.fixture
    async def timing_client(self, app):
        """Create client for precise timing measurements."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client

    @pytest.mark.asyncio
    async def test_user_enumeration_timing_protection(self, timing_client):
        """Test protection against user enumeration via timing attacks."""
        # Test with valid and invalid usernames
        valid_username = "existing_user@example.com"
        invalid_usernames = [
            "nonexistent1@example.com",
            "nonexistent2@example.com",
            "nonexistent3@example.com",
        ]

        # Register valid user
        await timing_client.post(
            "/api/v1/auth/register",
            json={"email": valid_username, "password": "test_password_123"},
        )

        # Measure timing for valid username
        valid_timings = []
        for _ in range(20):
            start = time.perf_counter()
            await timing_client.post(
                "/api/v1/auth/login",
                json={
                    "username": valid_username,
                    "password": "wrong_password",
                },
            )
            valid_timings.append(time.perf_counter() - start)

        # Measure timing for invalid usernames
        invalid_timings = []
        for username in invalid_usernames:
            for _ in range(20):
                start = time.perf_counter()
                await timing_client.post(
                    "/api/v1/auth/login",
                    json={"username": username, "password": "wrong_password"},
                )
                invalid_timings.append(time.perf_counter() - start)

        # Calculate statistics
        valid_mean = np.mean(valid_timings)
        valid_std = np.std(valid_timings)
        invalid_mean = np.mean(invalid_timings)
        invalid_std = np.std(invalid_timings)

        # Timing should be statistically indistinguishable
        # Using t-test would be more rigorous, but simple comparison for now
        timing_diff = abs(valid_mean - invalid_mean)
        combined_std = np.sqrt((valid_std**2 + invalid_std**2) / 2)

        # Difference should be less than 2 standard deviations
        assert (
            timing_diff < 2 * combined_std
        ), f"Timing difference {timing_diff} exceeds threshold {2 * combined_std}"

    @pytest.mark.asyncio
    async def test_password_length_timing_protection(self, timing_client):
        """Test protection against password length timing attacks."""
        username = "timing_test@example.com"

        # Register user
        await timing_client.post(
            "/api/v1/auth/register",
            json={"email": username, "password": "correct_password_123"},
        )

        # Test different password lengths
        password_lengths = [1, 5, 10, 20, 50, 100, 200]
        timing_by_length = defaultdict(list)

        for length in password_lengths:
            test_password = "a" * length

            for _ in range(10):
                start = time.perf_counter()
                await timing_client.post(
                    "/api/v1/auth/login",
                    json={"username": username, "password": test_password},
                )
                timing_by_length[length].append(time.perf_counter() - start)

        # Verify no correlation between password length and timing
        lengths = []
        mean_timings = []

        for length, timings in sorted(timing_by_length.items()):
            lengths.append(length)
            mean_timings.append(np.mean(timings))

        # Calculate correlation coefficient
        correlation = np.corrcoef(lengths, mean_timings)[0, 1]

        # Correlation should be near zero (no relationship)
        assert (
            abs(correlation) < 0.3
        ), f"Password length correlation {correlation} indicates timing leak"

    @pytest.mark.asyncio
    async def test_hash_computation_timing_protection(self, timing_client):
        """Test protection against hash computation timing attacks."""
        # Create users with different password complexities
        users = [
            {"email": "simple@example.com", "password": "a"},
            {"email": "medium@example.com", "password": "a" * 50},
            {"email": "complex@example.com", "password": "A1b2C3d4!" * 10},
        ]

        # Register users
        for user in users:
            await timing_client.post("/api/v1/auth/register", json=user)

        # Test login timing for each
        timing_results = {}

        for user in users:
            timings = []

            for _ in range(30):
                start = time.perf_counter()
                await timing_client.post(
                    "/api/v1/auth/login",
                    json={
                        "username": user["email"],
                        "password": "wrong_password",
                    },
                )
                timings.append(time.perf_counter() - start)

            timing_results[user["email"]] = {
                "mean": np.mean(timings),
                "std": np.std(timings),
                "password_length": len(user["password"]),
            }

        # All users should have similar timing regardless of password complexity
        means = [r["mean"] for r in timing_results.values()]
        max_diff = max(means) - min(means)
        avg_std = np.mean([r["std"] for r in timing_results.values()])

        assert (
            max_diff < 3 * avg_std
        ), "Hash computation timing varies too much with password complexity"


class TestDistributedCoordinatedAttacks:
    """Test protection against distributed and coordinated attacks."""

    @pytest.mark.asyncio
    async def test_botnet_simulation(self, client, redis_client):
        """Test protection against botnet-style distributed attacks."""
        # Simulate botnet with multiple IPs
        botnet_size = 50
        botnet_ips = [f"192.168.{i//256}.{i%256}" for i in range(1, botnet_size + 1)]

        # Target account
        target_email = "botnet_target@example.com"

        # Each bot tries a few passwords
        passwords_per_bot = 3
        attack_results = []

        # Simulate coordinated attack
        for bot_num, ip in enumerate(botnet_ips):
            headers = {"X-Real-IP": ip}

            for pwd_num in range(passwords_per_bot):
                password = f"bot{bot_num}_try{pwd_num}"

                response = await client.post(
                    "/api/v1/auth/login",
                    json={"username": target_email, "password": password},
                    headers=headers,
                )

                attack_results.append(
                    {
                        "ip": ip,
                        "attempt": pwd_num,
                        "status": response.status_code,
                        "timestamp": time.time(),
                    }
                )

                # Small delay to simulate real botnet
                await asyncio.sleep(random.uniform(0.01, 0.05))

        # Analyze attack detection
        blocked_ips = set(r["ip"] for r in attack_results if r["status"] == 429)
        success_rate = sum(1 for r in attack_results if r["status"] != 429) / len(attack_results)

        # Should detect coordinated attack pattern
        assert len(blocked_ips) > botnet_size * 0.5, "Should block majority of botnet IPs"
        assert success_rate < 0.2, "Botnet success rate should be low"

        # Check if target account is protected
        legit_response = await client.post(
            "/api/v1/auth/login",
            json={"username": target_email, "password": "legitimate_password"},
            headers={"X-Real-IP": "10.0.0.1"},  # Different IP
        )

        # Account should have elevated protection
        assert legit_response.status_code in [
            403,
            429,
        ], "Target account should be protected"

    @pytest.mark.asyncio
    async def test_rotating_proxy_attack(self, client, redis_client):
        """Test detection of attacks using rotating proxies."""
        # Simulate proxy rotation
        proxy_pool = [{"ip": f"proxy-{i}.example.com", "port": 8000 + i} for i in range(20)]

        # Attack parameters
        target_accounts = [
            "user1@example.com",
            "user2@example.com",
            "user3@example.com",
        ]

        rotation_results = []

        # Rotate through proxies
        for attempt in range(100):
            proxy = proxy_pool[attempt % len(proxy_pool)]
            target = target_accounts[attempt % len(target_accounts)]

            headers = {
                "X-Real-IP": proxy["ip"],
                "X-Forwarded-For": f"{proxy['ip']}, 10.0.0.1",
                "Via": f"1.1 {proxy['ip']}:{proxy['port']}",
            }

            response = await client.post(
                "/api/v1/auth/login",
                json={"username": target, "password": f"attempt_{attempt}"},
                headers=headers,
            )

            rotation_results.append(
                {
                    "proxy": proxy["ip"],
                    "target": target,
                    "status": response.status_code,
                }
            )

        # Should detect proxy rotation pattern
        unique_proxies_blocked = len(
            set(r["proxy"] for r in rotation_results if r["status"] == 429)
        )

        assert unique_proxies_blocked > 10, "Should detect and block rotating proxies"

    @pytest.mark.asyncio
    async def test_coordinated_timing_attack(self, client, redis_client):
        """Test detection of coordinated attacks with specific timing."""
        # Simulate coordinated attack waves
        wave_size = 20
        num_waves = 5
        wave_delay = 2.0  # seconds between waves

        wave_results = []

        for wave in range(num_waves):
            # Launch wave
            wave_start = time.time()
            wave_tasks = []

            async with asyncio.TaskGroup() as tg:
                for i in range(wave_size):
                    ip = f"wave{wave}-bot{i}"
                    task = tg.create_task(
                        client.post(
                            "/api/v1/auth/login",
                            json={
                                "username": "coordinated_target@example.com",
                                "password": f"wave{wave}_attempt{i}",
                            },
                            headers={"X-Real-IP": ip},
                        )
                    )
                    wave_tasks.append(task)

            # Collect results
            for i, task in enumerate(wave_tasks):
                result = task.result()
                wave_results.append(
                    {
                        "wave": wave,
                        "bot": i,
                        "status": result.status_code,
                        "timestamp": time.time() - wave_start,
                    }
                )

            # Wait before next wave
            if wave < num_waves - 1:
                await asyncio.sleep(wave_delay)

        # Analyze wave pattern detection
        waves_blocked = defaultdict(int)
        for result in wave_results:
            if result["status"] == 429:
                waves_blocked[result["wave"]] += 1

        # Later waves should have higher block rates
        block_rates = [waves_blocked[w] / wave_size for w in range(num_waves)]

        # Block rate should increase with each wave
        for i in range(1, len(block_rates)):
            assert (
                block_rates[i] >= block_rates[i - 1]
            ), "Protection should strengthen with repeated waves"


class TestAccountTakeoverProtection:
    """Test protection against account takeover attempts."""

    @pytest.mark.asyncio
    async def test_credential_spray_detection(self, client, redis_client):
        """Test detection of password spray attacks."""
        # Create multiple target accounts
        target_accounts = []
        for i in range(20):
            email = f"spray_target_{i}@example.com"
            await client.post(
                "/api/v1/auth/register",
                json={"email": email, "password": f"unique_password_{i}"},
            )
            target_accounts.append(email)

        # Common passwords to spray
        common_passwords = [
            "Password123",
            "Welcome1",
            "Summer2024",
            "Company123",
            "Qwerty123",
            "Admin123",
            "Test123",
            "Default1",
        ]

        spray_results = []

        # Spray each password across all accounts
        for password in common_passwords:
            for email in target_accounts:
                response = await client.post(
                    "/api/v1/auth/login",
                    json={"username": email, "password": password},
                )

                spray_results.append(
                    {
                        "email": email,
                        "password": password,
                        "status": response.status_code,
                    }
                )

                # Small delay to avoid obvious rate limiting
                await asyncio.sleep(0.1)

        # Analyze spray detection
        successful_attempts = sum(1 for r in spray_results if r["status"] == 200)
        blocked_attempts = sum(1 for r in spray_results if r["status"] == 429)

        # Should detect spray pattern
        assert (
            blocked_attempts > len(spray_results) * 0.5
        ), "Password spray pattern should be detected"
        assert successful_attempts == 0, "No spray attempts should succeed"

    @pytest.mark.asyncio
    async def test_account_lockout_evasion_detection(self, client, redis_client):
        """Test detection of lockout evasion techniques."""
        target_email = "evasion_target@example.com"

        # Register target
        await client.post(
            "/api/v1/auth/register",
            json={"email": target_email, "password": "correct_password_123"},
        )

        # Evasion techniques
        evasion_results = []

        # Technique 1: Just below threshold attacks
        for i in range(2):  # Just below lockout threshold
            response = await client.post(
                "/api/v1/auth/login",
                json={"username": target_email, "password": f"wrong_{i}"},
            )
            evasion_results.append(("below_threshold", response.status_code))

        # Wait to reset counter
        await asyncio.sleep(3)

        # Technique 2: Case variations
        email_variations = [
            target_email.upper(),
            target_email.lower(),
            target_email.title(),
            "EVASION_target@example.com",
        ]

        for variant in email_variations:
            response = await client.post(
                "/api/v1/auth/login",
                json={"username": variant, "password": "guess"},
            )
            evasion_results.append(("case_variation", response.status_code))

        # Technique 3: Unicode variations
        unicode_variants = [
            "evasion_tаrget@example.com",  # Cyrillic 'a'
            "evasion_target@exаmple.com",  # Cyrillic 'a'
        ]

        for variant in unicode_variants:
            response = await client.post(
                "/api/v1/auth/login",
                json={"username": variant, "password": "guess"},
            )
            evasion_results.append(("unicode_variation", response.status_code))

        # Should detect evasion attempts
        blocked = sum(1 for _, status in evasion_results if status == 429)
        assert blocked > 0, "Evasion techniques should be detected"

    @pytest.mark.asyncio
    async def test_session_fixation_protection(self, client, redis_client):
        """Test protection against session fixation attacks."""
        # Create victim account
        victim_email = "victim@example.com"
        victim_password = "victim_password_123"

        await client.post(
            "/api/v1/auth/register",
            json={"email": victim_email, "password": victim_password},
        )

        # Attacker tries to fix session
        attacker_session_id = "fixed_session_12345"

        # Attempt to set fixed session
        response = await client.post(
            "/api/v1/auth/login",
            json={"username": victim_email, "password": victim_password},
            cookies={"session_id": attacker_session_id},
        )

        # Extract actual session ID
        if response.status_code == 200:
            actual_session = response.cookies.get("session_id")

            # Session ID should be different (regenerated)
            assert actual_session != attacker_session_id, "Session should be regenerated on login"

        # Try to hijack with fixed session
        hijack_response = await client.get(
            "/api/v1/users/me", cookies={"session_id": attacker_session_id}
        )

        assert hijack_response.status_code == 401, "Fixed session should not be valid"


class TestZeroDayPatternDetection:
    """Test detection of novel attack patterns."""

    @pytest.mark.asyncio
    async def test_anomaly_detection(self, client, redis_client):
        """Test anomaly detection for unusual attack patterns."""
        # Establish baseline behavior
        baseline_ips = [f"normal_user_{i}" for i in range(10)]

        for ip in baseline_ips:
            # Normal login attempts
            await client.post(
                "/api/v1/auth/login",
                json={"username": f"{ip}@example.com", "password": "password"},
                headers={"X-Real-IP": ip},
            )
            await asyncio.sleep(1)  # Normal user delay

        # Introduce anomalous patterns
        anomaly_results = []

        # Pattern 1: Unusual user agent cycling
        user_agents = [
            "CustomBot/1.0",
            "Mozilla/5.0 (compatible; Attackerbot/2.0)",
            "python-requests/2.28.0",
            "curl/7.64.0",
        ]

        for ua in user_agents * 5:
            response = await client.post(
                "/api/v1/auth/login",
                json={"username": "anomaly1@example.com", "password": "test"},
                headers={"User-Agent": ua},
            )
            anomaly_results.append(("ua_cycling", response.status_code))

        # Pattern 2: Unusual geographic distribution
        geo_ips = [
            "1.2.3.4",  # Asia
            "185.2.3.4",  # Europe
            "201.2.3.4",  # South America
            "41.2.3.4",  # Africa
            "203.2.3.4",  # Oceania
        ]

        for ip in geo_ips:
            response = await client.post(
                "/api/v1/auth/login",
                json={"username": "anomaly2@example.com", "password": "test"},
                headers={"X-Real-IP": ip},
            )
            anomaly_results.append(("geo_anomaly", response.status_code))

        # Should detect anomalous patterns
        blocked = sum(1 for _, status in anomaly_results if status == 429)
        assert blocked > 0, "Anomalous patterns should trigger protection"

    @pytest.mark.asyncio
    async def test_ml_pattern_learning(self, client, redis_client):
        """Test machine learning-based pattern detection."""
        # Simulate evolving attack pattern
        attack_phases = {
            "reconnaissance": {
                "endpoints": ["/api/v1/", "/api/v1/docs", "/api/v1/version"],
                "rate": 0.5,  # requests per second
            },
            "enumeration": {
                "endpoints": [
                    "/api/v1/users",
                    "/api/v1/users/1",
                    "/api/v1/users/admin",
                ],
                "rate": 1.0,
            },
            "exploitation": {
                "endpoints": ["/api/v1/auth/login", "/api/v1/auth/reset"],
                "rate": 2.0,
            },
        }

        ml_results = []

        for phase_name, phase_config in attack_phases.items():
            phase_start = time.time()

            for endpoint in phase_config["endpoints"] * 10:
                response = await client.get(endpoint)

                ml_results.append(
                    {
                        "phase": phase_name,
                        "endpoint": endpoint,
                        "status": response.status_code,
                        "timestamp": time.time() - phase_start,
                    }
                )

                # Maintain attack rate
                await asyncio.sleep(1.0 / phase_config["rate"])

        # Should adapt and detect evolving pattern
        phase_block_rates = {}

        for phase in attack_phases:
            phase_results = [r for r in ml_results if r["phase"] == phase]
            blocked = sum(1 for r in phase_results if r["status"] == 429)
            phase_block_rates[phase] = blocked / len(phase_results) if phase_results else 0

        # Later phases should have higher block rates (adaptive learning)
        assert (
            phase_block_rates["exploitation"] > phase_block_rates["reconnaissance"]
        ), "Protection should adapt and strengthen"


class TestAdaptiveProtectionMechanisms:
    """Test adaptive and self-tuning protection mechanisms."""

    @pytest.mark.asyncio
    async def test_dynamic_threshold_adjustment(self, client, redis_client):
        """Test dynamic adjustment of protection thresholds."""
        # Simulate varying attack intensities
        attack_periods = [
            {"name": "low", "rate": 0.5, "duration": 5},
            {"name": "medium", "rate": 2.0, "duration": 5},
            {"name": "high", "rate": 10.0, "duration": 5},
            {"name": "spike", "rate": 50.0, "duration": 2},
        ]

        threshold_results = []

        for period in attack_periods:
            period_start = time.time()
            period_results = []

            while time.time() - period_start < period["duration"]:
                response = await client.post(
                    "/api/v1/auth/login",
                    json={
                        "username": "adaptive_test@example.com",
                        "password": "wrong",
                    },
                )

                period_results.append(
                    {
                        "status": response.status_code,
                        "headers": dict(response.headers),
                    }
                )

                await asyncio.sleep(1.0 / period["rate"])

            threshold_results.append(
                {
                    "period": period["name"],
                    "results": period_results,
                    "block_rate": sum(1 for r in period_results if r["status"] == 429)
                    / len(period_results),
                }
            )

        # Thresholds should adapt to attack intensity
        for i, result in enumerate(threshold_results):
            if i > 0 and result["period"] != "low":
                # Higher intensity should trigger more blocking
                assert (
                    result["block_rate"] > 0.5
                ), f"Period {result['period']} should have high block rate"

    @pytest.mark.asyncio
    async def test_reputation_based_filtering(self, client, redis_client):
        """Test reputation-based request filtering."""
        # Build IP reputation
        good_ips = [f"good_ip_{i}" for i in range(5)]
        bad_ips = [f"bad_ip_{i}" for i in range(5)]

        # Establish good reputation
        for ip in good_ips:
            for _ in range(10):
                response = await client.get("/api/v1/health", headers={"X-Real-IP": ip})
                await asyncio.sleep(1)  # Normal behavior

        # Establish bad reputation
        for ip in bad_ips:
            for _ in range(50):
                await client.post(
                    "/api/v1/auth/login",
                    json={"username": "test@example.com", "password": "wrong"},
                    headers={"X-Real-IP": ip},
                )
                # No delay - aggressive behavior

        # Test reputation-based filtering
        reputation_tests = []

        # Good IPs should have more lenient limits
        for ip in good_ips:
            allowed = 0
            for _ in range(20):
                response = await client.post(
                    "/api/v1/auth/login",
                    json={"username": "test@example.com", "password": "test"},
                    headers={"X-Real-IP": ip},
                )
                if response.status_code != 429:
                    allowed += 1

            reputation_tests.append(("good", ip, allowed))

        # Bad IPs should have strict limits
        for ip in bad_ips:
            allowed = 0
            for _ in range(20):
                response = await client.post(
                    "/api/v1/auth/login",
                    json={"username": "test@example.com", "password": "test"},
                    headers={"X-Real-IP": ip},
                )
                if response.status_code != 429:
                    allowed += 1

            reputation_tests.append(("bad", ip, allowed))

        # Good IPs should be allowed more requests
        good_allowed = [t[2] for t in reputation_tests if t[0] == "good"]
        bad_allowed = [t[2] for t in reputation_tests if t[0] == "bad"]

        assert (
            np.mean(good_allowed) > np.mean(bad_allowed) * 2
        ), "Good reputation IPs should have more lenient limits"

    @pytest.mark.asyncio
    async def test_contextual_protection(self, client, redis_client):
        """Test context-aware protection mechanisms."""
        # Different contexts require different protection levels
        contexts = [
            {
                "name": "business_hours",
                "time": "14:00",
                "day": "Tuesday",
                "expected_traffic": "high",
            },
            {
                "name": "after_hours",
                "time": "03:00",
                "day": "Sunday",
                "expected_traffic": "low",
            },
            {
                "name": "maintenance_window",
                "time": "02:00",
                "day": "Saturday",
                "expected_traffic": "minimal",
            },
        ]

        context_results = []

        for context in contexts:
            # Simulate context
            headers = {
                "X-Context-Time": context["time"],
                "X-Context-Day": context["day"],
            }

            # Test protection sensitivity
            blocked_count = 0

            for i in range(50):
                response = await client.post(
                    "/api/v1/auth/login",
                    json={
                        "username": f"context_test_{i}@example.com",
                        "password": "test",
                    },
                    headers=headers,
                )

                if response.status_code == 429:
                    blocked_count += 1

            context_results.append(
                {
                    "context": context["name"],
                    "blocked": blocked_count,
                    "block_rate": blocked_count / 50,
                }
            )

        # Protection should be context-aware
        business_hours_rate = next(
            r["block_rate"] for r in context_results if r["context"] == "business_hours"
        )
        after_hours_rate = next(
            r["block_rate"] for r in context_results if r["context"] == "after_hours"
        )

        # After hours should have stricter protection
        assert after_hours_rate > business_hours_rate, "After hours should have stricter protection"


# Performance and stress testing utilities
class TestBruteForcePerformance:
    """Test performance under various brute force scenarios."""

    @pytest.mark.asyncio
    async def test_high_volume_attack_performance(self, client, redis_client):
        """Test system performance under high-volume attacks."""
        # Attack parameters
        concurrent_attackers = 100
        requests_per_attacker = 50

        async def attacker_task(attacker_id: int):
            """Simulate single attacker."""
            results = []

            for i in range(requests_per_attacker):
                start = time.perf_counter()

                response = await client.post(
                    "/api/v1/auth/login",
                    json={
                        "username": f"attacker_{attacker_id}@example.com",
                        "password": f"attempt_{i}",
                    },
                    headers={"X-Real-IP": f"attacker_{attacker_id}"},
                )

                results.append(
                    {
                        "response_time": time.perf_counter() - start,
                        "status": response.status_code,
                    }
                )

            return results

        # Launch concurrent attack
        start_time = time.time()

        tasks = []
        async with asyncio.TaskGroup() as tg:
            for i in range(concurrent_attackers):
                task = tg.create_task(attacker_task(i))
                tasks.append(task)

        # Collect results
        all_results = []
        for task in tasks:
            all_results.extend(task.result())

        total_duration = time.time() - start_time

        # Performance metrics
        response_times = [r["response_time"] for r in all_results]
        sum(1 for r in all_results if r["status"] != 429)

        # Performance assertions
        assert (
            np.percentile(response_times, 95) < 0.5
        ), "95th percentile response time should be < 500ms"
        assert (
            np.percentile(response_times, 99) < 1.0
        ), "99th percentile response time should be < 1s"
        assert total_duration < 30, f"Total test should complete in < 30s, took {total_duration}s"

    @pytest.mark.asyncio
    async def test_memory_stability_under_attack(self, client, redis_client):
        """Test memory stability during prolonged attacks."""
        import gc
        import os

        import psutil

        process = psutil.Process(os.getpid())

        # Force garbage collection and get baseline
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Sustained attack simulation
        attack_duration = 10  # seconds
        memory_samples = []

        start_time = time.time()
        request_count = 0

        while time.time() - start_time < attack_duration:
            # Send batch of requests
            batch_tasks = []

            async with asyncio.TaskGroup() as tg:
                for _ in range(10):
                    task = tg.create_task(
                        client.post(
                            "/api/v1/auth/login",
                            json={
                                "username": f"memory_test_{request_count}@example.com",
                                "password": "test",
                            },
                        )
                    )
                    batch_tasks.append(task)
                    request_count += 1

            # Sample memory
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_samples.append(current_memory)

            # Small delay between batches
            await asyncio.sleep(0.1)

        # Final memory check
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Memory analysis
        memory_increase = final_memory - baseline_memory
        max_memory = max(memory_samples)
        memory_variance = np.var(memory_samples)

        # Memory should remain stable
        assert memory_increase < 50, f"Memory increase should be < 50MB, got {memory_increase}MB"
        assert max_memory - baseline_memory < 100, "Peak memory should be < 100MB above baseline"
        assert memory_variance < 100, f"Memory variance should be low, got {memory_variance}"


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([__file__, "-v", "--tb=short", "--asyncio-mode=strict", "-k", "test_"])
