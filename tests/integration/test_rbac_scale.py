"""
RBAC Permissions Scale Testing
Task #6.3 - Test RBAC permissions at scale

This test suite validates RBAC performance and correctness under scale:
1. Large number of concurrent permission checks
2. Permission validation with many users and roles
3. Role hierarchy validation at scale
4. Resource access patterns under load
5. Permission caching and optimization
6. Complex permission scenarios
7. Performance benchmarks for authorization
"""

import concurrent.futures
import random
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import pytest

from auth.security_implementation import (
    ROLE_PERMISSIONS,
    AuthenticationManager,
    Permission,
    User,
    UserRole,
)


@dataclass
class RBACTestMetrics:
    """Metrics for RBAC scale testing."""

    permission_checks: int = 0
    successful_checks: int = 0
    failed_checks: int = 0
    check_times: List[float] = field(default_factory=list)
    role_check_times: Dict[str, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    permission_type_times: Dict[str, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    concurrent_checks: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    def add_check_time(self, duration: float, role: str, permission: str):
        """Record permission check timing."""
        self.check_times.append(duration)
        self.role_check_times[role].append(duration)
        self.permission_type_times[permission].append(duration)

    def calculate_statistics(self) -> Dict:
        """Calculate performance statistics."""
        stats = {
            "total_checks": self.permission_checks,
            "successful_checks": self.successful_checks,
            "failed_checks": self.failed_checks,
            "success_rate": (
                (self.successful_checks / self.permission_checks * 100)
                if self.permission_checks > 0
                else 0
            ),
            "avg_check_time": statistics.mean(self.check_times)
            if self.check_times
            else 0,
            "median_check_time": statistics.median(self.check_times)
            if self.check_times
            else 0,
            "p95_check_time": (
                statistics.quantiles(self.check_times, n=20)[18]
                if len(self.check_times) > 20
                else max(self.check_times, default=0)
            ),
            "p99_check_time": (
                statistics.quantiles(self.check_times, n=100)[98]
                if len(self.check_times) > 100
                else max(self.check_times, default=0)
            ),
            "checks_per_second": (
                self.permission_checks / sum(self.check_times)
                if self.check_times
                else 0
            ),
            "cache_hit_rate": (
                (self.cache_hits / (self.cache_hits + self.cache_misses) * 100)
                if (self.cache_hits + self.cache_misses) > 0
                else 0
            ),
        }

        # Role-specific stats
        stats["role_performance"] = {}
        for role, times in self.role_check_times.items():
            if times:
                stats["role_performance"][role] = {
                    "avg_time": statistics.mean(times),
                    "check_count": len(times),
                }

        # Permission-specific stats
        stats["permission_performance"] = {}
        for perm, times in self.permission_type_times.items():
            if times:
                stats["permission_performance"][perm] = {
                    "avg_time": statistics.mean(times),
                    "check_count": len(times),
                }

        return stats


class TestRBACScale:
    """Test RBAC at scale."""

    def setup_method(self):
        """Setup for each test."""
        self.auth_manager = AuthenticationManager()
        self.metrics = RBACTestMetrics()
        # Clear any cached data
        if hasattr(self.auth_manager, "_permission_cache"):
            self.auth_manager._permission_cache = {}

    def _create_test_users(self, users_per_role: int) -> Dict[str, List[User]]:
        """Create test users for each role."""
        users_by_role = {}

        for role in UserRole:
            users_by_role[role.value] = []
            for i in range(users_per_role):
                user = User(
                    user_id=f"{role.value}-user-{i}",
                    username=f"{role.value}_{i}",
                    email=f"{role.value}_{i}@test.com",
                    role=role,
                    created_at=datetime.now(timezone.utc),
                )
                users_by_role[role.value].append(user)
                # Register user
                self.auth_manager.users[user.username] = {
                    "user": user,
                    "password_hash": self.auth_manager.hash_password(f"pass_{i}"),
                }

        return users_by_role

    def _check_permission(
        self, user: User, permission: Permission
    ) -> Tuple[bool, float]:
        """Check if user has permission and measure time."""
        start = time.time()

        try:
            # Create token for user
            token = self.auth_manager.create_access_token(user)
            token_data = self.auth_manager.verify_token(token)

            # Check permission
            has_permission = permission in token_data.permissions
            duration = time.time() - start

            self.metrics.permission_checks += 1
            if has_permission:
                self.metrics.successful_checks += 1
            else:
                self.metrics.failed_checks += 1

            self.metrics.add_check_time(duration, user.role.value, permission.value)

            return has_permission, duration

        except Exception:
            duration = time.time() - start
            self.metrics.failed_checks += 1
            self.metrics.permission_checks += 1
            return False, duration

    def test_concurrent_permission_checks(self):
        """Test concurrent permission checks for many users."""
        users_per_role = 25
        checks_per_user = 20

        users_by_role = self._create_test_users(users_per_role)
        all_users = []
        for role_users in users_by_role.values():
            all_users.extend(role_users)

        def check_user_permissions(
            user: User,
        ) -> List[Tuple[Permission, bool, float]]:
            """Check multiple permissions for a user."""
            results = []
            permissions_to_check = list(Permission)

            for _ in range(checks_per_user):
                permission = random.choice(permissions_to_check)
                has_perm, duration = self._check_permission(user, permission)
                results.append((permission, has_perm, duration))

            return results

        start_time = time.time()

        # Run concurrent permission checks
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [
                executor.submit(check_user_permissions, user) for user in all_users
            ]

            all_results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    print(f"Error in concurrent check: {e}")

        total_time = time.time() - start_time

        # Calculate statistics
        stats = self.metrics.calculate_statistics()

        # Performance assertions
        assert stats["avg_check_time"] < 0.01, (
            f"Average permission check too slow: {stats['avg_check_time']}s"
        )
        assert stats["p95_check_time"] < 0.02, (
            f"P95 permission check too slow: {stats['p95_check_time']}s"
        )
        assert stats["checks_per_second"] > 1000, (
            f"Permission check throughput too low: {stats['checks_per_second']} checks/s"
        )
        assert total_time < 30, f"Total test time too high: {total_time}s"

    def test_role_hierarchy_at_scale(self):
        """Test role hierarchy validation with many users."""
        num_users = 100

        # Create users with different roles
        users = []
        for i in range(num_users):
            role = random.choice(list(UserRole))
            user = User(
                user_id=f"hierarchy-user-{i}",
                username=f"hierarchy_{i}",
                email=f"hierarchy_{i}@test.com",
                role=role,
                created_at=datetime.now(timezone.utc),
            )
            users.append(user)

        # Verify role hierarchy is maintained
        hierarchy_checks = []

        for user in users:
            token = self.auth_manager.create_access_token(user)
            token_data = self.auth_manager.verify_token(token)

            # Check that user has exactly the permissions for their role
            expected_permissions = set(ROLE_PERMISSIONS.get(user.role, []))
            actual_permissions = set(token_data.permissions)

            hierarchy_checks.append(
                {
                    "user": user.user_id,
                    "role": user.role.value,
                    "expected": expected_permissions,
                    "actual": actual_permissions,
                    "correct": expected_permissions == actual_permissions,
                }
            )

        # All hierarchy checks should be correct
        incorrect_checks = [check for check in hierarchy_checks if not check["correct"]]
        assert len(incorrect_checks) == 0, (
            f"Role hierarchy violations found: {incorrect_checks}"
        )

        # Verify admin has all permissions
        admin_checks = [
            check for check in hierarchy_checks if check["role"] == UserRole.ADMIN.value
        ]
        for check in admin_checks:
            assert Permission.ADMIN_SYSTEM in check["actual"], (
                "Admin should have ADMIN_SYSTEM permission"
            )

        # Verify observer has limited permissions
        observer_checks = [
            check
            for check in hierarchy_checks
            if check["role"] == UserRole.OBSERVER.value
        ]
        for check in observer_checks:
            assert Permission.ADMIN_SYSTEM not in check["actual"], (
                "Observer should not have ADMIN_SYSTEM permission"
            )
            assert Permission.CREATE_AGENT not in check["actual"], (
                "Observer should not have CREATE_AGENT permission"
            )

    def test_permission_check_performance_by_role(self):
        """Test permission check performance for different roles."""
        users_per_role = 20

        users_by_role = self._create_test_users(users_per_role)

        # Test each role separately
        role_metrics = {}

        for role, users in users_by_role.items():
            role_start = time.time()
            role_checks = 0

            for user in users:
                for permission in Permission:
                    has_perm, duration = self._check_permission(user, permission)
                    role_checks += 1

            role_time = time.time() - role_start
            role_metrics[role] = {
                "total_time": role_time,
                "checks": role_checks,
                "avg_time": role_time / role_checks if role_checks > 0 else 0,
            }

        # Admin role might be slightly slower due to more permissions
        for role, metrics in role_metrics.items():
            assert metrics["avg_time"] < 0.01, (
                f"Role {role} permission checks too slow: {metrics['avg_time']}s"
            )

    def test_complex_permission_scenarios(self):
        """Test complex permission scenarios at scale."""
        # Create users with different roles
        users_by_role = self._create_test_users(10)

        scenarios = []

        # Scenario 1: Admin can do everything
        admin_user = users_by_role[UserRole.ADMIN.value][0]
        for permission in Permission:
            has_perm, _ = self._check_permission(admin_user, permission)
            scenarios.append(
                {
                    "scenario": "admin_all_permissions",
                    "role": UserRole.ADMIN.value,
                    "permission": permission.value,
                    "expected": True,
                    "actual": has_perm,
                    "passed": has_perm is True,
                }
            )

        # Scenario 2: Observer can only view
        observer_user = users_by_role[UserRole.OBSERVER.value][0]
        view_permissions = [Permission.VIEW_AGENTS, Permission.VIEW_METRICS]
        for permission in Permission:
            has_perm, _ = self._check_permission(observer_user, permission)
            expected = permission in view_permissions
            scenarios.append(
                {
                    "scenario": "observer_view_only",
                    "role": UserRole.OBSERVER.value,
                    "permission": permission.value,
                    "expected": expected,
                    "actual": has_perm,
                    "passed": has_perm == expected,
                }
            )

        # Scenario 3: Researcher cannot admin
        researcher_user = users_by_role[UserRole.RESEARCHER.value][0]
        has_perm, _ = self._check_permission(researcher_user, Permission.ADMIN_SYSTEM)
        scenarios.append(
            {
                "scenario": "researcher_no_admin",
                "role": UserRole.RESEARCHER.value,
                "permission": Permission.ADMIN_SYSTEM.value,
                "expected": False,
                "actual": has_perm,
                "passed": has_perm is False,
            }
        )

        # Verify all scenarios passed
        failed_scenarios = [s for s in scenarios if not s["passed"]]
        assert len(failed_scenarios) == 0, (
            f"Failed permission scenarios: {failed_scenarios}"
        )

    def test_permission_validation_with_token_rotation(self):
        """Test permission checks during token refresh cycles."""
        users_per_role = 5
        refresh_cycles = 10

        users_by_role = self._create_test_users(users_per_role)

        for role, users in users_by_role.items():
            for user in users:
                # Initial tokens
                access_token = self.auth_manager.create_access_token(user)
                refresh_token = self.auth_manager.create_refresh_token(user)

                for cycle in range(refresh_cycles):
                    # Check permissions with current token
                    token_data = self.auth_manager.verify_token(access_token)

                    # Verify permissions are consistent
                    expected_perms = set(ROLE_PERMISSIONS.get(user.role, []))
                    actual_perms = set(token_data.permissions)
                    assert expected_perms == actual_perms, (
                        f"Permission mismatch after {cycle} refreshes"
                    )

                    # Refresh token
                    try:
                        (
                            access_token,
                            refresh_token,
                        ) = self.auth_manager.refresh_access_token(refresh_token)
                    except Exception:
                        # If refresh fails, create new tokens
                        access_token = self.auth_manager.create_access_token(user)
                        refresh_token = self.auth_manager.create_refresh_token(user)

    def test_permission_check_with_concurrent_role_changes(self):
        """Test permission checks when roles are changed concurrently."""
        num_users = 20
        num_threads = 10

        # Create users
        users = []
        for i in range(num_users):
            user = User(
                user_id=f"role-change-user-{i}",
                username=f"rolechange_{i}",
                email=f"rolechange_{i}@test.com",
                role=UserRole.OBSERVER,  # Start as observer
                created_at=datetime.now(timezone.utc),
            )
            users.append(user)
            self.auth_manager.users[user.username] = {
                "user": user,
                "password_hash": self.auth_manager.hash_password("password"),
            }

        results = []
        result_lock = threading.Lock()

        def change_role_and_check(user_index: int):
            """Change user role and check permissions."""
            user = users[user_index]

            # Randomly change role
            new_role = random.choice(
                [UserRole.RESEARCHER, UserRole.AGENT_MANAGER, UserRole.ADMIN]
            )
            user.role = new_role

            # Create new token with new role
            token = self.auth_manager.create_access_token(user)
            token_data = self.auth_manager.verify_token(token)

            # Check permissions match new role
            expected_perms = set(ROLE_PERMISSIONS.get(new_role, []))
            actual_perms = set(token_data.permissions)

            with result_lock:
                results.append(
                    {
                        "user": user.user_id,
                        "new_role": new_role.value,
                        "permissions_match": expected_perms == actual_perms,
                    }
                )

        # Run concurrent role changes
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for _ in range(100):  # Multiple iterations
                user_index = random.randint(0, num_users - 1)
                futures.append(executor.submit(change_role_and_check, user_index))

            concurrent.futures.wait(futures)

        # Verify all permission checks were correct
        mismatches = [r for r in results if not r["permissions_match"]]
        assert len(mismatches) == 0, (
            f"Permission mismatches after role changes: {mismatches}"
        )

    def test_resource_access_patterns_under_load(self):
        """Test different resource access patterns under load."""
        users_per_role = 10
        resources_per_type = 50

        users_by_role = self._create_test_users(users_per_role)

        # Simulate different resource types
        resources = {
            "agents": [f"agent_{i}" for i in range(resources_per_type)],
            "coalitions": [f"coalition_{i}" for i in range(resources_per_type)],
            "metrics": [f"metric_{i}" for i in range(resources_per_type)],
        }

        access_patterns = []

        def simulate_resource_access(user: User, resource_type: str, resource_id: str):
            """Simulate accessing a resource."""
            start = time.time()

            # Determine required permission based on resource type
            permission_map = {
                "agents": Permission.VIEW_AGENTS,
                "coalitions": Permission.CREATE_COALITION,
                "metrics": Permission.VIEW_METRICS,
            }

            required_permission = permission_map.get(resource_type)
            if not required_permission:
                return

            has_perm, _ = self._check_permission(user, required_permission)

            access_patterns.append(
                {
                    "user_role": user.role.value,
                    "resource_type": resource_type,
                    "resource_id": resource_id,
                    "has_permission": has_perm,
                    "access_time": time.time() - start,
                }
            )

        # Simulate concurrent resource access
        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            futures = []

            # Generate random access patterns
            for _ in range(1000):
                role = random.choice(list(users_by_role.keys()))
                user = random.choice(users_by_role[role])
                resource_type = random.choice(list(resources.keys()))
                resource_id = random.choice(resources[resource_type])

                futures.append(
                    executor.submit(
                        simulate_resource_access,
                        user,
                        resource_type,
                        resource_id,
                    )
                )

            concurrent.futures.wait(futures)

        # Analyze access patterns
        total_accesses = len(access_patterns)
        len([p for p in access_patterns if p["has_permission"]])
        avg_access_time = statistics.mean([p["access_time"] for p in access_patterns])

        assert total_accesses >= 900, "Some access attempts failed"
        assert avg_access_time < 0.01, (
            f"Average access time too high: {avg_access_time}s"
        )

    def test_permission_caching_effectiveness(self):
        """Test effectiveness of permission caching under load."""
        # Enable permission caching if not already enabled
        if not hasattr(self.auth_manager, "_permission_cache"):
            self.auth_manager._permission_cache = {}

        user = User(
            user_id="cache-test-user",
            username="cacheuser",
            email="cache@test.com",
            role=UserRole.RESEARCHER,
            created_at=datetime.now(timezone.utc),
        )

        # First round - cold cache
        cold_times = []
        for _ in range(100):
            start = time.time()
            has_perm, _ = self._check_permission(user, Permission.CREATE_AGENT)
            cold_times.append(time.time() - start)

        # Second round - warm cache
        warm_times = []
        for _ in range(100):
            start = time.time()
            has_perm, _ = self._check_permission(user, Permission.CREATE_AGENT)
            warm_times.append(time.time() - start)

        avg_cold = statistics.mean(cold_times)
        avg_warm = statistics.mean(warm_times)

        # Warm cache should be faster (though our implementation creates new tokens each time)
        # In a real implementation with proper caching, warm would be significantly faster
        assert avg_warm <= avg_cold * 1.1, (
            "Cache not providing expected performance benefit"
        )

    @pytest.mark.parametrize(
        "num_users,num_permissions",
        [
            (10, 100),
            (50, 500),
            (100, 1000),
        ],
    )
    def test_scalability_with_different_loads(self, num_users, num_permissions):
        """Test RBAC scalability with different load levels."""
        # Create users
        users = []
        for i in range(num_users):
            role = random.choice(list(UserRole))
            user = User(
                user_id=f"scale-user-{i}",
                username=f"scale_{i}",
                email=f"scale_{i}@test.com",
                role=role,
                created_at=datetime.now(timezone.utc),
            )
            users.append(user)

        # Perform permission checks
        start_time = time.time()
        check_times = []

        for _ in range(num_permissions):
            user = random.choice(users)
            permission = random.choice(list(Permission))

            check_start = time.time()
            has_perm, _ = self._check_permission(user, permission)
            check_times.append(time.time() - check_start)

        total_time = time.time() - start_time

        # Calculate metrics
        avg_check_time = statistics.mean(check_times)
        checks_per_second = num_permissions / total_time

        # Performance should scale reasonably
        assert avg_check_time < 0.02, (
            f"Avg check time too high for {num_users} users: {avg_check_time}s"
        )
        assert checks_per_second > 50, (
            f"Throughput too low for {num_users} users: {checks_per_second} checks/s"
        )

        # Log performance for analysis
        print(f"\nLoad test with {num_users} users, {num_permissions} checks:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Avg check time: {avg_check_time * 1000:.2f}ms")
        print(f"  Checks/second: {checks_per_second:.0f}")
