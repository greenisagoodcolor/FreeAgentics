"""
RBAC Permissions at Scale Test Suite

Tests role-based access control functionality and performance with large permission sets,
complex role hierarchies, and large user bases.
"""

import itertools
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, List, Set

import pytest

from auth.security_implementation import (
    AuthenticationManager,
    Permission,
    ROLE_PERMISSIONS,
    User,
    UserRole,
    require_permission,
    require_role,
)


class TestRBACPermissionsAtScale:
    """Test RBAC system with large permission sets and complex scenarios."""

    @pytest.fixture
    def auth_manager(self):
        """Create fresh AuthenticationManager instance."""
        return AuthenticationManager()

    @pytest.fixture
    def test_users_by_role(self, auth_manager):
        """Create test users for each role."""
        users_by_role = {}
        
        for role in UserRole:
            users_by_role[role] = []
            for i in range(50):  # 50 users per role
                user = User(
                    user_id=f"{role.value}_user_{i}",
                    username=f"{role.value}_user_{i}",
                    email=f"{role.value}_user_{i}@example.com",
                    role=role,
                    created_at=datetime.now(timezone.utc),
                    is_active=True,
                )
                # Register user in auth manager
                auth_manager.users[user.username] = {
                    "user": user,
                    "password_hash": auth_manager.hash_password("password123"),
                }
                users_by_role[role].append(user)
        
        return users_by_role

    def test_role_permission_mapping_correctness(self):
        """Test that role-permission mapping is correct and complete."""
        # Verify all roles have permissions defined
        for role in UserRole:
            assert role in ROLE_PERMISSIONS, f"Role {role} missing from ROLE_PERMISSIONS"
            assert len(ROLE_PERMISSIONS[role]) > 0, f"Role {role} has no permissions"

        # Verify all permissions are valid
        for role, permissions in ROLE_PERMISSIONS.items():
            for permission in permissions:
                assert isinstance(permission, Permission), f"Invalid permission {permission} for role {role}"

        # Verify permission hierarchy makes sense
        admin_permissions = set(ROLE_PERMISSIONS[UserRole.ADMIN])
        researcher_permissions = set(ROLE_PERMISSIONS[UserRole.RESEARCHER])
        agent_manager_permissions = set(ROLE_PERMISSIONS[UserRole.AGENT_MANAGER])
        observer_permissions = set(ROLE_PERMISSIONS[UserRole.OBSERVER])

        # Admin should have all permissions
        all_permissions = set(Permission)
        assert admin_permissions == all_permissions, "Admin should have all permissions"

        # Observer should have subset of agent manager permissions
        assert observer_permissions.issubset(agent_manager_permissions), "Observer should be subset of agent manager"

        # Agent manager should have subset of researcher permissions
        assert agent_manager_permissions.issubset(researcher_permissions), "Agent manager should be subset of researcher"

        # Researcher should have subset of admin permissions
        assert researcher_permissions.issubset(admin_permissions), "Researcher should be subset of admin"

    def test_token_permission_validation_at_scale(self, auth_manager, test_users_by_role):
        """Test token permission validation with large number of users."""
        results = []
        
        # Test all users across all roles
        all_users = []
        for role_users in test_users_by_role.values():
            all_users.extend(role_users)
        
        def validate_user_permissions(user):
            try:
                start_time = time.time()
                
                # Create access token
                token = auth_manager.create_access_token(user)
                
                # Verify token and check permissions
                token_data = auth_manager.verify_token(token)
                
                # Verify permissions match role
                expected_permissions = set(ROLE_PERMISSIONS[user.role])
                actual_permissions = set(token_data.permissions)
                
                end_time = time.time()
                
                return {
                    "user_id": user.user_id,
                    "role": user.role,
                    "permissions_correct": expected_permissions == actual_permissions,
                    "permission_count": len(actual_permissions),
                    "duration": end_time - start_time,
                    "success": True,
                }
            except Exception as e:
                return {
                    "user_id": user.user_id,
                    "role": user.role,
                    "success": False,
                    "error": str(e),
                }
        
        # Test with concurrent users
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(validate_user_permissions, user) for user in all_users]
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
        
        # Verify all results
        successful_results = [r for r in results if r["success"]]
        assert len(successful_results) == len(all_users), f"Only {len(successful_results)} out of {len(all_users)} succeeded"
        
        # Verify all permissions are correct
        assert all(r["permissions_correct"] for r in successful_results), "Some users have incorrect permissions"
        
        # Verify performance
        durations = [r["duration"] for r in successful_results]
        avg_duration = sum(durations) / len(durations)
        assert avg_duration < 0.1, f"Average permission validation too slow: {avg_duration:.3f}s"

    def test_permission_inheritance_scenarios(self, auth_manager, test_users_by_role):
        """Test complex permission inheritance scenarios."""
        
        # Test scenarios where higher roles should have all permissions of lower roles
        inheritance_tests = [
            (UserRole.ADMIN, UserRole.RESEARCHER),
            (UserRole.ADMIN, UserRole.AGENT_MANAGER),
            (UserRole.ADMIN, UserRole.OBSERVER),
            (UserRole.RESEARCHER, UserRole.AGENT_MANAGER),
            (UserRole.RESEARCHER, UserRole.OBSERVER),
            (UserRole.AGENT_MANAGER, UserRole.OBSERVER),
        ]
        
        for higher_role, lower_role in inheritance_tests:
            higher_permissions = set(ROLE_PERMISSIONS[higher_role])
            lower_permissions = set(ROLE_PERMISSIONS[lower_role])
            
            # Higher role should have all permissions of lower role
            assert lower_permissions.issubset(higher_permissions), \
                f"{higher_role} should have all permissions of {lower_role}"
            
            # Test with actual users
            higher_user = test_users_by_role[higher_role][0]
            lower_user = test_users_by_role[lower_role][0]
            
            higher_token = auth_manager.create_access_token(higher_user)
            lower_token = auth_manager.create_access_token(lower_user)
            
            higher_token_data = auth_manager.verify_token(higher_token)
            lower_token_data = auth_manager.verify_token(lower_token)
            
            # Verify inheritance in actual tokens
            assert set(lower_token_data.permissions).issubset(set(higher_token_data.permissions)), \
                f"{higher_role} token should have all permissions of {lower_role} token"

    def test_permission_access_control_enforcement(self, auth_manager, test_users_by_role):
        """Test that permission access control is properly enforced."""
        
        # Test each permission with users that should and shouldn't have it
        for permission in Permission:
            # Find roles that should have this permission
            roles_with_permission = [role for role, perms in ROLE_PERMISSIONS.items() if permission in perms]
            roles_without_permission = [role for role in UserRole if role not in roles_with_permission]
            
            # Test users with permission
            for role in roles_with_permission:
                user = test_users_by_role[role][0]
                token = auth_manager.create_access_token(user)
                token_data = auth_manager.verify_token(token)
                
                assert permission in token_data.permissions, \
                    f"User with role {role} should have permission {permission}"
            
            # Test users without permission
            for role in roles_without_permission:
                user = test_users_by_role[role][0]
                token = auth_manager.create_access_token(user)
                token_data = auth_manager.verify_token(token)
                
                assert permission not in token_data.permissions, \
                    f"User with role {role} should not have permission {permission}"

    def test_concurrent_permission_checks(self, auth_manager, test_users_by_role):
        """Test concurrent permission checks don't interfere with each other."""
        
        # Create mixed set of users with different roles
        mixed_users = []
        for role, users in test_users_by_role.items():
            mixed_users.extend(users[:10])  # 10 users per role
        
        # Shuffle for randomness
        random.shuffle(mixed_users)
        
        results = []
        
        def check_permissions(user):
            try:
                start_time = time.time()
                
                # Create token
                token = auth_manager.create_access_token(user)
                token_data = auth_manager.verify_token(token)
                
                # Check multiple permissions
                expected_permissions = set(ROLE_PERMISSIONS[user.role])
                actual_permissions = set(token_data.permissions)
                
                # Verify role is correct
                assert token_data.role == user.role
                
                # Verify permissions are correct
                assert expected_permissions == actual_permissions
                
                end_time = time.time()
                
                return {
                    "user_id": user.user_id,
                    "role": user.role,
                    "duration": end_time - start_time,
                    "success": True,
                }
            except Exception as e:
                return {
                    "user_id": user.user_id,
                    "role": user.role,
                    "success": False,
                    "error": str(e),
                }
        
        # Run concurrent permission checks
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(check_permissions, user) for user in mixed_users]
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
        
        # Verify all succeeded
        successful_results = [r for r in results if r["success"]]
        assert len(successful_results) == len(mixed_users), "Some concurrent permission checks failed"
        
        # Verify performance
        durations = [r["duration"] for r in successful_results]
        avg_duration = sum(durations) / len(durations)
        assert avg_duration < 0.1, f"Concurrent permission checks too slow: {avg_duration:.3f}s"

    def test_permission_scalability_with_large_user_base(self, auth_manager):
        """Test permission system scalability with a large user base."""
        
        # Create a large number of users
        large_user_count = 500
        users = []
        
        for i in range(large_user_count):
            role = random.choice(list(UserRole))
            user = User(
                user_id=f"scale_user_{i}",
                username=f"scale_user_{i}",
                email=f"scale_user_{i}@example.com",
                role=role,
                created_at=datetime.now(timezone.utc),
                is_active=True,
            )
            auth_manager.users[user.username] = {
                "user": user,
                "password_hash": auth_manager.hash_password("password123"),
            }
            users.append(user)
        
        # Test permission validation at scale
        results = []
        
        def validate_user_at_scale(user):
            start_time = time.time()
            
            # Create token
            token = auth_manager.create_access_token(user)
            
            # Verify token and permissions
            token_data = auth_manager.verify_token(token)
            
            # Verify permissions match role
            expected_permissions = set(ROLE_PERMISSIONS[user.role])
            actual_permissions = set(token_data.permissions)
            
            end_time = time.time()
            
            return {
                "user_id": user.user_id,
                "role": user.role,
                "permissions_correct": expected_permissions == actual_permissions,
                "duration": end_time - start_time,
            }
        
        # Test with high concurrency
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(validate_user_at_scale, user) for user in users]
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
        
        # Verify all permissions are correct
        assert all(r["permissions_correct"] for r in results), "Some users have incorrect permissions at scale"
        
        # Verify performance doesn't degrade significantly
        durations = [r["duration"] for r in results]
        avg_duration = sum(durations) / len(durations)
        max_duration = max(durations)
        
        assert avg_duration < 0.2, f"Average permission validation too slow at scale: {avg_duration:.3f}s"
        assert max_duration < 1.0, f"Max permission validation too slow at scale: {max_duration:.3f}s"

    def test_permission_edge_cases(self, auth_manager, test_users_by_role):
        """Test edge cases in permission handling."""
        
        # Test user with no permissions (if such a role existed)
        # In our current system, all roles have some permissions
        
        # Test permission checking with invalid permissions
        user = test_users_by_role[UserRole.OBSERVER][0]
        token = auth_manager.create_access_token(user)
        token_data = auth_manager.verify_token(token)
        
        # Test non-existent permission
        assert "non_existent_permission" not in [p.value for p in token_data.permissions]
        
        # Test permission case sensitivity
        for permission in token_data.permissions:
            assert permission.value.lower() == permission.value or permission.value.upper() == permission.value, \
                f"Permission {permission} should be consistent case"

    def test_role_based_access_matrix(self, auth_manager, test_users_by_role):
        """Test complete role-based access matrix."""
        
        # Create access matrix
        access_matrix = {}
        
        for role in UserRole:
            access_matrix[role] = {}
            user = test_users_by_role[role][0]
            token = auth_manager.create_access_token(user)
            token_data = auth_manager.verify_token(token)
            
            for permission in Permission:
                access_matrix[role][permission] = permission in token_data.permissions
        
        # Verify access matrix properties
        
        # Admin should have access to everything
        for permission in Permission:
            assert access_matrix[UserRole.ADMIN][permission], \
                f"Admin should have access to {permission}"
        
        # Observer should have minimal access
        observer_permissions = [p for p in Permission if access_matrix[UserRole.OBSERVER][p]]
        assert len(observer_permissions) <= 2, "Observer should have minimal permissions"
        
        # Verify hierarchical access
        role_hierarchy = [UserRole.OBSERVER, UserRole.AGENT_MANAGER, UserRole.RESEARCHER, UserRole.ADMIN]
        
        for i in range(len(role_hierarchy) - 1):
            lower_role = role_hierarchy[i]
            higher_role = role_hierarchy[i + 1]
            
            for permission in Permission:
                if access_matrix[lower_role][permission]:
                    assert access_matrix[higher_role][permission], \
                        f"{higher_role} should have {permission} if {lower_role} has it"

    def test_permission_performance_under_load(self, auth_manager, test_users_by_role):
        """Test permission system performance under sustained load."""
        
        # Create continuous load
        load_duration = 5  # 5 seconds
        users_to_test = []
        
        for role, role_users in test_users_by_role.items():
            users_to_test.extend(role_users[:5])  # 5 users per role
        
        start_time = time.time()
        results = []
        
        def continuous_permission_check():
            test_results = []
            while time.time() - start_time < load_duration:
                user = random.choice(users_to_test)
                
                check_start = time.time()
                token = auth_manager.create_access_token(user)
                token_data = auth_manager.verify_token(token)
                
                # Verify permissions
                expected_permissions = set(ROLE_PERMISSIONS[user.role])
                actual_permissions = set(token_data.permissions)
                
                check_end = time.time()
                
                test_results.append({
                    "user_id": user.user_id,
                    "role": user.role,
                    "permissions_correct": expected_permissions == actual_permissions,
                    "duration": check_end - check_start,
                })
            
            return test_results
        
        # Run continuous load with multiple threads
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(continuous_permission_check) for _ in range(10)]
            
            for future in as_completed(futures):
                thread_results = future.result()
                results.extend(thread_results)
        
        # Verify all permissions were correct under load
        assert all(r["permissions_correct"] for r in results), "Some permissions incorrect under load"
        
        # Verify performance remained consistent
        durations = [r["duration"] for r in results]
        avg_duration = sum(durations) / len(durations)
        
        assert avg_duration < 0.1, f"Permission checks too slow under load: {avg_duration:.3f}s"
        
        # Verify we processed a reasonable number of requests
        assert len(results) > 100, f"Should have processed many requests under load, got {len(results)}"

    def test_multi_role_permission_scenarios(self, auth_manager, test_users_by_role):
        """Test scenarios involving multiple roles and complex permission interactions."""
        
        # Test permission combinations
        permission_combinations = list(itertools.combinations(Permission, 2))
        
        for perm1, perm2 in permission_combinations[:10]:  # Test first 10 combinations
            # Find roles that have both permissions
            roles_with_both = []
            roles_with_one = []
            roles_with_none = []
            
            for role in UserRole:
                role_permissions = set(ROLE_PERMISSIONS[role])
                if perm1 in role_permissions and perm2 in role_permissions:
                    roles_with_both.append(role)
                elif perm1 in role_permissions or perm2 in role_permissions:
                    roles_with_one.append(role)
                else:
                    roles_with_none.append(role)
            
            # Test users with both permissions
            for role in roles_with_both:
                user = test_users_by_role[role][0]
                token = auth_manager.create_access_token(user)
                token_data = auth_manager.verify_token(token)
                
                assert perm1 in token_data.permissions, f"{role} should have {perm1}"
                assert perm2 in token_data.permissions, f"{role} should have {perm2}"
            
            # Test users with only one permission
            for role in roles_with_one:
                user = test_users_by_role[role][0]
                token = auth_manager.create_access_token(user)
                token_data = auth_manager.verify_token(token)
                
                has_perm1 = perm1 in token_data.permissions
                has_perm2 = perm2 in token_data.permissions
                
                # Should have exactly one of the permissions
                assert has_perm1 or has_perm2, f"{role} should have at least one of {perm1} or {perm2}"
                assert not (has_perm1 and has_perm2), f"{role} should not have both {perm1} and {perm2}"

    def test_permission_consistency_across_tokens(self, auth_manager, test_users_by_role):
        """Test that permissions are consistent across multiple token generations."""
        
        # Test each role
        for role, users in test_users_by_role.items():
            user = users[0]
            
            # Generate multiple tokens for the same user
            tokens = []
            for _ in range(10):
                token = auth_manager.create_access_token(user)
                tokens.append(token)
            
            # Verify all tokens have the same permissions
            permission_sets = []
            for token in tokens:
                token_data = auth_manager.verify_token(token)
                permission_sets.append(set(token_data.permissions))
            
            # All permission sets should be identical
            first_permission_set = permission_sets[0]
            for i, permission_set in enumerate(permission_sets[1:], 1):
                assert permission_set == first_permission_set, \
                    f"Token {i} has different permissions than token 0 for role {role}"
            
            # Verify permissions match role definition
            expected_permissions = set(ROLE_PERMISSIONS[role])
            assert first_permission_set == expected_permissions, \
                f"Token permissions don't match role definition for {role}"