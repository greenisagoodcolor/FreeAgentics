"""Security-critical tests for RBAC permissions following TDD principles.

This test suite covers Role-Based Access Control:
- Permission checks
- Role hierarchy
- Resource access control
- Permission inheritance
- Deny-by-default principle
"""

from enum import Enum
from typing import Dict, List, Optional, Set

import pytest


class Permission(Enum):
    """System permissions."""
    # Agent permissions
    AGENT_CREATE = "agent:create"
    AGENT_READ = "agent:read"
    AGENT_UPDATE = "agent:update"
    AGENT_DELETE = "agent:delete"
    
    # Coalition permissions
    COALITION_CREATE = "coalition:create"
    COALITION_READ = "coalition:read"
    COALITION_MANAGE = "coalition:manage"
    
    # System permissions
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_MONITOR = "system:monitor"
    
    # User management
    USER_MANAGE = "user:manage"
    USER_READ = "user:read"


class Role:
    """Role definition with permissions."""
    
    def __init__(self, name: str, permissions: Set[Permission], parent: Optional['Role'] = None):
        self.name = name
        self.permissions = permissions
        self.parent = parent
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if role has a specific permission."""
        # Check direct permissions
        if permission in self.permissions:
            return True
        
        # Check inherited permissions
        if self.parent:
            return self.parent.has_permission(permission)
        
        return False
    
    def get_all_permissions(self) -> Set[Permission]:
        """Get all permissions including inherited ones."""
        all_perms = self.permissions.copy()
        
        if self.parent:
            all_perms.update(self.parent.get_all_permissions())
        
        return all_perms


class RBACManager:
    """Role-Based Access Control manager."""
    
    def __init__(self):
        self.roles: Dict[str, Role] = {}
        self.user_roles: Dict[str, List[str]] = {}
        self.resource_permissions: Dict[str, Dict[str, Set[Permission]]] = {}
        self._setup_default_roles()
    
    def _setup_default_roles(self):
        """Set up default system roles."""
        # Basic roles
        guest = Role("guest", {Permission.AGENT_READ})
        user = Role("user", {
            Permission.AGENT_CREATE,
            Permission.COALITION_READ,
        }, parent=guest)
        
        power_user = Role("power_user", {
            Permission.AGENT_UPDATE,
            Permission.COALITION_CREATE,
        }, parent=user)
        
        admin = Role("admin", {
            Permission.AGENT_DELETE,
            Permission.COALITION_MANAGE,
            Permission.USER_MANAGE,
            Permission.SYSTEM_MONITOR,
        }, parent=power_user)
        
        super_admin = Role("super_admin", {
            Permission.SYSTEM_ADMIN,
        }, parent=admin)
        
        # Register roles
        self.roles = {
            "guest": guest,
            "user": user,
            "power_user": power_user,
            "admin": admin,
            "super_admin": super_admin,
        }
    
    def assign_role(self, user_id: str, role_name: str):
        """Assign a role to a user."""
        if role_name not in self.roles:
            raise ValueError(f"Unknown role: {role_name}")
        
        if user_id not in self.user_roles:
            self.user_roles[user_id] = []
        
        if role_name not in self.user_roles[user_id]:
            self.user_roles[user_id].append(role_name)
    
    def revoke_role(self, user_id: str, role_name: str):
        """Revoke a role from a user."""
        if user_id in self.user_roles:
            self.user_roles[user_id] = [
                r for r in self.user_roles[user_id] if r != role_name
            ]
            if not self.user_roles[user_id]:
                del self.user_roles[user_id]
    
    def check_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has a permission."""
        # Deny by default
        if user_id not in self.user_roles:
            return False
        
        # Check all user's roles
        for role_name in self.user_roles[user_id]:
            role = self.roles.get(role_name)
            if role and role.has_permission(permission):
                return True
        
        return False
    
    def check_resource_permission(
        self, 
        user_id: str, 
        resource_type: str, 
        resource_id: str, 
        permission: Permission
    ) -> bool:
        """Check if user has permission on a specific resource."""
        # Check resource-specific permissions first
        resource_key = f"{resource_type}:{resource_id}"
        if resource_key in self.resource_permissions:
            resource_perms = self.resource_permissions[resource_key]
            if user_id in resource_perms and permission in resource_perms[user_id]:
                return True
        
        # Fall back to general permission
        return self.check_permission(user_id, permission)
    
    def grant_resource_permission(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        permission: Permission
    ):
        """Grant permission on a specific resource."""
        resource_key = f"{resource_type}:{resource_id}"
        
        if resource_key not in self.resource_permissions:
            self.resource_permissions[resource_key] = {}
        
        if user_id not in self.resource_permissions[resource_key]:
            self.resource_permissions[resource_key][user_id] = set()
        
        self.resource_permissions[resource_key][user_id].add(permission)
    
    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for a user."""
        all_perms = set()
        
        for role_name in self.user_roles.get(user_id, []):
            role = self.roles.get(role_name)
            if role:
                all_perms.update(role.get_all_permissions())
        
        return all_perms


class TestRoleHierarchy:
    """Test role hierarchy and inheritance."""
    
    def test_role_permission_inheritance(self):
        """Test that roles inherit permissions from parents."""
        # Arrange
        parent = Role("parent", {Permission.AGENT_READ})
        child = Role("child", {Permission.AGENT_CREATE}, parent=parent)
        
        # Act & Assert
        assert child.has_permission(Permission.AGENT_CREATE) is True
        assert child.has_permission(Permission.AGENT_READ) is True  # Inherited
        assert child.has_permission(Permission.AGENT_DELETE) is False
    
    def test_get_all_permissions_includes_inherited(self):
        """Test getting all permissions includes inherited ones."""
        # Arrange
        grandparent = Role("grandparent", {Permission.AGENT_READ})
        parent = Role("parent", {Permission.AGENT_CREATE}, parent=grandparent)
        child = Role("child", {Permission.AGENT_UPDATE}, parent=parent)
        
        # Act
        all_perms = child.get_all_permissions()
        
        # Assert
        assert Permission.AGENT_READ in all_perms  # From grandparent
        assert Permission.AGENT_CREATE in all_perms  # From parent
        assert Permission.AGENT_UPDATE in all_perms  # Direct


class TestRBACManager:
    """Test RBAC manager functionality."""
    
    @pytest.fixture
    def rbac(self):
        """Create RBAC manager instance."""
        return RBACManager()
    
    def test_assign_role_to_user(self, rbac):
        """Test assigning roles to users."""
        # Act
        rbac.assign_role("user-123", "user")
        
        # Assert
        assert "user" in rbac.user_roles["user-123"]
    
    def test_assign_invalid_role_raises_error(self, rbac):
        """Test that assigning invalid role raises error."""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            rbac.assign_role("user-123", "invalid_role")
        
        assert "Unknown role" in str(exc_info.value)
    
    def test_revoke_role_from_user(self, rbac):
        """Test revoking roles from users."""
        # Arrange
        rbac.assign_role("user-123", "user")
        rbac.assign_role("user-123", "admin")
        
        # Act
        rbac.revoke_role("user-123", "admin")
        
        # Assert
        assert "admin" not in rbac.user_roles["user-123"]
        assert "user" in rbac.user_roles["user-123"]
    
    def test_check_permission_deny_by_default(self, rbac):
        """Test that permissions are denied by default."""
        # Act & Assert
        assert rbac.check_permission("unknown-user", Permission.AGENT_CREATE) is False
    
    def test_check_permission_with_role(self, rbac):
        """Test checking permissions for users with roles."""
        # Arrange
        rbac.assign_role("user-123", "user")
        
        # Act & Assert
        assert rbac.check_permission("user-123", Permission.AGENT_CREATE) is True
        assert rbac.check_permission("user-123", Permission.AGENT_READ) is True  # Inherited
        assert rbac.check_permission("user-123", Permission.AGENT_DELETE) is False
    
    def test_check_permission_with_multiple_roles(self, rbac):
        """Test checking permissions with multiple roles."""
        # Arrange
        rbac.assign_role("user-123", "user")
        rbac.assign_role("user-123", "power_user")
        
        # Act & Assert
        assert rbac.check_permission("user-123", Permission.AGENT_UPDATE) is True
        assert rbac.check_permission("user-123", Permission.COALITION_CREATE) is True
    
    def test_admin_has_all_lower_permissions(self, rbac):
        """Test that admin inherits all lower role permissions."""
        # Arrange
        rbac.assign_role("admin-user", "admin")
        
        # Act & Assert
        assert rbac.check_permission("admin-user", Permission.AGENT_READ) is True
        assert rbac.check_permission("admin-user", Permission.AGENT_CREATE) is True
        assert rbac.check_permission("admin-user", Permission.AGENT_UPDATE) is True
        assert rbac.check_permission("admin-user", Permission.AGENT_DELETE) is True
        assert rbac.check_permission("admin-user", Permission.SYSTEM_MONITOR) is True
        assert rbac.check_permission("admin-user", Permission.SYSTEM_ADMIN) is False  # Super admin only


class TestResourcePermissions:
    """Test resource-specific permissions."""
    
    @pytest.fixture
    def rbac(self):
        """Create RBAC manager instance."""
        return RBACManager()
    
    def test_grant_resource_permission(self, rbac):
        """Test granting permission on specific resource."""
        # Arrange
        rbac.assign_role("user-123", "user")
        
        # User has general AGENT_READ but not UPDATE
        assert rbac.check_permission("user-123", Permission.AGENT_UPDATE) is False
        
        # Act - Grant UPDATE on specific agent
        rbac.grant_resource_permission(
            "user-123", "agent", "agent-456", Permission.AGENT_UPDATE
        )
        
        # Assert
        assert rbac.check_resource_permission(
            "user-123", "agent", "agent-456", Permission.AGENT_UPDATE
        ) is True
    
    def test_resource_permission_overrides_general(self, rbac):
        """Test that resource-specific permissions can override general permissions."""
        # Arrange - User with no roles (no general permissions)
        rbac.grant_resource_permission(
            "user-123", "agent", "agent-456", Permission.AGENT_DELETE
        )
        
        # Act & Assert - Should be allowed for specific resource
        assert rbac.check_resource_permission(
            "user-123", "agent", "agent-456", Permission.AGENT_DELETE
        ) is True
        
        # But not for other resources
        assert rbac.check_resource_permission(
            "user-123", "agent", "agent-789", Permission.AGENT_DELETE
        ) is False
    
    def test_resource_permission_isolation(self, rbac):
        """Test that resource permissions are isolated."""
        # Arrange
        rbac.assign_role("user-123", "user")
        rbac.grant_resource_permission(
            "user-123", "agent", "agent-456", Permission.AGENT_UPDATE
        )
        
        # Act & Assert
        # Has permission on specific resource
        assert rbac.check_resource_permission(
            "user-123", "agent", "agent-456", Permission.AGENT_UPDATE
        ) is True
        
        # No permission on different resource
        assert rbac.check_resource_permission(
            "user-123", "agent", "agent-789", Permission.AGENT_UPDATE
        ) is False


class TestSecurityPrinciples:
    """Test security principles in RBAC."""
    
    @pytest.fixture
    def rbac(self):
        """Create RBAC manager instance."""
        return RBACManager()
    
    def test_least_privilege_principle(self, rbac):
        """Test that users have minimum necessary permissions."""
        # Arrange
        rbac.assign_role("regular-user", "user")
        rbac.assign_role("admin-user", "admin")
        
        # Act
        regular_perms = rbac.get_user_permissions("regular-user")
        admin_perms = rbac.get_user_permissions("admin-user")
        
        # Assert
        # Regular user shouldn't have admin permissions
        assert Permission.USER_MANAGE not in regular_perms
        assert Permission.SYSTEM_MONITOR not in regular_perms
        
        # Admin should have more permissions
        assert len(admin_perms) > len(regular_perms)
    
    def test_separation_of_duties(self, rbac):
        """Test separation of duties between roles."""
        # Different roles should have different responsibilities
        user_perms = rbac.roles["user"].get_all_permissions()
        admin_perms = rbac.roles["admin"].permissions  # Direct permissions only
        
        # User permissions shouldn't include admin-only permissions
        admin_only = admin_perms - user_perms
        assert Permission.USER_MANAGE in admin_only
        assert Permission.SYSTEM_MONITOR in admin_only
    
    def test_no_permission_escalation(self, rbac):
        """Test that users cannot escalate their own permissions."""
        # Arrange
        rbac.assign_role("user-123", "user")
        
        # Act - User tries to give themselves admin role
        # This should be prevented by checking USER_MANAGE permission
        can_manage = rbac.check_permission("user-123", Permission.USER_MANAGE)
        
        # Assert
        assert can_manage is False