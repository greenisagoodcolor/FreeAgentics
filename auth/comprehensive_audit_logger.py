"""Comprehensive audit logging for all access control decisions.

This module provides detailed audit logging for all RBAC, ABAC, and
resource access control decisions in the FreeAgentics platform.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union

from auth.security_logging import SecurityEventSeverity, SecurityEventType, security_auditor

logger = logging.getLogger(__name__)


class AccessDecisionAuditor:
    """Comprehensive auditor for all access control decisions."""
    
    def __init__(self):
        """Initialize the access decision auditor."""
        self.decision_log: List[Dict[str, Any]] = []
        self.session_log: Dict[str, Dict[str, Any]] = {}
        
    def log_rbac_decision(
        self,
        user_id: str,
        username: str,
        role: str,
        required_permission: str,
        has_permission: bool,
        endpoint: str,
        resource_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log RBAC access decision."""
        
        decision_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision_type": "rbac",
            "user_id": user_id,
            "username": username,
            "role": role,
            "required_permission": required_permission,
            "has_permission": has_permission,
            "endpoint": endpoint,
            "resource_id": resource_id,
            "decision": "allow" if has_permission else "deny",
            "metadata": metadata or {}
        }
        
        self.decision_log.append(decision_entry)
        
        # Log to security auditor
        event_type = SecurityEventType.ACCESS_GRANTED if has_permission else SecurityEventType.ACCESS_DENIED
        severity = SecurityEventSeverity.INFO if has_permission else SecurityEventSeverity.WARNING
        
        security_auditor.log_event(
            event_type,
            severity,
            f"RBAC decision: {username} ({role}) -> {endpoint} [{required_permission}] = {'ALLOW' if has_permission else 'DENY'}",
            user_id=user_id,
            username=username,
            details={
                "decision_type": "rbac",
                "required_permission": required_permission,
                "endpoint": endpoint,
                "resource_id": resource_id,
                "decision": "allow" if has_permission else "deny",
                **(metadata or {})
            }
        )
    
    def log_abac_decision(
        self,
        user_id: str,
        username: str,
        resource_type: str,
        resource_id: Optional[str],
        action: str,
        decision: bool,
        reason: str,
        applied_rules: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log ABAC access decision."""
        
        decision_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision_type": "abac",
            "user_id": user_id,
            "username": username,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "action": action,
            "decision": "allow" if decision else "deny",
            "reason": reason,
            "applied_rules": applied_rules,
            "context": context or {}
        }
        
        self.decision_log.append(decision_entry)
        
        # Log to security auditor
        event_type = SecurityEventType.ACCESS_GRANTED if decision else SecurityEventType.ACCESS_DENIED
        severity = SecurityEventSeverity.INFO if decision else SecurityEventSeverity.WARNING
        
        security_auditor.log_event(
            event_type,
            severity,
            f"ABAC decision: {username} -> {resource_type}:{resource_id or 'N/A'} ({action}) = {'ALLOW' if decision else 'DENY'} - {reason}",
            user_id=user_id,
            username=username,
            details={
                "decision_type": "abac",
                "resource_type": resource_type,
                "resource_id": resource_id,
                "action": action,
                "decision": "allow" if decision else "deny",
                "reason": reason,
                "applied_rules": applied_rules,
                **(context or {})
            }
        )
    
    def log_ownership_check(
        self,
        user_id: str,
        username: str,
        resource_type: str,
        resource_id: str,
        is_owner: bool,
        admin_override: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log ownership check decision."""
        
        decision_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision_type": "ownership",
            "user_id": user_id,
            "username": username,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "is_owner": is_owner,
            "admin_override": admin_override,
            "decision": "allow" if (is_owner or admin_override) else "deny",
            "metadata": metadata or {}
        }
        
        self.decision_log.append(decision_entry)
        
        # Log to security auditor
        decision_result = is_owner or admin_override
        event_type = SecurityEventType.ACCESS_GRANTED if decision_result else SecurityEventType.ACCESS_DENIED
        severity = SecurityEventSeverity.INFO if decision_result else SecurityEventSeverity.WARNING
        
        reason = "owner" if is_owner else ("admin_override" if admin_override else "not_owner")
        
        security_auditor.log_event(
            event_type,
            severity,
            f"Ownership check: {username} -> {resource_type}:{resource_id} = {'ALLOW' if decision_result else 'DENY'} ({reason})",
            user_id=user_id,
            username=username,
            details={
                "decision_type": "ownership",
                "resource_type": resource_type,
                "resource_id": resource_id,
                "is_owner": is_owner,
                "admin_override": admin_override,
                "decision": "allow" if decision_result else "deny",
                "reason": reason,
                **(metadata or {})
            }
        )
    
    def log_session_event(
        self,
        user_id: str,
        username: str,
        event_type: str,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log session-related events."""
        
        session_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "username": username,
            "session_id": session_id,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "metadata": metadata or {}
        }
        
        # Update session log
        if session_id:
            if session_id not in self.session_log:
                self.session_log[session_id] = {
                    "user_id": user_id,
                    "username": username,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "events": []
                }
            self.session_log[session_id]["events"].append(session_entry)
        
        # Log to security auditor
        event_type_map = {
            "session_start": SecurityEventType.LOGIN_SUCCESS,
            "session_end": SecurityEventType.LOGOUT,
            "session_expired": SecurityEventType.TOKEN_EXPIRED,
            "session_invalidated": SecurityEventType.TOKEN_REVOKED,
            "suspicious_activity": SecurityEventType.SUSPICIOUS_PATTERN
        }
        
        security_event_type = event_type_map.get(event_type, SecurityEventType.API_ACCESS)
        severity = SecurityEventSeverity.WARNING if event_type == "suspicious_activity" else SecurityEventSeverity.INFO
        
        security_auditor.log_event(
            security_event_type,
            severity,
            f"Session event: {username} - {event_type}",
            user_id=user_id,
            username=username,
            details={
                "event_type": event_type,
                "session_id": session_id,
                "ip_address": ip_address,
                "user_agent": user_agent,
                **(metadata or {})
            }
        )
    
    def log_rate_limit_event(
        self,
        user_id: Optional[str],
        username: Optional[str],
        ip_address: str,
        endpoint: str,
        rate_limit_type: str,
        limit_exceeded: bool,
        current_count: int,
        limit: int,
        window_seconds: int
    ) -> None:
        """Log rate limiting events."""
        
        rate_limit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "rate_limit",
            "user_id": user_id,
            "username": username,
            "ip_address": ip_address,
            "endpoint": endpoint,
            "rate_limit_type": rate_limit_type,
            "limit_exceeded": limit_exceeded,
            "current_count": current_count,
            "limit": limit,
            "window_seconds": window_seconds
        }
        
        self.decision_log.append(rate_limit_entry)
        
        # Log to security auditor
        if limit_exceeded:
            security_auditor.log_event(
                SecurityEventType.RATE_LIMIT_EXCEEDED,
                SecurityEventSeverity.WARNING,
                f"Rate limit exceeded: {username or 'anonymous'} ({ip_address}) -> {endpoint} ({current_count}/{limit})",
                user_id=user_id,
                username=username,
                details={
                    "ip_address": ip_address,
                    "endpoint": endpoint,
                    "rate_limit_type": rate_limit_type,
                    "current_count": current_count,
                    "limit": limit,
                    "window_seconds": window_seconds
                }
            )
    
    def log_permission_escalation_attempt(
        self,
        user_id: str,
        username: str,
        current_role: str,
        attempted_permission: str,
        endpoint: str,
        blocked: bool,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log permission escalation attempts."""
        
        escalation_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "permission_escalation",
            "user_id": user_id,
            "username": username,
            "current_role": current_role,
            "attempted_permission": attempted_permission,
            "endpoint": endpoint,
            "blocked": blocked,
            "metadata": metadata or {}
        }
        
        self.decision_log.append(escalation_entry)
        
        # Log to security auditor
        security_auditor.log_event(
            SecurityEventType.PRIVILEGE_ESCALATION,
            SecurityEventSeverity.ERROR if blocked else SecurityEventSeverity.CRITICAL,
            f"Permission escalation attempt: {username} ({current_role}) -> {attempted_permission} on {endpoint} [{'BLOCKED' if blocked else 'ALLOWED'}]",
            user_id=user_id,
            username=username,
            details={
                "current_role": current_role,
                "attempted_permission": attempted_permission,
                "endpoint": endpoint,
                "blocked": blocked,
                **(metadata or {})
            }
        )
    
    def log_data_access(
        self,
        user_id: str,
        username: str,
        data_type: str,
        data_id: str,
        action: str,
        sensitivity_level: str,
        authorized: bool,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log data access events."""
        
        data_access_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "data_access",
            "user_id": user_id,
            "username": username,
            "data_type": data_type,
            "data_id": data_id,
            "action": action,
            "sensitivity_level": sensitivity_level,
            "authorized": authorized,
            "metadata": metadata or {}
        }
        
        self.decision_log.append(data_access_entry)
        
        # Log to security auditor
        event_type = SecurityEventType.ACCESS_GRANTED if authorized else SecurityEventType.ACCESS_DENIED
        severity = SecurityEventSeverity.INFO if authorized else SecurityEventSeverity.WARNING
        
        # Increase severity for sensitive data
        if sensitivity_level in ["confidential", "restricted"]:
            severity = SecurityEventSeverity.WARNING if authorized else SecurityEventSeverity.ERROR
        
        security_auditor.log_event(
            event_type,
            severity,
            f"Data access: {username} -> {data_type}:{data_id} ({action}) [{sensitivity_level}] = {'ALLOW' if authorized else 'DENY'}",
            user_id=user_id,
            username=username,
            details={
                "data_type": data_type,
                "data_id": data_id,
                "action": action,
                "sensitivity_level": sensitivity_level,
                "authorized": authorized,
                **(metadata or {})
            }
        )
    
    def get_user_activity_summary(
        self,
        user_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get activity summary for a user."""
        
        if start_time is None:
            start_time = datetime.now(timezone.utc) - timedelta(hours=24)
        if end_time is None:
            end_time = datetime.now(timezone.utc)
        
        # Filter decisions by user and time range
        user_decisions = [
            decision for decision in self.decision_log
            if decision.get("user_id") == user_id and
            start_time <= datetime.fromisoformat(decision["timestamp"].replace("Z", "+00:00")) <= end_time
        ]
        
        # Calculate statistics
        total_decisions = len(user_decisions)
        allowed_decisions = len([d for d in user_decisions if d.get("decision") == "allow"])
        denied_decisions = len([d for d in user_decisions if d.get("decision") == "deny"])
        
        # Group by decision type
        decision_types = {}
        for decision in user_decisions:
            decision_type = decision.get("decision_type", "unknown")
            if decision_type not in decision_types:
                decision_types[decision_type] = {"total": 0, "allowed": 0, "denied": 0}
            decision_types[decision_type]["total"] += 1
            if decision.get("decision") == "allow":
                decision_types[decision_type]["allowed"] += 1
            else:
                decision_types[decision_type]["denied"] += 1
        
        # Get accessed resources
        accessed_resources = set()
        for decision in user_decisions:
            if decision.get("resource_id"):
                resource_type = decision.get("resource_type", "unknown")
                accessed_resources.add(f"{resource_type}:{decision['resource_id']}")
        
        return {
            "user_id": user_id,
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "summary": {
                "total_decisions": total_decisions,
                "allowed_decisions": allowed_decisions,
                "denied_decisions": denied_decisions,
                "success_rate": (allowed_decisions / total_decisions * 100) if total_decisions > 0 else 0
            },
            "decision_types": decision_types,
            "accessed_resources": list(accessed_resources),
            "recent_decisions": user_decisions[-10:]  # Last 10 decisions
        }
    
    def get_security_incidents(
        self,
        severity_threshold: str = "warning",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get security incidents based on access patterns."""
        
        if start_time is None:
            start_time = datetime.now(timezone.utc) - timedelta(hours=24)
        if end_time is None:
            end_time = datetime.now(timezone.utc)
        
        incidents = []
        
        # Analyze patterns in decision log
        time_filtered_decisions = [
            decision for decision in self.decision_log
            if start_time <= datetime.fromisoformat(decision["timestamp"].replace("Z", "+00:00")) <= end_time
        ]
        
        # Group by user for analysis
        user_decisions = {}
        for decision in time_filtered_decisions:
            user_id = decision.get("user_id")
            if user_id:
                if user_id not in user_decisions:
                    user_decisions[user_id] = []
                user_decisions[user_id].append(decision)
        
        # Look for suspicious patterns
        for user_id, decisions in user_decisions.items():
            denied_decisions = [d for d in decisions if d.get("decision") == "deny"]
            
            # High number of denied access attempts
            if len(denied_decisions) > 10:
                incidents.append({
                    "type": "high_denial_rate",
                    "user_id": user_id,
                    "username": decisions[0].get("username"),
                    "severity": "high",
                    "description": f"High number of denied access attempts: {len(denied_decisions)}",
                    "count": len(denied_decisions),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            
            # Permission escalation attempts
            escalation_attempts = [d for d in decisions if d.get("event_type") == "permission_escalation"]
            if escalation_attempts:
                incidents.append({
                    "type": "permission_escalation",
                    "user_id": user_id,
                    "username": decisions[0].get("username"),
                    "severity": "critical",
                    "description": f"Permission escalation attempts: {len(escalation_attempts)}",
                    "count": len(escalation_attempts),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            
            # Access to sensitive resources
            sensitive_access = [
                d for d in decisions 
                if d.get("metadata", {}).get("sensitivity_level") in ["confidential", "restricted"]
            ]
            if sensitive_access:
                incidents.append({
                    "type": "sensitive_data_access",
                    "user_id": user_id,
                    "username": decisions[0].get("username"),
                    "severity": "medium",
                    "description": f"Access to sensitive resources: {len(sensitive_access)}",
                    "count": len(sensitive_access),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
        
        # Filter by severity threshold
        severity_order = ["info", "warning", "high", "critical"]
        threshold_index = severity_order.index(severity_threshold)
        filtered_incidents = [
            incident for incident in incidents
            if severity_order.index(incident["severity"]) >= threshold_index
        ]
        
        return sorted(filtered_incidents, key=lambda x: x["timestamp"], reverse=True)
    
    def generate_audit_report(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive audit report."""
        
        if start_time is None:
            start_time = datetime.now(timezone.utc) - timedelta(hours=24)
        if end_time is None:
            end_time = datetime.now(timezone.utc)
        
        # Filter decisions by time range
        time_filtered_decisions = [
            decision for decision in self.decision_log
            if start_time <= datetime.fromisoformat(decision["timestamp"].replace("Z", "+00:00")) <= end_time
        ]
        
        total_decisions = len(time_filtered_decisions)
        allowed_decisions = len([d for d in time_filtered_decisions if d.get("decision") == "allow"])
        denied_decisions = len([d for d in time_filtered_decisions if d.get("decision") == "deny"])
        
        # Decision type breakdown
        decision_types = {}
        for decision in time_filtered_decisions:
            decision_type = decision.get("decision_type", "unknown")
            if decision_type not in decision_types:
                decision_types[decision_type] = {"total": 0, "allowed": 0, "denied": 0}
            decision_types[decision_type]["total"] += 1
            if decision.get("decision") == "allow":
                decision_types[decision_type]["allowed"] += 1
            else:
                decision_types[decision_type]["denied"] += 1
        
        # User activity
        user_activity = {}
        for decision in time_filtered_decisions:
            user_id = decision.get("user_id")
            if user_id:
                if user_id not in user_activity:
                    user_activity[user_id] = {
                        "username": decision.get("username"),
                        "total": 0,
                        "allowed": 0,
                        "denied": 0
                    }
                user_activity[user_id]["total"] += 1
                if decision.get("decision") == "allow":
                    user_activity[user_id]["allowed"] += 1
                else:
                    user_activity[user_id]["denied"] += 1
        
        # Top accessed resources
        resource_access = {}
        for decision in time_filtered_decisions:
            resource_type = decision.get("resource_type")
            if resource_type:
                if resource_type not in resource_access:
                    resource_access[resource_type] = {"total": 0, "allowed": 0, "denied": 0}
                resource_access[resource_type]["total"] += 1
                if decision.get("decision") == "allow":
                    resource_access[resource_type]["allowed"] += 1
                else:
                    resource_access[resource_type]["denied"] += 1
        
        # Get security incidents
        incidents = self.get_security_incidents(start_time=start_time, end_time=end_time)
        
        return {
            "report_metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "report_type": "comprehensive_access_audit"
            },
            "summary": {
                "total_decisions": total_decisions,
                "allowed_decisions": allowed_decisions,
                "denied_decisions": denied_decisions,
                "success_rate": (allowed_decisions / total_decisions * 100) if total_decisions > 0 else 0,
                "active_users": len(user_activity),
                "accessed_resource_types": len(resource_access),
                "security_incidents": len(incidents)
            },
            "decision_types": decision_types,
            "user_activity": user_activity,
            "resource_access": resource_access,
            "security_incidents": incidents,
            "top_denied_users": sorted(
                [
                    {"user_id": uid, "username": data["username"], "denied_count": data["denied"]}
                    for uid, data in user_activity.items()
                ],
                key=lambda x: x["denied_count"],
                reverse=True
            )[:10]
        }
    
    async def cleanup_old_logs(self, retention_days: int = 30) -> int:
        """Clean up old log entries."""
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
        
        # Clean decision log
        initial_count = len(self.decision_log)
        self.decision_log = [
            decision for decision in self.decision_log
            if datetime.fromisoformat(decision["timestamp"].replace("Z", "+00:00")) > cutoff_date
        ]
        
        # Clean session log
        expired_sessions = [
            session_id for session_id, session_data in self.session_log.items()
            if datetime.fromisoformat(session_data["started_at"].replace("Z", "+00:00")) < cutoff_date
        ]
        
        for session_id in expired_sessions:
            del self.session_log[session_id]
        
        cleaned_count = initial_count - len(self.decision_log) + len(expired_sessions)
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old audit log entries")
        
        return cleaned_count


# Global audit logger instance
comprehensive_auditor = AccessDecisionAuditor()


def get_audit_logger() -> AccessDecisionAuditor:
    """Get the global audit logger instance."""
    return comprehensive_auditor