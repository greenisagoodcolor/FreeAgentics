#!/usr/bin/env python3
"""
Test script to validate AlertManager configuration and alert routing logic.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml


def validate_alertmanager_config(config_path: str) -> bool:
    """Validate AlertManager configuration file."""
    print(f"🧪 Validating AlertManager configuration: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Check required sections
        required_sections = ["global", "route", "receivers", "inhibit_rules"]
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing required section: {section}")
                return False

        print(f"✅ Configuration structure valid")

        # Validate global configuration
        global_config = config["global"]
        if "resolve_timeout" not in global_config:
            print("❌ Missing resolve_timeout in global configuration")
            return False

        # Validate route configuration
        route_config = config["route"]
        required_route_fields = [
            "group_by",
            "group_wait",
            "group_interval",
            "repeat_interval",
            "receiver",
        ]
        for field in required_route_fields:
            if field not in route_config:
                print(f"❌ Missing required route field: {field}")
                return False

        print(f"✅ Route configuration valid")

        # Validate receivers
        receivers = config["receivers"]
        if not isinstance(receivers, list) or len(receivers) == 0:
            print("❌ No receivers defined")
            return False

        receiver_names = set()
        for receiver in receivers:
            if "name" not in receiver:
                print("❌ Receiver missing name")
                return False
            receiver_names.add(receiver["name"])

        # Check that all referenced receivers exist
        def check_routes(routes: List[Dict], depth: int = 0) -> bool:
            for route in routes:
                if "receiver" in route:
                    if route["receiver"] not in receiver_names:
                        print(f"❌ Referenced receiver '{route['receiver']}' not found")
                        return False
                if "routes" in route:
                    if not check_routes(route["routes"], depth + 1):
                        return False
            return True

        if route_config["receiver"] not in receiver_names:
            print(f"❌ Default receiver '{route_config['receiver']}' not found")
            return False

        if "routes" in route_config:
            if not check_routes(route_config["routes"]):
                return False

        print(f"✅ Receivers configuration valid")

        # Validate inhibit rules
        inhibit_rules = config["inhibit_rules"]
        for rule in inhibit_rules:
            required_fields = ["source_match", "target_match"]
            for field in required_fields:
                if field not in rule and f"{field}_re" not in rule:
                    print(f"❌ Inhibit rule missing {field} or {field}_re")
                    return False

        print(f"✅ Inhibit rules configuration valid")

        # Count different alert routing configurations
        critical_routes = 0
        high_routes = 0
        medium_routes = 0

        def count_severity_routes(routes: List[Dict]) -> None:
            nonlocal critical_routes, high_routes, medium_routes
            for route in routes:
                if "match" in route:
                    if route["match"].get("severity") == "critical":
                        critical_routes += 1
                    elif route["match"].get("severity") == "high":
                        high_routes += 1
                    elif route["match"].get("severity") == "medium":
                        medium_routes += 1
                if "routes" in route:
                    count_severity_routes(route["routes"])

        if "routes" in route_config:
            count_severity_routes(route_config["routes"])

        print(f"📊 Alert routing summary:")
        print(f"   Critical severity routes: {critical_routes}")
        print(f"   High severity routes: {high_routes}")
        print(f"   Medium severity routes: {medium_routes}")
        print(f"   Total receivers: {len(receivers)}")
        print(f"   Total inhibit rules: {len(inhibit_rules)}")

        return True

    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


def validate_prometheus_rules(rules_path: str) -> bool:
    """Validate Prometheus alert rules file."""
    print(f"🧪 Validating Prometheus alert rules: {rules_path}")

    try:
        with open(rules_path, "r") as f:
            rules = yaml.safe_load(f)

        if "groups" not in rules:
            print("❌ No rule groups found")
            return False

        groups = rules["groups"]
        if not isinstance(groups, list) or len(groups) == 0:
            print("❌ No rule groups defined")
            return False

        total_rules = 0
        severity_counts = {"critical": 0, "high": 0, "medium": 0}

        for group in groups:
            if "name" not in group:
                print("❌ Rule group missing name")
                return False

            if "rules" not in group:
                print(f"❌ Rule group '{group['name']}' has no rules")
                return False

            for rule in group["rules"]:
                if "alert" not in rule:
                    print(f"❌ Rule missing alert name in group '{group['name']}'")
                    return False

                required_fields = ["expr", "labels", "annotations"]
                for field in required_fields:
                    if field not in rule:
                        print(f"❌ Rule '{rule['alert']}' missing {field}")
                        return False

                # Check severity label
                if "severity" in rule["labels"]:
                    severity = rule["labels"]["severity"]
                    if severity in severity_counts:
                        severity_counts[severity] += 1

                total_rules += 1

        print(f"✅ Prometheus rules configuration valid")
        print(f"📊 Alert rules summary:")
        print(f"   Total rule groups: {len(groups)}")
        print(f"   Total alert rules: {total_rules}")
        print(f"   Critical alerts: {severity_counts['critical']}")
        print(f"   High severity alerts: {severity_counts['high']}")
        print(f"   Medium severity alerts: {severity_counts['medium']}")

        return True

    except Exception as e:
        print(f"❌ Rules validation failed: {e}")
        return False


def validate_integration_consistency() -> bool:
    """Validate that AlertManager and Prometheus configurations are consistent."""
    print(f"🧪 Validating integration consistency...")

    try:
        # Load both configurations
        alertmanager_path = "/home/green/FreeAgentics/monitoring/alertmanager-intelligent.yml"
        rules_path = "/home/green/FreeAgentics/monitoring/rules/freeagentics-alerts.yml"

        with open(alertmanager_path, "r") as f:
            alertmanager_config = yaml.safe_load(f)

        with open(rules_path, "r") as f:
            rules_config = yaml.safe_load(f)

        # Extract severity levels from alert rules
        rule_severities = set()
        for group in rules_config["groups"]:
            for rule in group["rules"]:
                if "severity" in rule["labels"]:
                    rule_severities.add(rule["labels"]["severity"])

        # Extract severity levels from AlertManager routes
        am_severities = set()

        def extract_severities(routes: List[Dict]) -> None:
            for route in routes:
                if "match" in route and "severity" in route["match"]:
                    am_severities.add(route["match"]["severity"])
                if "routes" in route:
                    extract_severities(route["routes"])

        if "routes" in alertmanager_config["route"]:
            extract_severities(alertmanager_config["route"]["routes"])

        # Check consistency
        missing_in_am = rule_severities - am_severities
        missing_in_rules = am_severities - rule_severities

        if missing_in_am:
            print(f"⚠️  Alert rule severities not handled by AlertManager: {missing_in_am}")

        if missing_in_rules:
            print(f"⚠️  AlertManager severities not defined in rules: {missing_in_rules}")

        if not missing_in_am and not missing_in_rules:
            print("✅ Severity levels consistent between rules and AlertManager")

        # Check for required FreeAgentics-specific alerts
        required_alerts = {
            "FreeAgenticsSystemDown",
            "AgentCoordinationFailure",
            "SystemMemoryUsageCritical",
            "HighSystemErrorRate",
            "FreeEnergyAnomaly",
        }

        found_alerts = set()
        for group in rules_config["groups"]:
            for rule in group["rules"]:
                found_alerts.add(rule["alert"])

        missing_alerts = required_alerts - found_alerts
        if missing_alerts:
            print(f"❌ Missing required FreeAgentics alerts: {missing_alerts}")
            return False

        print(f"✅ All required FreeAgentics alerts are defined")

        return True

    except Exception as e:
        print(f"❌ Integration consistency check failed: {e}")
        return False


def main():
    """Main validation function."""
    print("🧪 FreeAgentics AlertManager Configuration Validation")
    print("=" * 60)

    # Configuration file paths
    alertmanager_config = "/home/green/FreeAgentics/monitoring/alertmanager-intelligent.yml"
    prometheus_rules = "/home/green/FreeAgentics/monitoring/rules/freeagentics-alerts.yml"

    success = True

    # Check if files exist
    if not Path(alertmanager_config).exists():
        print(f"❌ AlertManager configuration file not found: {alertmanager_config}")
        return False

    if not Path(prometheus_rules).exists():
        print(f"❌ Prometheus rules file not found: {prometheus_rules}")
        return False

    # Validate AlertManager configuration
    if not validate_alertmanager_config(alertmanager_config):
        success = False

    print()

    # Validate Prometheus rules
    if not validate_prometheus_rules(prometheus_rules):
        success = False

    print()

    # Validate integration consistency
    if not validate_integration_consistency():
        success = False

    print()
    print("=" * 60)

    if success:
        print("🎉 All AlertManager configuration validations passed!")
        print()
        print("📋 Configuration Summary:")
        print("   ✅ AlertManager configuration is valid")
        print("   ✅ Prometheus alert rules are valid")
        print("   ✅ Integration consistency verified")
        print("   ✅ Required FreeAgentics alerts are defined")
        print("   ✅ Severity-based routing is properly configured")
        print("   ✅ Alert inhibition rules are in place")
        print()
        print("🚀 AlertManager is ready for production deployment!")
        return True
    else:
        print("❌ AlertManager configuration validation failed!")
        print("Please fix the issues above before deploying.")
        return False


if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
