#!/usr/bin/env python3
"""
Validation script for FreeAgentics Grafana dashboards.
"""

import json
import os
import sys
from pathlib import Path


def validate_dashboard_json(file_path):
    """Validate a single dashboard JSON file."""
    print(f"🧪 Validating: {file_path}")

    try:
        with open(file_path, "r") as f:
            dashboard = json.load(f)

        # Check required fields
        required_fields = ["title", "panels", "time", "refresh"]
        for field in required_fields:
            if field not in dashboard:
                print(f"❌ Missing required field: {field}")
                return False

        # Check panels
        if not isinstance(dashboard["panels"], list):
            print(f"❌ Panels must be a list")
            return False

        if len(dashboard["panels"]) == 0:
            print(f"❌ Dashboard must have at least one panel")
            return False

        # Validate panels
        for i, panel in enumerate(dashboard["panels"]):
            if "id" not in panel:
                print(f"❌ Panel {i} missing id")
                return False

            if "title" not in panel:
                print(f"❌ Panel {i} missing title")
                return False

            if "type" not in panel:
                print(f"❌ Panel {i} missing type")
                return False

            if "targets" not in panel:
                print(f"❌ Panel {i} missing targets")
                return False

        # Check templating
        if "templating" in dashboard:
            templating = dashboard["templating"]
            if "list" in templating:
                for var in templating["list"]:
                    if "name" not in var:
                        print(f"❌ Template variable missing name")
                        return False

        # Check tags
        if "tags" in dashboard:
            if not isinstance(dashboard["tags"], list):
                print(f"❌ Tags must be a list")
                return False

            if "freeagentics" not in dashboard["tags"]:
                print(f"⚠️  Dashboard should have 'freeagentics' tag")

        print(f"✅ Dashboard structure valid")
        print(f"   Title: {dashboard['title']}")
        print(f"   Panels: {len(dashboard['panels'])}")
        print(f"   Refresh: {dashboard['refresh']}")

        if "tags" in dashboard:
            print(f"   Tags: {', '.join(dashboard['tags'])}")

        return True

    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False


def validate_provisioning_config(file_path):
    """Validate provisioning configuration files."""
    print(f"🧪 Validating provisioning config: {file_path}")

    try:
        if file_path.suffix == ".json":
            with open(file_path, "r") as f:
                config = json.load(f)
        else:
            # Assume YAML
            import yaml

            with open(file_path, "r") as f:
                config = yaml.safe_load(f)

        print(f"✅ Provisioning config valid")
        return True

    except Exception as e:
        print(f"❌ Provisioning config error: {e}")
        return False


def main():
    """Main validation function."""
    print("🧪 FreeAgentics Dashboard Validation")
    print("=" * 50)

    # Get dashboard directory
    dashboard_dir = Path(__file__).parent / "monitoring" / "grafana" / "dashboards"
    provisioning_dir = Path(__file__).parent / "monitoring" / "grafana" / "provisioning"

    if not dashboard_dir.exists():
        print(f"❌ Dashboard directory not found: {dashboard_dir}")
        return False

    success = True

    # Validate dashboard files
    dashboard_files = list(dashboard_dir.glob("*.json"))

    if not dashboard_files:
        print(f"❌ No dashboard files found in {dashboard_dir}")
        return False

    print(f"📊 Found {len(dashboard_files)} dashboard files")

    for dashboard_file in dashboard_files:
        if not validate_dashboard_json(dashboard_file):
            success = False
        print()

    # Validate provisioning files
    if provisioning_dir.exists():
        print("📁 Validating provisioning configuration...")

        provisioning_files = list(provisioning_dir.rglob("*.yaml")) + list(
            provisioning_dir.rglob("*.yml")
        )

        for provisioning_file in provisioning_files:
            if not validate_provisioning_config(provisioning_file):
                success = False
        print()

    # Summary
    print("=" * 50)
    if success:
        print("🎉 All dashboard validations passed!")
        print(f"✅ {len(dashboard_files)} dashboards validated successfully")
        print("🚀 Dashboards are ready for deployment!")
    else:
        print("❌ Some dashboard validations failed!")
        print("Please fix the issues above before deploying.")

    return success


if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
