#!/usr/bin/env python3
"""Verify that the FreeAgentics setup is working correctly."""

import subprocess
import sys
from pathlib import Path


def check_command(cmd, name):
    """Check if a command exists."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            print(f"✅ {name}: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {name}: Not found")
            return False
    except:
        print(f"❌ {name}: Not found")
        return False


def check_file(path, name):
    """Check if a file exists."""
    if Path(path).exists():
        print(f"✅ {name}: Found")
        return True
    else:
        print(f"❌ {name}: Not found")
        return False


def check_service(url, name):
    """Check if a service is running."""
    try:
        import requests

        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            print(f"✅ {name}: Running at {url}")
            return True
        else:
            print(f"⚠️  {name}: Responded with {resp.status_code}")
            return True
    except:
        print(f"❌ {name}: Not running")
        return False


def main():
    print("🔍 FreeAgentics Setup Verification")
    print("=" * 50)

    # Check system requirements
    print("\n📋 System Requirements:")
    python_ok = check_command("python3 --version", "Python")
    node_ok = check_command("node --version", "Node.js")
    npm_ok = check_command("npm --version", "npm")

    # Check installation
    print("\n📦 Installation Status:")
    venv_ok = check_file("venv", "Python virtual environment")
    node_modules_ok = check_file("web/node_modules", "Node.js dependencies")
    env_ok = check_file(".env", "Environment configuration")

    # Check running services
    print("\n🚀 Service Status:")
    backend_ok = check_service("http://localhost:8000/health", "Backend API")
    frontend_ok = check_service("http://localhost:3000", "Frontend")

    # Check API endpoints
    if backend_ok:
        print("\n🔌 API Endpoints:")
        check_service("http://localhost:8000/docs", "API Documentation")
        check_service("http://localhost:8000/graphql", "GraphQL")
        check_service("http://localhost:8000/api/v1/dev-config", "Dev Config")

    # Summary
    print("\n" + "=" * 50)
    all_ok = all([python_ok, node_ok, npm_ok, venv_ok, node_modules_ok, env_ok])

    if not (venv_ok and node_modules_ok):
        print("❌ Installation incomplete. Run: make install")
    elif not (backend_ok or frontend_ok):
        print("⚠️  Services not running. Run: make dev")
    elif all_ok and backend_ok and frontend_ok:
        print("✅ Everything is working! Visit http://localhost:3000")
    else:
        print("⚠️  Some components need attention")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
