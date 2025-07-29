#!/usr/bin/env python3
"""Unified development server launcher.

This script starts the FreeAgentics platform in development mode with
automatic provider selection based on available services.
"""

import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def check_port(port: int) -> bool:
    """Check if a port is in use."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def kill_port(port: int):
    """Kill process on a port."""
    try:
        subprocess.run(
            f"lsof -ti:{port} | xargs kill -9",
            shell=True,
            capture_output=True
        )
    except:
        pass


def print_banner():
    """Print startup banner with configuration."""
    print("\n" + "="*60)
    print("🚀 FreeAgentics Development Environment")
    print("="*60)
    
    # Always dev mode
    print("🔥 Mode: Dev")
    
    # Show configuration
    print("\n📋 Configuration:")
    print(f"  • Backend:  http://localhost:8000")
    print(f"  • Frontend: http://localhost:3000")
    print(f"  • API Docs: http://localhost:8000/docs")
    print(f"  • GraphQL:  http://localhost:8000/graphql")
    
    print("\n💡 Dev Mode Features:")
    print("  • SQLite in-memory database")
    print("  • Auto-generated dev token")
    print("  • Mock LLM responses (unless OPENAI_KEY set)")
    print("  • No external dependencies")
    
    print("\n" + "="*60 + "\n")


def setup_environment():
    """Set up environment variables."""
    # Ensure we're in project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Set Python path to include project root
    python_path = os.environ.get("PYTHONPATH", "")
    if str(project_root) not in python_path:
        os.environ["PYTHONPATH"] = f"{project_root}:{python_path}" if python_path else str(project_root)
    
    # Load .env if exists
    env_file = project_root / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key] = value
    
    # Set development defaults if not set
    os.environ.setdefault("ENVIRONMENT", "development")
    os.environ.setdefault("LOG_LEVEL", "INFO")
    os.environ.setdefault("DEVELOPMENT_MODE", "true")


def start_backend():
    """Start the backend server."""
    print("🔥 Starting Backend (FastAPI)...")
    
    # Clear port conflicts
    if check_port(8000):
        print("  → Clearing port 8000...")
        kill_port(8000)
        time.sleep(1)
    
    # Ensure we use the venv Python
    project_root = Path(__file__).parent.parent
    venv_python = project_root / "venv" / "bin" / "python"
    
    # Use venv python if it exists, otherwise current python
    python_exe = str(venv_python) if venv_python.exists() else sys.executable
    
    cmd = [
        python_exe,
        "-m", "uvicorn",
        "api.main:app",
        "--reload",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--log-level", os.getenv("LOG_LEVEL", "info").lower()
    ]
    
    return subprocess.Popen(cmd)


def start_frontend():
    """Start the frontend server."""
    print("⚛️  Starting Frontend (Next.js)...")
    
    web_dir = Path.cwd() / "web"
    if not web_dir.exists():
        print("  ⚠️  No web directory found")
        return None
    
    # Check package.json
    if not (web_dir / "package.json").exists():
        print("  ⚠️  No package.json found in web/")
        return None
    
    # Clear port conflicts
    if check_port(3000):
        print("  → Clearing port 3000...")
        kill_port(3000)
        time.sleep(1)
    
    # Install dependencies if needed
    if not (web_dir / "node_modules").exists():
        print("  → Installing Node.js dependencies...")
        subprocess.run(["npm", "install"], cwd=web_dir, check=True)
    
    return subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=web_dir,
        env={**os.environ, "NEXT_TELEMETRY_DISABLED": "1"}
    )


def wait_for_services():
    """Wait for services to be ready."""
    print("\n⏳ Waiting for services to start...")
    
    # Wait for backend
    for i in range(30):
        if check_port(8000):
            print("  ✅ Backend ready")
            break
        time.sleep(1)
    else:
        print("  ⚠️  Backend failed to start")
    
    # Wait for frontend
    for i in range(30):
        if check_port(3000):
            print("  ✅ Frontend ready")
            break
        time.sleep(1)
    else:
        print("  ⚠️  Frontend may not be running")
    
    # Show dev config endpoint
    print("\n🔑 Dev Configuration:")
    print("  Get auth token: curl http://localhost:8000/api/v1/dev-config")


def main():
    """Main entry point."""
    # Setup
    setup_environment()
    print_banner()
    
    # Initialize providers
    print("🔧 Initializing providers...")
    try:
        from core.providers import init_providers
        init_providers()
    except Exception as e:
        print(f"  ⚠️  Provider initialization warning: {e}")
    
    # Start services
    processes = []
    
    backend = start_backend()
    if backend:
        processes.append(backend)
    
    frontend = start_frontend()
    if frontend:
        processes.append(frontend)
    
    if not processes:
        print("\n❌ No services started!")
        sys.exit(1)
    
    # Wait for services
    wait_for_services()
    
    print("\n✨ Development environment ready!")
    print("Press Ctrl+C to stop all services\n")
    
    # Handle shutdown
    def shutdown(signum, frame):
        print("\n🛑 Shutting down...")
        for p in processes:
            p.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    # Keep running
    try:
        while True:
            time.sleep(1)
            # Check if processes are still running
            for p in processes:
                if p.poll() is not None:
                    print(f"\n⚠️  Process {p.args[0]} exited")
    except KeyboardInterrupt:
        shutdown(None, None)


if __name__ == "__main__":
    main()