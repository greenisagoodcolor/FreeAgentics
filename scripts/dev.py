#!/usr/bin/env python3
"""Unified development server launcher.

This script starts the FreeAgentics platform in development mode with
automatic provider selection based on available services.
"""

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
        return s.connect_ex(("localhost", port)) == 0


def kill_port(port: int):
    """Kill process on a port."""
    try:
        # Try multiple methods to ensure port is freed
        # Method 1: lsof (most reliable on macOS/Linux)
        result = subprocess.run(f"lsof -ti:{port}", shell=True, capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                try:
                    # Use SIGTERM first, then SIGKILL if needed
                    subprocess.run(f"kill -15 {pid}", shell=True)
                    time.sleep(0.5)
                    # Check if still running
                    check_result = subprocess.run(f"kill -0 {pid}", shell=True, capture_output=True)
                    if check_result.returncode == 0:
                        # Still running, force kill
                        subprocess.run(f"kill -9 {pid}", shell=True)
                except:
                    pass

        # Method 2: fuser (backup, works on Linux)
        try:
            subprocess.run(
                f"fuser -k {port}/tcp", shell=True, capture_output=True, stderr=subprocess.DEVNULL
            )
        except:
            pass

        # Method 3: netstat + kill (works on more systems)
        try:
            # Find process using netstat
            result = subprocess.run(
                f"netstat -tlnp 2>/dev/null | grep :{port}",
                shell=True,
                capture_output=True,
                text=True,
            )
            if result.stdout:
                # Extract PID from netstat output
                lines = result.stdout.strip().split("\n")
                for line in lines:
                    parts = line.split()
                    if len(parts) > 6 and "/" in parts[6]:
                        pid = parts[6].split("/")[0]
                        try:
                            subprocess.run(f"kill -9 {pid}", shell=True)
                        except:
                            pass
        except:
            pass

        # Give OS time to release the port
        time.sleep(2)
    except Exception as e:
        print(f"  ⚠️  Error killing port {port}: {e}")


# Removed print_banner - now inline in main with dynamic port


def setup_environment():
    """Set up environment variables."""
    # Ensure we're in project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Set Python path to include project root
    python_path = os.environ.get("PYTHONPATH", "")
    if str(project_root) not in python_path:
        os.environ["PYTHONPATH"] = (
            f"{project_root}:{python_path}" if python_path else str(project_root)
        )

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
        "-m",
        "uvicorn",
        "api.main:app",
        "--reload",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--log-level",
        os.getenv("LOG_LEVEL", "info").lower(),
    ]

    return subprocess.Popen(cmd)


def start_frontend():
    """Start the frontend server."""
    print("⚛️  Starting Frontend (Next.js)...")

    web_dir = Path.cwd() / "web"
    if not web_dir.exists():
        print("  ⚠️  No web directory found")
        return None, None

    # Check package.json
    if not (web_dir / "package.json").exists():
        print("  ⚠️  No package.json found in web/")
        return None, None

    # Clear port conflicts
    frontend_port = 3000
    if check_port(frontend_port):
        print(f"  → Clearing port {frontend_port}...")
        kill_port(frontend_port)

        # Wait a bit more after killing
        time.sleep(1)

        # If port is still in use after killing, try alternative ports
        if check_port(frontend_port):
            print(f"  → Port {frontend_port} still in use, trying alternatives...")
            for alt_port in [3001, 3002, 3003, 3004, 3005]:
                if not check_port(alt_port):
                    frontend_port = alt_port
                    print(f"  → Using alternative port {frontend_port}")
                    break
            else:
                # Try to find any available port in a wider range
                import socket

                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(("", 0))
                        frontend_port = s.getsockname()[1]
                        print(f"  → Using dynamically allocated port {frontend_port}")
                except:
                    print("  ❌ Could not find available port for frontend")
                    return None, None

    # Install dependencies if needed
    if not (web_dir / "node_modules").exists():
        print("  → Installing Node.js dependencies...")
        subprocess.run(["npm", "install"], cwd=web_dir, check=True)

    # Update the dev script in package.json if using alt port
    if frontend_port != 3000:
        process = subprocess.Popen(
            ["npm", "run", "dev", "--", "-p", str(frontend_port)],
            cwd=web_dir,
            env={**os.environ, "NEXT_TELEMETRY_DISABLED": "1", "PORT": str(frontend_port)},
        )
    else:
        process = subprocess.Popen(
            ["npm", "run", "dev"], cwd=web_dir, env={**os.environ, "NEXT_TELEMETRY_DISABLED": "1"}
        )

    return process, frontend_port


def wait_for_services(frontend_port=3000):
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

    # Wait for frontend (only if frontend_port is not None)
    if frontend_port is not None:
        for i in range(30):
            if check_port(frontend_port):
                print(f"  ✅ Frontend ready on port {frontend_port}")
                break
            time.sleep(1)
        else:
            print("  ⚠️  Frontend may not be running")
    else:
        print("  ⚠️  Frontend not started")

    # Show dev config endpoint
    print("\n🔑 Dev Configuration:")
    print("  Get auth token: curl http://localhost:8000/api/v1/dev-config")


def main():
    """Main entry point."""
    # Setup
    setup_environment()

    # Initialize providers
    print("🔧 Initializing providers...")
    try:
        # Ensure the project root is in sys.path
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from core.providers import init_providers

        init_providers()
    except ImportError as e:
        print(f"  ⚠️  Provider initialization warning (import error): {e}")
        print("  ℹ️  Continuing without provider initialization...")
    except Exception as e:
        print(f"  ⚠️  Provider initialization warning: {e}")
        print("  ℹ️  Continuing without provider initialization...")

    # Start services
    processes = []
    frontend_port = 3000

    backend = start_backend()
    if backend:
        processes.append(backend)

    frontend, frontend_port = start_frontend()
    if frontend:
        processes.append(frontend)

    if not processes:
        print("\n❌ No services started!")
        sys.exit(1)

    # Print banner with actual ports
    print("\n" + "=" * 60)
    print("🚀 FreeAgentics Development Environment")
    print("=" * 60)
    print("🔥 Mode: Dev")
    print("\n📋 Configuration:")
    print(f"  • Backend:  http://localhost:8000")
    if frontend_port is not None:
        print(f"  • Frontend: http://localhost:{frontend_port}")
    else:
        print(f"  • Frontend: Not started (check web/ directory)")
    print(f"  • API Docs: http://localhost:8000/docs")
    print(f"  • GraphQL:  http://localhost:8000/graphql")
    print("\n💡 Dev Mode Features:")
    print("  • SQLite in-memory database")
    print("  • Auto-generated dev token")
    print("  • Mock LLM responses (unless OPENAI_KEY set)")
    print("  • No external dependencies")
    print("\n" + "=" * 60 + "\n")

    # Wait for services
    wait_for_services(frontend_port)

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
