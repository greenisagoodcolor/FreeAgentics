."""
Deployment Scripts Generator

Generates platform-specific deployment and management scripts.
"""

import logging
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ScriptTemplate:
    """Template for a deployment script."""

    name: str
    filename: str
    content: str
    executable: bool = True
    platform: Optional[str] = None  # None means all platforms


class DeploymentScriptGenerator:
    """Generates deployment scripts for various platforms."""

    def __init__(self) -> None:
        """Initialize script generator."""
        self.templates = self._create_templates()

    def generate_scripts(
        self,
        output_dir: Path,
        target_platform: str,
        hardware_config: Dict[str, Any],
        agent_config: Dict[str, Any],
    ) -> List[Path]:
        """
        Generate all deployment scripts for target platform.

        Args:
            output_dir: Directory to write scripts
            target_platform: Target platform (raspberrypi, mac, jetson, etc.)
            hardware_config: Hardware configuration dict
            agent_config: Agent configuration dict

        Returns:
            List of generated script paths
        """
        logger.info(f"Generating deployment scripts for {target_platform}")

        output_dir.mkdir(parents=True, exist_ok=True)
        generated_files = []

        # Filter templates for platform
        templates = [
            t for t in self.templates if t.platform is None or t.platform == target_platform
        ]

        # Generate each script
        for template in templates:
            content = self._render_template(
                template.content,
                platform=target_platform,
                hardware=hardware_config,
                agent=agent_config,
            )

            script_path = output_dir / template.filename
            with open(script_path, "w") as f:
                f.write(content)

            # Make executable if needed
            if template.executable:
                st = os.stat(script_path)
                os.chmod(script_path, st.st_mode | stat.S_IEXEC)

            generated_files.append(script_path)
            logger.debug(f"Generated {template.name}: {script_path}")

        # Generate platform-specific extras
        if target_platform == "raspberrypi":
            generated_files.extend(self._generate_raspberry_pi_extras(output_dir,
                hardware_config))
        elif target_platform == "mac":
            generated_files.extend(self._generate_mac_extras(output_dir,
                hardware_config))
        elif target_platform == "jetson":
            generated_files.extend(self._generate_jetson_extras(output_dir,
                hardware_config))

        return generated_files

    def _create_templates(self) -> List[ScriptTemplate]:
        """Create script templates."""
        return [
            # Main run script
            ScriptTemplate(
                name="Run Script",
                filename="run.sh",
                content=self._run_script_template(),
            ),
            # Stop script
            ScriptTemplate(
                name="Stop Script",
                filename="stop.sh",
                content=self._stop_script_template(),
            ),
            # Status script
            ScriptTemplate(
                name="Status Script",
                filename="status.sh",
                content=self._status_script_template(),
            ),
            # Backup script
            ScriptTemplate(
                name="Backup Script",
                filename="backup.sh",
                content=self._backup_script_template(),
            ),
            # Restore script
            ScriptTemplate(
                name="Restore Script",
                filename="restore.sh",
                content=self._restore_script_template(),
            ),
            # Monitor script
            ScriptTemplate(
                name="Monitor Script",
                filename="monitor.sh",
                content=self._monitor_script_template(),
            ),
            # Python runner
            ScriptTemplate(
                name="Python Runner",
                filename="freeagentics_agent.py",
                content=self._python_runner_template(),
            ),
        ]

    def _render_template(self, template: str, **kwargs) -> str:
        """Render template with variables."""
        # Simple template rendering
        for key, value in kwargs.items():
            if isinstance(value, dict):
                # Handle nested dicts
                for subkey, subvalue in value.items():
                    template = (
                        template.replace(f"{{{{{key}.{subkey}}}}}", str(subvalue)))
            else:
                template = template.replace(f"{{{{{key}}}}}", str(value))

        return template

    def _run_script_template(self) -> str:
        """Template for main run script."""
        return """#!/bin/bash
# FreeAgentics Agent Run Script
# Platform: {{platform}}

set -e

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment
if [ -f "$BASE_DIR/.env" ]; then
    source "$BASE_DIR/.env"
fi

# Check if already running
if [ -f "$BASE_DIR/agent.pid" ]; then
    PID=$(cat "$BASE_DIR/agent.pid")
    if ps -p $PID > /dev/null 2>&1; then
        echo "Agent is already running (PID: $PID)"
        exit 1
    fi
fi

# Activate virtual environment if exists
if [ -d "$BASE_DIR/venv" ]; then
    source "$BASE_DIR/venv/bin/activate"
fi

# Set environment variables
export FREEAGENTICS_BASE_DIR="$BASE_DIR"
export FREEAGENTICS_CONFIG_DIR="$BASE_DIR/config"
export FREEAGENTICS_MODEL_DIR="$BASE_DIR/model"
export FREEAGENTICS_KNOWLEDGE_DIR="$BASE_DIR/knowledge"
export FREEAGENTICS_LOG_DIR="$BASE_DIR/logs"
export FREEAGENTICS_PLATFORM="{{platform}}"

# Hardware-specific settings
export FREEAGENTICS_CPU_THREADS="{{hardware.cpu_threads}}"
export FREEAGENTICS_MEMORY_LIMIT_MB="{{hardware.memory_limit_mb}}"
export FREEAGENTICS_INFERENCE_THREADS="{{hardware.inference_threads}}"

# Create necessary directories
mkdir -p "$FREEAGENTICS_LOG_DIR"
mkdir -p "$BASE_DIR/checkpoints"
mkdir -p "$BASE_DIR/cache"

# Log rotation
if [ -f "$FREEAGENTICS_LOG_DIR/agent.log" ]; then
    mv "$FREEAGENTICS_LOG_DIR/agent.log" "$FREEAGENTICS_LOG_DIR/agent.log.$(date +%Y%m%d_%H%M%S)"
fi

# Start agent
echo "Starting FreeAgentics Agent..."
echo "Platform: {{platform}}"
echo "Agent ID: {{agent.agent_id}}"
echo "CPU Threads: {{hardware.cpu_threads}}"
echo "Memory Limit: {{hardware.memory_limit_mb}}MB"

# Run with appropriate resource limits
if command -v systemd-run &> /dev/null; then
    # Use systemd-run for resource limits on Linux
    systemd-run --scope -p MemoryLimit={{hardware.memory_limit_mb}}M \
        python "$SCRIPT_DIR/freeagentics_agent.py" \
        > "$FREEAGENTICS_LOG_DIR/agent.log" 2>&1 &
else
    # Basic run
    python "$SCRIPT_DIR/freeagentics_agent.py" \
        > "$FREEAGENTICS_LOG_DIR/agent.log" 2>&1 &
fi

# Save PID
echo $! > "$BASE_DIR/agent.pid"

echo "Agent started with PID: $(cat $BASE_DIR/agent.pid)"
echo "Logs: $FREEAGENTICS_LOG_DIR/agent.log"
"""

    def _stop_script_template(self) -> str:
        """Template for stop script."""
        return """#!/bin/bash
# FreeAgentics Agent Stop Script

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

if [ ! -f "$BASE_DIR/agent.pid" ]; then
    echo "No PID file found. Agent may not be running."
    exit 1
fi

PID=$(cat "$BASE_DIR/agent.pid")

if ! ps -p $PID > /dev/null 2>&1; then
    echo "Agent is not running (stale PID: $PID)"
    rm "$BASE_DIR/agent.pid"
    exit 1
fi

echo "Stopping agent (PID: $PID)..."

# Send SIGTERM for graceful shutdown
kill -TERM $PID

# Wait for process to exit
TIMEOUT=30
COUNTER=0
while ps -p $PID > /dev/null 2>&1; do
    sleep 1
    COUNTER=$((COUNTER + 1))
    if [ $COUNTER -ge $TIMEOUT ]; then
        echo "Agent did not stop gracefully. Forcing..."
        kill -KILL $PID
        break
    fi
done

# Clean up PID file
rm -f "$BASE_DIR/agent.pid"

echo "Agent stopped."
"""

    def _status_script_template(self) -> str:
        """Template for status script."""
        return """#!/bin/bash
# FreeAgentics Agent Status Script

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

echo "FreeAgentics Agent Status"
echo "========================"

# Check if running
if [ -f "$BASE_DIR/agent.pid" ]; then
    PID=$(cat "$BASE_DIR/agent.pid")
    if ps -p $PID > /dev/null 2>&1; then
        echo "Status: RUNNING"
        echo "PID: $PID"

        # Get process info
        if command -v ps &> /dev/null; then
            echo ""
            echo "Process Info:"
            ps -p $PID -o pid,vsz,rss,pcpu,pmem,etime,comm
        fi

        # Get recent logs
        if [ -f "$BASE_DIR/logs/agent.log" ]; then
            echo ""
            echo "Recent Logs:"
            tail -n 10 "$BASE_DIR/logs/agent.log"
        fi
    else
        echo "Status: NOT RUNNING (stale PID file)"
    fi
else
    echo "Status: NOT RUNNING"
fi

echo ""
echo "Configuration:"
echo "- Platform: {{platform}}"
echo "- Agent ID: {{agent.agent_id}}"
echo "- Base Directory: $BASE_DIR"

# Check disk space
echo ""
echo "Disk Usage:"
df -h "$BASE_DIR" | tail -n 1

# Check memory if agent is running
if [ -f "$BASE_DIR/agent.pid" ]; then
    PID=$(cat "$BASE_DIR/agent.pid")
    if ps -p $PID > /dev/null 2>&1; then
        if command -v pmap &> /dev/null; then
            echo ""
            echo "Memory Usage:"
            pmap -x $PID | tail -n 1
        fi
    fi
fi
"""

    def _backup_script_template(self) -> str:
        """Template for backup script."""
        return """#!/bin/bash
# FreeAgentics Agent Backup Script

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="$BASE_DIR/backups"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Generate backup filename
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/agent_backup_$TIMESTAMP.tar.gz"

echo "Creating backup..."

# Create temporary directory for backup
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Copy important files
cp -r "$BASE_DIR/config" "$TEMP_DIR/"
cp -r "$BASE_DIR/model" "$TEMP_DIR/"
cp -r "$BASE_DIR/knowledge" "$TEMP_DIR/"

# Copy checkpoints if they exist
if [ -d "$BASE_DIR/checkpoints" ]; then
    cp -r "$BASE_DIR/checkpoints" "$TEMP_DIR/"
fi

# Create backup archive
tar -czf "$BACKUP_FILE" -C "$TEMP_DIR" .

echo "Backup created: $BACKUP_FILE"
echo "Size: $(du -h "$BACKUP_FILE" | cut -f1)"

# Cleanup old backups (keep last 5)
cd "$BACKUP_DIR"
ls -t agent_backup_*.tar.gz | tail -n +6 | xargs -r rm -f

echo "Backup complete."
"""

    def _restore_script_template(self) -> str:
        """Template for restore script."""
        return """#!/bin/bash
# FreeAgentics Agent Restore Script

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="$BASE_DIR/backups"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <backup_file>"
    echo ""
    echo "Available backups:"
    ls -la "$BACKUP_DIR"/agent_backup_*.tar.gz 2>/dev/null || echo "No backups found"
    exit 1
fi

BACKUP_FILE="$1"

if [ ! -f "$BACKUP_FILE" ]; then
    # Check if it's just a filename in backup dir
    if [ -f "$BACKUP_DIR/$BACKUP_FILE" ]; then
        BACKUP_FILE="$BACKUP_DIR/$BACKUP_FILE"
    else
        echo "Error: Backup file not found: $BACKUP_FILE"
        exit 1
    fi
fi

# Check if agent is running
if [ -f "$BASE_DIR/agent.pid" ]; then
    PID=$(cat "$BASE_DIR/agent.pid")
    if ps -p $PID > /dev/null 2>&1; then
        echo "Error: Agent is running. Please stop it first."
        exit 1
    fi
fi

echo "Restoring from: $BACKUP_FILE"

# Create restore backup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PRE_RESTORE_BACKUP="$BACKUP_DIR/pre_restore_$TIMESTAMP.tar.gz"

echo "Creating pre-restore backup..."
tar -czf "$PRE_RESTORE_BACKUP" -C "$BASE_DIR" \
    config model knowledge checkpoints 2>/dev/null || true

# Extract backup
echo "Extracting backup..."
tar -xzf "$BACKUP_FILE" -C "$BASE_DIR"

echo "Restore complete."
echo "Pre-restore backup saved to: $PRE_RESTORE_BACKUP"
"""

    def _monitor_script_template(self) -> str:
        """Template for monitoring script."""
        return """#!/bin/bash
# FreeAgentics Agent Monitor Script

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# Monitoring interval (seconds)
INTERVAL=${1:-60}

echo "FreeAgentics Agent Monitor"
echo "Interval: ${INTERVAL}s"
echo "Press Ctrl+C to stop"
echo ""

while true; do
    clear
    echo "=== FreeAgentics Agent Monitor ==="
    echo "Time: $(date)"
    echo ""

    # Check if running
    if [ -f "$BASE_DIR/agent.pid" ]; then
        PID=$(cat "$BASE_DIR/agent.pid")
        if ps -p $PID > /dev/null 2>&1; then
            echo "Status: RUNNING (PID: $PID)"

            # CPU and Memory
            if command -v ps &> /dev/null; then
                echo ""
                echo "Resource Usage:"
                ps -p $PID -o pid,vsz,rss,pcpu,pmem,etime
            fi

            # Network connections
            if command -v netstat &> /dev/null; then
                echo ""
                echo "Network Connections:"
                netstat -tnp 2>/dev/null | grep $PID || echo "No active connections"
            fi

            # Recent logs
            if [ -f "$BASE_DIR/logs/agent.log" ]; then
                echo ""
                echo "Recent Activity:"
                tail -n 5 "$BASE_DIR/logs/agent.log"
            fi
        else
            echo "Status: NOT RUNNING (stale PID)"
        fi
    else
        echo "Status: NOT RUNNING"
    fi

    # System resources
    echo ""
    echo "System Resources:"
    echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
    echo "Memory: $(free -h | awk '/^Mem:/ {print $3 "/" $2}')"
    echo "Disk: $(df -h "$BASE_DIR" | tail -1 | awk '{print $3 "/" $2 " (" $5 ")"}')"

    sleep $INTERVAL
done
"""

    def _python_runner_template(self) -> str:
        """Template for Python runner script."""
        return '''#!/usr/bin/env python3
"""
FreeAgentics Agent Runner

Main entry point for the deployed agent.
"""

import os
import sys
import json
import signal
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

# Setup paths
BASE_DIR = (
    Path(os.environ.get('FREEAGENTICS_BASE_DIR', Path(__file__).parent.parent)))
sys.path.insert(0, str(BASE_DIR))

# Configure logging
LOG_DIR = BASE_DIR / 'logs'
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'agent.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class AgentRunner:
    """Runs the FreeAgentics agent."""

    def __init__(self) -> None:
        """Initialize runner."""
        self.running = False
        self.agent = None
        self.config = self._load_config()

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration files."""
        config = {}

        # Load agent config
        agent_config_path = BASE_DIR / 'config' / 'agent_config.json'
        if agent_config_path.exists():
            with open(agent_config_path) as f:
                config['agent'] = json.load(f)

        # Load hardware config
        hardware_config_path = BASE_DIR / 'config' / 'hardware_config.json'
        if hardware_config_path.exists():
            with open(hardware_config_path) as f:
                config['hardware'] = json.load(f)

        # Load runtime config
        runtime_config_path = BASE_DIR / 'config' / 'runtime_config.json'
        if runtime_config_path.exists():
            with open(runtime_config_path) as f:
                config['runtime'] = json.load(f)

        return config

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def run(self):
        """Run the agent."""
        logger.info("Starting FreeAgentics Agent")
        logger.info(f"Agent ID: {self.config.get('agent', {}).get('agent_id',
            'unknown')}")
        logger.info(f"Platform: {os.environ.get('FREEAGENTICS_PLATFORM',
            'unknown')}")

        try:
            # Import agent module (would be the actual agent implementation)
            # from freeagentics.agent import Agent

            # Initialize agent
            # self.agent = Agent.from_config(self.config)

            # For now, simulate agent running
            self.running = True
            logger.info("Agent initialized successfully")

            # Main loop
            while self.running:
                # Simulate agent work
                time.sleep(1)

                # In real implementation:
                # self.agent.step()
                # self.agent.process_messages()
                # etc.

            logger.info("Agent shutdown complete")

        except Exception as e:
            logger.error(f"Agent failed: {e}", exc_info=True)
            sys.exit(1)


def main():
    """Main entry point."""
    runner = AgentRunner()
    runner.run()


if __name__ == '__main__':
    main()
'''

    def _generate_raspberry_pi_extras(
        self, output_dir: Path, hardware_config: Dict[str, Any]
    ) -> List[Path]:
        """Generate Raspberry Pi specific files."""
        extras = []

        # GPIO setup script
        gpio_script = output_dir / "setup_gpio.sh"
        with open(gpio_script, "w") as f:
            f.write(
                """#!/bin/bash
# Setup GPIO for Raspberry Pi

# Enable I2C for sensors
sudo raspi-config nonint do_i2c 0

# Enable SPI if needed
sudo raspi-config nonint do_spi 0

# Set up GPIO permissions
sudo usermod -a -G gpio $USER

echo "GPIO setup complete. Please reboot for changes to take effect."
"""
            )
        os.chmod(gpio_script, 0o755)
        extras.append(gpio_script)

        # Temperature monitoring
        temp_script = output_dir / "check_temp.sh"
        with open(temp_script, "w") as f:
            f.write(
                """#!/bin/bash
# Check Raspberry Pi temperature

TEMP=$(vcgencmd measure_temp | cut -d'=' -f2 | cut -d"'" -f1)
echo "CPU Temperature: ${TEMP}°C"

# Warn if too hot
if (( $(echo "$TEMP > 70" | bc -l) )); then
    echo "WARNING: Temperature is high!"
fi
"""
            )
        os.chmod(temp_script, 0o755)
        extras.append(temp_script)

        return extras

    def _generate_mac_extras(self, output_dir: Path, hardware_config: Dict[str,
        Any]) -> List[Path]:
        """Generate macOS specific files."""
        extras = []

        # LaunchAgent installer
        install_service = output_dir / "install_service.sh"
        with open(install_service, "w") as f:
            f.write(
                """#!/bin/bash
# Install macOS LaunchAgent

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
PLIST_NAME="com.freeagentics.agent.plist"

# Create LaunchAgent plist
cat > ~/Library/LaunchAgents/$PLIST_NAME << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.freeagentics.agent</string>
    <key>ProgramArguments</key>
    <array>
        <string>$BASE_DIR/scripts/run.sh</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$BASE_DIR</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$BASE_DIR/logs/stdout.log</string>
    <key>StandardErrorPath</key>
    <string>$BASE_DIR/logs/stderr.log</string>
</dict>
</plist>
EOF

# Load the service
launchctl load ~/Library/LaunchAgents/$PLIST_NAME

echo "Service installed and started"
"""
            )
        os.chmod(install_service, 0o755)
        extras.append(install_service)

        return extras

    def _generate_jetson_extras(
        self, output_dir: Path, hardware_config: Dict[str, Any]
    ) -> List[Path]:
        """Generate Jetson specific files."""
        extras = []

        # CUDA setup
        cuda_setup = output_dir / "setup_cuda.sh"
        with open(cuda_setup, "w") as f:
            f.write(
                """#!/bin/bash
# Setup CUDA for Jetson

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH= (
    /usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc)

# Set power mode
sudo nvpmodel -m 0  # Max performance mode

# Show current settings
sudo nvpmodel -q
sudo jetson_clocks --show

echo "CUDA setup complete. Please source ~/.bashrc or restart shell."
"""
            )
        os.chmod(cuda_setup, 0o755)
        extras.append(cuda_setup)

        return extras
