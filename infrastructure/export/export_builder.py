."""
Hardware Export Package Builder

Creates deployment packages containing agent's GNN model, compressed knowledge graph,
personality, and learned patterns for various hardware targets.
"""

import gzip
import hashlib
import json
import logging
import pickle
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class HardwareTarget:
    """Hardware target configuration."""

    name: str
    platform: str  # raspberrypi, mac, jetson, etc.
    cpu_arch: str  # arm64, x86_64, etc.
    ram_gb: int
    storage_gb: int
    accelerators: List[str] = (
        field(default_factory=list)  # coral_tpu, cuda, etc.)

    # LLM configuration
    llm_model: str = "llama2-7b-q4_K_M"
    llm_context_size: int = 2048
    llm_inference_threads: int = 4

    # Resource limits
    max_memory_mb: int = 0  # 0 = no limit
    max_cpu_percent: int = 80

    def __post_init__(self):
        """Set default max memory based on RAM if not specified."""
        if self.max_memory_mb == 0:
            # Use 75% of available RAM
            self.max_memory_mb = int(self.ram_gb * 1024 * 0.75)


# Predefined hardware targets
HARDWARE_TARGETS = {
    "raspberry_pi_4b": HardwareTarget(
        name="Raspberry Pi 4B",
        platform="raspberrypi",
        cpu_arch="arm64",
        ram_gb=8,
        storage_gb=32,
        accelerators=["coral_tpu"],
    ),
    "mac_mini_m2": HardwareTarget(
        name="Mac Mini M2",
        platform="mac",
        cpu_arch="arm64",
        ram_gb=8,
        storage_gb=256,
        accelerators=["metal"],
    ),
    "jetson_nano": HardwareTarget(
        name="NVIDIA Jetson Nano",
        platform="jetson",
        cpu_arch="arm64",
        ram_gb=4,
        storage_gb=16,
        accelerators=["cuda"],
        llm_inference_threads=2,  # Limited due to lower RAM
        max_memory_mb=3072,  # 3GB limit
    ),
}


@dataclass
class ExportPackage:
    """Export package metadata and contents."""

    package_id: str
    agent_id: str
    target: HardwareTarget
    created_at: datetime

    # Package contents
    model_path: Path
    knowledge_path: Path
    config_path: Path
    scripts_path: Path

    # Metadata
    model_size_mb: float = 0.0
    knowledge_size_mb: float = 0.0
    total_size_mb: float = 0.0
    compression_ratio: float = 1.0

    # Checksums
    checksums: Dict[str, str] = field(default_factory=dict)

    def to_manifest(self) -> Dict[str, Any]:
        """Generate manifest for the package."""
        return {
            "package_id": self.package_id,
            "agent_id": self.agent_id,
            "created_at": self.created_at.isoformat(),
            "target": {
                "name": self.target.name,
                "platform": self.target.platform,
                "cpu_arch": self.target.cpu_arch,
                "ram_gb": self.target.ram_gb,
                "accelerators": self.target.accelerators,
            },
            "contents": {
                "model": {
                    "path": "model/",
                    "size_mb": self.model_size_mb,
                    "checksum": self.checksums.get("model", ""),
                },
                "knowledge": {
                    "path": "knowledge/",
                    "size_mb": self.knowledge_size_mb,
                    "checksum": self.checksums.get("knowledge", ""),
                },
                "config": {
                    "path": "config/",
                    "checksum": self.checksums.get("config", ""),
                },
                "scripts": {
                    "path": "scripts/",
                    "checksum": self.checksums.get("scripts", ""),
                },
            },
            "metrics": {
                "total_size_mb": self.total_size_mb,
                "compression_ratio": self.compression_ratio,
            },
            "requirements": {
                "min_ram_gb": self.target.ram_gb,
                "min_storage_mb": self.total_size_mb * 2,  # 2x for runtime
                "accelerators": self.target.accelerators,
            },
        }


class ExportPackageBuilder:
    """
    Builds deployment packages for various hardware targets.

    Creates optimized packages containing:
    - Compressed GNN model
    - Pruned knowledge graph
    - Personality configuration
    - Deployment scripts
    """

    def __init__(self, output_dir: Path) -> None:
        """Initialize builder with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_package(
        self, agent, target: HardwareTarget, readiness_score: Optional[Any] = None
    ) -> ExportPackage:
        """
        Build deployment package for specific hardware target.

        Args:
            agent: Agent instance to export
            target: Hardware target configuration
            readiness_score: Optional readiness evaluation

        Returns:
            ExportPackage with all components
        """
        logger.info(f"Building export package for agent {agent.id} on {target.name}")

        # Generate package ID
        package_id = self._generate_package_id(agent.id, target)

        # Create temporary build directory
        with tempfile.TemporaryDirectory() as temp_dir:
            build_dir = Path(temp_dir) / package_id
            build_dir.mkdir(parents=True)

            # Create package structure
            model_dir = build_dir / "model"
            knowledge_dir = build_dir / "knowledge"
            config_dir = build_dir / "config"
            scripts_dir = build_dir / "scripts"

            for dir_path in [model_dir, knowledge_dir, config_dir,
                scripts_dir]:
                dir_path.mkdir(parents=True)

            # Export components
            model_size = self._export_model(agent, model_dir, target)
            knowledge_size = (
                self._export_knowledge(agent, knowledge_dir, target))
            self._export_config(agent, config_dir, target, readiness_score)
            self._generate_scripts(scripts_dir, target)

            # Calculate checksums
            checksums = self._calculate_checksums(build_dir)

            # Create package archive
            package_path = self.output_dir / f"{package_id}.tar.gz"
            total_size = self._create_archive(build_dir, package_path)

            # Calculate compression ratio
            original_size = model_size + knowledge_size
            compression_ratio = (
                original_size / total_size if total_size > 0 else 1.0)

            # Create package metadata
            package = ExportPackage(
                package_id=package_id,
                agent_id=agent.id,
                target=target,
                created_at=datetime.now(),
                model_path=package_path / "model",
                knowledge_path=package_path / "knowledge",
                config_path=package_path / "config",
                scripts_path=package_path / "scripts",
                model_size_mb=model_size,
                knowledge_size_mb=knowledge_size,
                total_size_mb=total_size,
                compression_ratio=compression_ratio,
                checksums=checksums,
            )

            # Save manifest
            manifest_path = self.output_dir / f"{package_id}_manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(package.to_manifest(), f, indent=2)

            logger.info(
                f"Package created: {package_path} ({total_size:.1f}MB, "
                f"compression ratio: {compression_ratio:.2f}x)"
            )

            return package

    def _generate_package_id(self, agent_id: str, target: HardwareTarget) -> str:
        """Generate unique package ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_id = target.platform.lower()
        return f"{agent_id}_{target_id}_{timestamp}"

    def _export_model(self, agent, model_dir: Path, target: HardwareTarget) -> float:
        """Export and compress GNN model."""
        logger.info("Exporting GNN model...")

        # Get model data
        model_data = {
            "gnn_model": agent.gnn_model.to_dict(),
            "personality": agent.personality,
            "agent_class": agent.agent_class,
            "version": "1.0.0",
        }

        # Optimize for target hardware
        if target.ram_gb < 8:
            # Apply quantization for low-memory devices
            model_data = self._quantize_model(model_data)

        # Save compressed model
        model_path = model_dir / "gnn_model.pkl.gz"
        with gzip.open(model_path, "wb") as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save human-readable version
        json_path = model_dir / "gnn_model.json"
        with open(json_path, "w") as f:
            json.dump(model_data, f, indent=2, default=str)

        # Calculate size
        size_mb = model_path.stat().st_size / (1024 * 1024)
        return size_mb

    def _export_knowledge(self, agent, knowledge_dir: Path, target: HardwareTarget) -> float:
        """Export and compress knowledge graph."""
        logger.info("Exporting knowledge graph...")

        # Prune knowledge based on target constraints
        if target.storage_gb < 32:
            knowledge_data = (
                self._prune_knowledge(agent.knowledge_graph, max_items=10000))
        else:
            knowledge_data = {
                "experiences": [
                    exp.to_dict() for exp in agent.knowledge_graph.experiences[-50000:]
                ],
                "patterns": {id: p.to_dict() for id, p in agent.knowledge_graph.patterns.items()},
                "relationships": list(agent.knowledge_graph.relationships),
            }

        # Save compressed knowledge
        knowledge_path = knowledge_dir / "knowledge_graph.pkl.gz"
        with gzip.open(knowledge_path, "wb") as f:
            pickle.dump(knowledge_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save summary
        summary_path = knowledge_dir / "knowledge_summary.json"
        summary = {
            "experience_count": len(knowledge_data.get("experiences", [])),
            "pattern_count": len(knowledge_data.get("patterns", {})),
            "relationship_count": len(knowledge_data.get("relationships", [])),
            "pruned": target.storage_gb < 32,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Calculate size
        size_mb = knowledge_path.stat().st_size / (1024 * 1024)
        return size_mb

    def _export_config(
        self,
        agent,
        config_dir: Path,
        target: HardwareTarget,
        readiness_score: Optional[Any] = None,
    ):
        """Export configuration files."""
        logger.info("Exporting configuration...")

        # Agent configuration
        agent_config = {
            "agent_id": agent.id,
            "agent_class": agent.agent_class,
            "personality": agent.personality,
            "created_at": (agent.created_at.isoformat() if hasattr(agent,
                "created_at") else None),
            "readiness": readiness_score.to_dict() if readiness_score else None,
        }

        with open(config_dir / "agent_config.json", "w") as f:
            json.dump(agent_config, f, indent=2)

        # Hardware configuration
        hardware_config = {
            "target": target.name,
            "platform": target.platform,
            "cpu_arch": target.cpu_arch,
            "ram_gb": target.ram_gb,
            "storage_gb": target.storage_gb,
            "accelerators": target.accelerators,
            "llm": {
                "model": target.llm_model,
                "context_size": target.llm_context_size,
                "inference_threads": target.llm_inference_threads,
            },
            "limits": {
                "max_memory_mb": target.max_memory_mb,
                "max_cpu_percent": target.max_cpu_percent,
            },
        }

        with open(config_dir / "hardware_config.json", "w") as f:
            json.dump(hardware_config, f, indent=2)

        # Runtime configuration
        runtime_config = {
            "auto_start": True,
            "log_level": "INFO",
            "checkpoint_interval": 3600,  # 1 hour
            "telemetry_enabled": True,
            "update_check_enabled": True,
            "api_endpoints": {
                "telemetry": "https://api.freeagentics.ai/telemetry",
                "updates": "https://api.freeagentics.ai/updates",
                "knowledge_sync": "https://api.freeagentics.ai/knowledge",
            },
        }

        with open(config_dir / "runtime_config.json", "w") as f:
            json.dump(runtime_config, f, indent=2)

    def _generate_scripts(self, scripts_dir: Path, target: HardwareTarget):
        """Generate deployment scripts for target platform."""
        logger.info(f"Generating deployment scripts for {target.platform}...")

        # Install script
        install_script = self._generate_install_script(target)
        install_path = scripts_dir / "install.sh"
        with open(install_path, "w") as f:
            f.write(install_script)
        install_path.chmod(0o755)

        # Run script
        run_script = self._generate_run_script(target)
        run_path = scripts_dir / "run.sh"
        with open(run_path, "w") as f:
            f.write(run_script)
        run_path.chmod(0o755)

        # Service script (systemd/launchd)
        if target.platform in ["raspberrypi", "jetson"]:
            service_script = self._generate_systemd_service(target)
            service_path = scripts_dir / "freeagentics-agent.service"
            with open(service_path, "w") as f:
                f.write(service_script)
        elif target.platform == "mac":
            service_script = self._generate_launchd_plist(target)
            service_path = scripts_dir / "com.freeagentics.agent.plist"
            with open(service_path, "w") as f:
                f.write(service_script)

        # Update script
        update_script = self._generate_update_script(target)
        update_path = scripts_dir / "update.sh"
        with open(update_path, "w") as f:
            f.write(update_script)
        update_path.chmod(0o755)

        # README
        readme_content = self._generate_readme(target)
        readme_path = scripts_dir / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme_content)

    def _quantize_model(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply quantization to reduce model size for low-memory devices."""
        # Simplified quantization - in practice would use proper quantization
        if "gnn_model" in model_data:
            gnn = model_data["gnn_model"]

            # Quantize float values to int8 range
            for component in ["nodes", "edges"]:
                if component in gnn:
                    for item in gnn[component]:
                        if "parameters" in item:
                            for key, value in item["parameters"].items():
                                if isinstance(value, float):
                                    # Simple quantization to reduce precision
                                    item["parameters"][key] = round(value, 3)

        return model_data

    def _prune_knowledge(self, knowledge_graph, max_items: int = 10000) -> Dict[str,
        Any]:
        """Prune knowledge graph to fit storage constraints."""
        # Keep most recent and high-confidence items
        experiences = sorted(
            knowledge_graph.experiences,
            key=lambda e: (e.timestamp, e.importance),
            reverse=True,
        )[: max_items // 2]

        patterns = sorted(
            knowledge_graph.patterns.values(), key=lambda p: p.confidence,
                reverse=True
        )[: max_items // 4]

        # Keep relationships for retained items
        retained_ids = {e.id for e in experiences} | {p.id for p in patterns}
        relationships = [
            r
            for r in knowledge_graph.relationships
            if r[0] in retained_ids and r[1] in retained_ids
        ]

        return {
            "experiences": [e.to_dict() for e in experiences],
            "patterns": {p.id: p.to_dict() for p in patterns},
            "relationships": relationships,
        }

    def _calculate_checksums(self, build_dir: Path) -> Dict[str, str]:
        """Calculate SHA256 checksums for package contents."""
        checksums = {}

        for component in ["model", "knowledge", "config", "scripts"]:
            component_dir = build_dir / component
            if component_dir.exists():
                # Calculate checksum of all files in directory
                hasher = hashlib.sha256()

                for file_path in sorted(component_dir.rglob("*")):
                    if file_path.is_file():
                        with open(file_path, "rb") as f:
                            hasher.update(f.read())

                checksums[component] = hasher.hexdigest()

        return checksums

    def _create_archive(self, build_dir: Path, output_path: Path) -> float:
        """Create compressed tar.gz archive."""
        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(build_dir, arcname=build_dir.name)

        # Return size in MB
        return output_path.stat().st_size / (1024 * 1024)

    def _generate_install_script(self, target: HardwareTarget) -> str:
        """Generate installation script for target platform."""
        script = f"""#!/bin/bash
# FreeAgentics Agent Installation Script
# Target: {target.name}

set -e

echo "Installing FreeAgentics Agent for {target.name}..."

# Check system requirements
check_requirements() {{
    echo "Checking system requirements..."

    # Check available RAM
    total_ram=$(free -g | awk '/^Mem:/ {{print $2}}')
    if [ "$total_ram" -lt {target.ram_gb} ]; then
        echo "ERROR: Insufficient RAM. Required: {target.ram_gb}GB,
            Available: ${{total_ram}}GB"
        exit 1
    fi

    # Check available storage
    available_storage=$(df -BG . | awk 'NR==2 {{print $4}}' | sed 's/G//')
    if [ "$available_storage" -lt 5 ]; then
        echo "ERROR: Insufficient storage. At least 5GB required."
        exit 1
    fi
}}

# Install dependencies
install_dependencies() {{
    echo "Installing dependencies..."

"""

        # Platform-specific dependencies
        if target.platform == "raspberrypi":
            script += """    # Raspberry Pi dependencies
    sudo apt-get update
    sudo apt-get install -y python3.9 python3-pip python3-venv
    sudo apt-get install -y libopenblas-dev liblapack-dev

    # Install Coral TPU runtime if available
    if [ -n "$(lsusb | grep Google)" ]; then
        echo "Installing Coral TPU runtime..."
        echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
        curl https://packages.cloud.google.com/apt/docs/apt-key.gpg | sudo apt-key add -
        sudo apt-get update
        sudo apt-get install -y libedgetpu1-std
    fi
"""
        elif target.platform == "jetson":
            script += """    # Jetson dependencies
    sudo apt-get update
    sudo apt-get install -y python3.8 python3-pip python3-venv
    sudo apt-get install -y cuda-toolkit-11-4
"""
        elif target.platform == "mac":
            script += """    # macOS dependencies
    # Check for Homebrew
    if ! command -v brew &> /dev/null; then
        echo "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi

    brew install python@3.9
"""

        script += f"""
}}

# Setup Python environment
setup_environment() {{
    echo "Setting up Python environment..."

    # Create virtual environment
    python3 -m venv venv
    source venv/bin/activate

    # Upgrade pip
    pip install --upgrade pip

    # Install Python dependencies
    pip install numpy scipy scikit-learn
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install networkx pandas

    # Install LLM runtime
    pip install llama-cpp-python
}}

# Download LLM model
download_llm() {{
    echo "Downloading LLM model: {target.llm_model}..."

    mkdir -p models
    cd models

    # Download quantized model
    if [ ! -f "{target.llm_model}.gguf" ]; then
        wget "https://huggingface.co/TheBloke/{target.llm_model}/resolve/main/{target.llm_model}.gguf"
    fi

    cd ..
}}

# Extract agent package
extract_package() {{
    echo "Extracting agent package..."

    # Package should be in same directory as install script
    tar -xzf *.tar.gz
}}

# Configure system service
configure_service() {{
    echo "Configuring system service..."

"""

        if target.platform in ["raspberrypi", "jetson"]:
            script += """    # Install systemd service
    sudo cp scripts/freeagentics-agent.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable freeagentics-agent.service
"""
        elif target.platform == "mac":
            script += """    # Install launchd service
    cp scripts/com.freeagentics.agent.plist ~/Library/LaunchAgents/
    launchctl load ~/Library/LaunchAgents/com.freeagentics.agent.plist
"""

        script += """
}

# Main installation
main() {
    check_requirements
    install_dependencies
    setup_environment
    download_llm
    extract_package
    configure_service

    echo "Installation complete!"
    echo "Start the agent with: ./scripts/run.sh"
    echo "Or use the system service: sudo systemctl start freeagentics-agent"
}

main "$@"
"""

        return script

    def _generate_run_script(self, target: HardwareTarget) -> str:
        """Generate run script for the agent."""
        script = f"""#!/bin/bash
# FreeAgentics Agent Run Script
# Target: {target.name}

# Activate virtual environment
source venv/bin/activate

# Set resource limits
ulimit -m {target.max_memory_mb * 1024}  # Memory limit in KB

# Set environment variables
export FREEAGENTICS_CONFIG_DIR="./config"
export FREEAGENTICS_MODEL_DIR="./model"
export FREEAGENTICS_KNOWLEDGE_DIR="./knowledge"
export FREEAGENTICS_LLM_MODEL="./models/{target.llm_model}.gguf"
export FREEAGENTICS_LLM_THREADS={target.llm_inference_threads}
export FREEAGENTICS_LLM_CONTEXT={target.llm_context_size}
export FREEAGENTICS_MAX_CPU_PERCENT={target.max_cpu_percent}

"""

        # Platform-specific environment
        if "coral_tpu" in target.accelerators:
            script += """# Enable Coral TPU
export FREEAGENTICS_USE_TPU=1
"""
        elif "cuda" in target.accelerators:
            script += """# Enable CUDA
export CUDA_VISIBLE_DEVICES=0
export FREEAGENTICS_USE_CUDA=1
"""
        elif "metal" in target.accelerators:
            script += """# Enable Metal acceleration
export FREEAGENTICS_USE_METAL=1
"""

        script += """
# Run the agent
echo "Starting FreeAgentics Agent..."
python -m freeagentics_agent \\
    --config ./config/agent_config.json \\
    --hardware ./config/hardware_config.json \\
    --runtime ./config/runtime_config.json \\
    --log-level INFO \\
    "$@"
"""

        return script

    def _generate_systemd_service(self, target: HardwareTarget) -> str:
        """Generate systemd service file."""
        return f"""[Unit]
Description=FreeAgentics Agent
After=network.target

[Service]
Type=simple
User=pi
Group=pi
WorkingDirectory=/home/pi/freeagentics-agent
ExecStart=/home/pi/freeagentics-agent/scripts/run.sh
Restart=always
RestartSec=10

# Resource limits
MemoryLimit={target.max_memory_mb}M
CPUQuota={target.max_cpu_percent}%

# Environment
Environment="PYTHONUNBUFFERED=1"

[Install]
WantedBy=multi-user.target
"""

    def _generate_launchd_plist(self, target: HardwareTarget) -> str:
        """Generate launchd plist for macOS."""
        return """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.freeagentics.agent</string>

    <key>ProgramArguments</key>
    <array>
        <string>/Users/$USER/freeagentics-agent/scripts/run.sh</string>
    </array>

    <key>WorkingDirectory</key>
    <string>/Users/$USER/freeagentics-agent</string>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>/Users/$USER/freeagentics-agent/logs/stdout.log</string>

    <key>StandardErrorPath</key>
    <string>/Users/$USER/freeagentics-agent/logs/stderr.log</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
    </dict>
</dict>
</plist>
"""

    def _generate_update_script(self, target: HardwareTarget) -> str:
        """Generate update script."""
        return """#!/bin/bash
# FreeAgentics Agent Update Script

set -e

echo "Checking for updates..."

# Stop the service
if systemctl is-active --quiet freeagentics-agent; then
    echo "Stopping agent service..."
    sudo systemctl stop freeagentics-agent
fi

# Backup current installation
if [ -d "./backup" ]; then
    rm -rf ./backup.old
    mv ./backup ./backup.old
fi
mkdir -p ./backup
cp -r model knowledge config ./backup/

# Download and apply update
update_url="https://api.freeagentics.ai/updates/check"
agent_id=$(jq -r '.agent_id' config/agent_config.json)

# Check for updates
update_info= (
    $(curl -s "$update_url?agent_id=$agent_id&platform={target.platform}"))

if [ "$(echo $update_info | jq -r '.update_available')" = "true" ]; then
    echo "Update available: $(echo $update_info | jq -r '.version')"

    # Download update package
    update_package_url=$(echo $update_info | jq -r '.package_url')
    wget -O update.tar.gz "$update_package_url"

    # Extract update
    tar -xzf update.tar.gz

    # Apply update
    cp -r update/* ./

    # Clean up
    rm -rf update update.tar.gz

    echo "Update complete!"
else
    echo "No updates available."
fi

# Restart service
if [ -f /etc/systemd/system/freeagentics-agent.service ]; then
    sudo systemctl start freeagentics-agent
fi
"""

    def _generate_readme(self, target: HardwareTarget) -> str:
        """Generate README for the deployment package."""
        return f"""# FreeAgentics Agent Deployment Package

This package contains a FreeAgentics agent configured for deployment on **{target.name}**.

## Contents

- `model/` - Compressed GNN model and personality configuration
- `knowledge/` - Agent's knowledge graph and learned patterns
- `config/` - Configuration files for agent, hardware, and runtime
- `scripts/` - Deployment and management scripts

## Quick Start

1. **Install the agent:**
   ```bash
   ./scripts/install.sh
   ```

2. **Start the agent:**
   ```bash
   ./scripts/run.sh
   ```

3. **Enable auto-start on boot:**
   - Linux: `sudo systemctl enable freeagentics-agent`
   - macOS: Already configured via launchd

## Hardware Requirements

- **Platform:** {target.platform}
- **Architecture:** {target.cpu_arch}
- **RAM:** {target.ram_gb}GB minimum
- **Storage:** {target.storage_gb}GB minimum
- **Accelerators:** {', '.join(target.accelerators) if target.accelerators else 'None'}

## Configuration

### Agent Configuration
Edit `config/agent_config.json` to modify agent personality and behavior.

### Hardware Configuration
Edit `config/hardware_config.json` to adjust resource limits and LLM settings.

### Runtime Configuration
Edit `config/runtime_config.json` to configure logging, telemetry, and
    API endpoints.

## Management

### View Logs
- Linux: `journalctl -u freeagentics-agent -f`
- macOS: `tail -f ~/freeagentics-agent/logs/stdout.log`

### Stop Agent
- Linux: `sudo systemctl stop freeagentics-agent`
- macOS: `launchctl unload ~/Library/LaunchAgents/com.freeagentics.agent.plist`

### Update Agent
```bash
./scripts/update.sh
```

## Troubleshooting

### Agent Won't Start
1. Check system requirements: `./scripts/install.sh` (will verify)
2. Check logs for errors
3. Ensure LLM model is downloaded: `ls models/`

### High Resource Usage
1. Adjust `max_memory_mb` in `config/hardware_config.json`
2. Reduce `llm_inference_threads` for lower CPU usage
3. Enable resource monitoring in runtime config

### Network Issues
1. Check firewall settings for outbound HTTPS
2. Verify API endpoints in `config/runtime_config.json`
3. Test connectivity: `curl https://api.freeagentics.ai/health`

## Support

- Documentation: https://docs.freeagentics.ai
- Community: https://discord.gg/freeagentics
- Issues: https://github.com/freeagentics/agent/issues

## License

This deployment package is licensed under the terms specified in the FreeAgentics Agent License.
See https://freeagentics.ai/license for details.
"""
