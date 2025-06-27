"""
Module for FreeAgentics Active Inference implementation.
"""

import asyncio
import json
import logging
import platform
import subprocess
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

"""
Ollama Integration for Local LLM
Provides specialized integration with Ollama for local model management and inference.
"""
logger = logging.getLogger(__name__)


@dataclass
class OllamaModel:
    """Ollama model information"""

    name: str
    size: int  # Size in bytes
    digest: str
    modified_at: str
    details: Dict[str, Any]

    @property
    def size_gb(self) -> float:
        """Get size in GB"""
        return self.size / (1024**3)

    @property
    def quantization(self) -> str:
        """Extract quantization from model name"""
        name_lower = self.name.lower()
        if "q3" in name_lower:
            return "q3"
        elif "q4" in name_lower:
            return "q4"
        elif "q5" in name_lower:
            return "q5"
        elif "q8" in name_lower:
            return "q8"
        else:
            return "unknown"


class OllamaManager:
    """
    Manages Ollama installation, models, and operations.
    Features:
    - Automatic Ollama installation
    - Model downloading and management
    - Async streaming responses
    - Model optimization for hardware
    """

    def __init__(self, host: str = "http://localhost:11434") -> None:
        """Initialize Ollama manager"""
        self.host = host
        self.api_base = f"{host}/api"

    async def is_installed(self) -> bool:
        """Check if Ollama is installed and running"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_base}/tags", timeout=2) as response:
                    return response.status == 200
        except Exception:
            return False

    async def install_ollama(self) -> bool:
        """Install Ollama if not present"""
        if await self.is_installed():
            logger.info("Ollama is already installed and running")
            return True
        system = platform.system().lower()
        try:
            if system == "darwin":  # macOS
                logger.info("Installing Ollama on macOS...")
                subprocess.run(
                    ["curl", "-fsSL", "https://ollama.ai/install.sh", "|", "sh"],
                    shell=True,
                    check=True,
                )
            elif system == "linux":
                logger.info("Installing Ollama on Linux...")
                subprocess.run(
                    ["curl", "-fsSL", "https://ollama.ai/install.sh", "|", "sh"],
                    shell=True,
                    check=True,
                )
            elif system == "windows":
                logger.error("Windows installation not yet supported. Please install manually.")
                return False
            # Start Ollama service
            logger.info("Starting Ollama service...")
            if system in ["darwin", "linux"]:
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            # Wait for service to start
            for _ in range(30):
                if await self.is_installed():
                    logger.info("Ollama installed and running successfully")
                    return True
                await asyncio.sleep(1)
            logger.error("Ollama service failed to start")
            return False
        except Exception as e:
            logger.error(f"Failed to install Ollama: {e}")
            return False

    async def list_models(self) -> List[OllamaModel]:
        """List available models"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_base}/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = []
                        for model_data in data.get("models", []):
                            models.append(
                                OllamaModel(
                                    name=model_data["name"],
                                    size=model_data["size"],
                                    digest=model_data["digest"],
                                    modified_at=model_data["modified_at"],
                                    details=model_data.get("details", {}),
                                )
                            )
                        return models
                    else:
                        logger.error(f"Failed to list models: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    async def pull_model(
        self, model_name: str, progress_callback: Optional[callable] = None
    ) -> bool:
        """
        Pull a model from Ollama library.
        Args:
            model_name: Name of model to pull (e.g., 'llama2:7b-q4_K_M')
            progress_callback: Optional callback for progress updates
        Returns:
            True if successful
        """
        try:
            async with aiohttp.ClientSession() as session:
                data = {"name": model_name}
                async with session.post(f"{self.api_base}/pull", json=data) as response:
                    if response.status != 200:
                        logger.error(f"Failed to pull model: {response.status}")
                        return False
                    # Stream progress updates
                    async for line in response.content:
                        if line:
                            try:
                                progress = json.loads(line.decode())
                                if progress_callback:
                                    progress_callback(progress)
                                status = progress.get("status", "")
                                if "error" in progress:
                                    logger.error(f"Pull error: {progress['error']}")
                                    return False
                                elif status:
                                    logger.info(f"Pull status: {status}")
                            except json.JSONDecodeError:
                                continue
                    return True
        except Exception as e:
            logger.error(f"Error pulling model: {e}")
            return False

    async def delete_model(self, model_name: str) -> bool:
        """Delete a model"""
        try:
            async with aiohttp.ClientSession() as session:
                data = {"name": model_name}
                async with session.delete(f"{self.api_base}/delete", json=data) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Error deleting model: {e}")
            return False

    async def generate(
        self, model: str, prompt: str, stream: bool = False, **options
    ) -> AsyncIterator[str]:
        """
        Generate response from model.
        Args:
            model: Model name
            prompt: Input prompt
            stream: Whether to stream response
            **options: Generation options (temperature, max_tokens, etc.)
        Yields:
            Response chunks if streaming, otherwise complete response
        """
        data = {"model": model, "prompt": prompt, "stream": stream, "options": options}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.api_base}/generate", json=data) as response:
                    if response.status != 200:
                        error = await response.text()
                        raise Exception(f"Generation failed: {error}")
                    if stream:
                        async for line in response.content:
                            if line:
                                try:
                                    chunk = json.loads(line.decode())
                                    if "response" in chunk:
                                        yield chunk["response"]
                                except json.JSONDecodeError:
                                    continue
                    else:
                        result = await response.json()
                        yield result["response"]
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise

    async def create_embedding(self, model: str, prompt: str) -> List[float]:
        """Create embedding for text"""
        data = {"model": model, "prompt": prompt}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.api_base}/embeddings", json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["embedding"]
                    else:
                        raise Exception(f"Embedding failed: {response.status}")
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise

    def recommend_model(self, ram_gb: float, storage_gb: float, use_case: str = "general") -> str:
        """
        Recommend best model based on hardware constraints.
        Args:
            ram_gb: Available RAM
            storage_gb: Available storage
            use_case: Intended use (general, code, chat, etc.)
        Returns:
            Recommended model name
        """
        # Model recommendations based on resources
        models = {
            # Model: (min_ram_gb, size_gb, use_cases)
            "tinyllama:latest": (2, 0.6, ["general", "chat"]),
            "phi:latest": (4, 1.6, ["general", "code"]),
            "mistral:7b-instruct-q4_K_M": (6, 4.1, ["general", "chat", "code"]),
            "llama2:7b-q4_K_M": (6, 3.8, ["general", "chat"]),
            "llama2:13b-q4_K_M": (10, 7.4, ["general", "chat", "code"]),
            "codellama:7b-q4_K_M": (6, 3.8, ["code"]),
            "mixtral:8x7b-instruct-q3_K_M": (24, 19.0, ["general", "chat", "code"]),
        }
        # Filter by hardware constraints
        suitable_models = []
        for model_name, (min_ram, size, use_cases) in models.items():
            if ram_gb >= min_ram and storage_gb >= size * 1.5:  # 1.5x for overhead
                if use_case in use_cases:
                    suitable_models.append((model_name, min_ram, size))
        if not suitable_models:
            # Fallback to smallest model
            return "tinyllama:latest"
        # Sort by size (prefer larger models if they fit)
        suitable_models.sort(key=lambda x: x[2], reverse=True)
        recommended = suitable_models[0][0]
        logger.info(
            f"Recommended model for {ram_gb}GB RAM, {storage_gb}GB storage, "
            f"{use_case} use case: {recommended}"
        )
        return recommended

    async def optimize_model_config(
        self, model_name: str, hardware_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize model configuration for specific hardware.
        Args:
            model_name: Model to optimize
            hardware_config: Hardware specifications
        Returns:
            Optimized configuration
        """
        ram_gb = hardware_config.get("ram_gb", 8)
        cpu_cores = hardware_config.get("cpu_cores", 4)
        has_gpu = hardware_config.get("has_gpu", False)
        # Base configuration
        config = {
            "num_thread": min(cpu_cores - 1, 8),
            "num_ctx": 2048,  # Context size
            "num_batch": 8,  # Batch size
            "repeat_penalty": 1.1,
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9,
        }
        # Adjust based on RAM
        if ram_gb < 4:
            config["num_ctx"] = 512
            config["num_batch"] = 4
        elif ram_gb < 8:
            config["num_ctx"] = 1024
            config["num_batch"] = 8
        elif ram_gb >= 16:
            config["num_ctx"] = 4096
            config["num_batch"] = 16
        # GPU optimizations
        if has_gpu:
            config["num_gpu"] = 1
            config["main_gpu"] = 0
            # Estimate layers to offload based on model size
            if "7b" in model_name:
                config["num_gqa"] = 8  # Group query attention
        logger.info(f"Optimized config for {model_name}: {config}")
        return config

    async def create_modelfile(
        self,
        base_model: str,
        system_prompt: str,
        parameters: Dict[str, Any],
        save_path: Path,
    ) -> bool:
        """
        Create a Modelfile for custom model configuration.
        Args:
            base_model: Base model to customize
            system_prompt: System prompt for the model
            parameters: Model parameters
            save_path: Where to save Modelfile
        Returns:
            True if successful
        """
        modelfile_content = f"""# Modelfile for FreeAgentics Agent
FROM {base_model}
# System prompt
SYSTEM "{system_prompt}"
# Parameters
"""
        for param, value in parameters.items():
            modelfile_content += f"PARAMETER {param} {value}\n"
        # Add template if needed
        modelfile_content += """
# Response template
TEMPLATE \"\"\"{{ .System }}
User: {{ .Prompt }}
Assistant: \"\"\"
"""
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(modelfile_content)
            logger.info(f"Created Modelfile at {save_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create Modelfile: {e}")
            return False

    async def create_custom_model(self, name: str, modelfile_path: Path) -> bool:
        """
        Create a custom model from a Modelfile.
        Args:
            name: Name for the custom model
            modelfile_path: Path to Modelfile
        Returns:
            True if successful
        """
        try:
            # Use Ollama CLI to create model
            result = subprocess.run(
                ["ollama", "create", name, "-f", str(modelfile_path)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                logger.info(f"Created custom model: {name}")
                return True
            else:
                logger.error(f"Failed to create model: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error creating custom model: {e}")
            return False


class OllamaAgentAdapter:
    """Adapts Ollama for FreeAgentics agent use"""

    def __init__(self, model_name: str, agent_class: str) -> None:
        """
        Initialize adapter for specific agent class.
        Args:
            model_name: Ollama model to use
            agent_class: Agent class (explorer, merchant, scholar, guardian)
        """
        self.model_name = model_name
        self.agent_class = agent_class
        self.manager = OllamaManager()
        # Agent-specific system prompts
        self.system_prompts = {
            "explorer": """You are an Explorer agent in FreeAgentics. Your role is to:
- Discover new locations and resources
- Map territories and identify patterns
- Share discoveries with other agents
- Optimize exploration paths
Be curious, thorough, and collaborative""",
            "merchant": """You are a Merchant agent in FreeAgentics. Your role is to:
- Identify trading opportunities
- Negotiate fair exchanges
- Manage resources efficiently
- Build trading relationships
Be fair, strategic, and profit-oriented""",
            "scholar": """You are a Scholar agent in FreeAgentics. Your role is to:
- Analyze data and extract patterns
- Formulate theories and hypotheses
- Share knowledge with the community
- Advance collective understanding
Be analytical, precise, and educational""",
            "guardian": """You are a Guardian agent in FreeAgentics. Your role is to:
- Protect assigned territories
- Detect and respond to threats
- Coordinate security measures
- Maintain order and safety
Be vigilant, protective, and responsive""",
        }

    async def setup_agent_model(self) -> bool:
        """Setup customized model for agent"""
        # Create custom model for agent
        system_prompt = self.system_prompts.get(
            self.agent_class,
            "You are a FreeAgentics agent. Be helpful and collaborative.",
        )
        # Agent-specific parameters
        parameters = {
            "temperature": 0.7 if self.agent_class != "guardian" else 0.3,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "seed": -1,  # Random seed
        }
        # Create Modelfile
        modelfile_path = Path(f"/tmp/freeagentics_{self.agent_class}.modelfile")
        success = await self.manager.create_modelfile(
            self.model_name, system_prompt, parameters, modelfile_path
        )
        if success:
            # Create custom model
            custom_name = f"freeagentics-{self.agent_class}:latest"
            return await self.manager.create_custom_model(custom_name, modelfile_path)
        return False

    async def think(self, context: str, options: Optional[Dict[str, Any]] = None) -> str:
        """
        Agent thinking process using Ollama.
        Args:
            context: Current context and observations
            options: Optional generation parameters
        Returns:
            Agent's response/decision
        """
        model = f"freeagentics-{self.agent_class}:latest"
        # Fallback to base model if custom not available
        models = await self.manager.list_models()
        model_names = [m.name for m in models]
        if model not in model_names:
            model = self.model_name
        # Generate response
        response_parts = []
        async for chunk in self.manager.generate(model, context, stream=True, **(options or {})):
            response_parts.append(chunk)
        return "".join(response_parts)
