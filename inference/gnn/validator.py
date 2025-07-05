"""Validator for GNN models and configurations."""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""

    pass


@dataclass
class ValidationResult:
    """Result of model validation."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    model: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


class GMNValidator:
    """Validator for Graph Machine Network models."""

    def __init__(self) -> None:
        """Initialize the validator."""
        self.allowed_architectures = {
            "GCN",
            "GAT",
            "GraphSAGE",
            "GIN",
            "EdgeConv",
            "MPNN",
            "SchNet",
            "DimeNet",
        }

        self.allowed_activations = {"relu", "tanh", "sigmoid", "elu", "leaky_relu", "gelu", "swish"}

        self.parameter_constraints = {
            "num_layers": (1, 20),
            "hidden_dim": (4, 2048),
            "learning_rate": (1e-6, 1.0),
            "dropout": (0.0, 0.9),
            "num_heads": (1, 16),  # For attention mechanisms
            "batch_size": (1, 1024),
            "num_epochs": (1, 10000),
        }

        self.security_patterns = [
            r"__import__",
            r"\bimport\s+\w+",  # import statements
            r"exec\s*\(",
            r"eval\s*\(",
            r"compile\s*\(",
            r"globals\s*\(",
            r"locals\s*\(",
            r"vars\s*\(",
            r"open\s*\(",
            r"file\s*\(",
            r"input\s*\(",
            r"raw_input\s*\(",
            r"os\.system\s*\(",  # os.system calls
            r"subprocess\.",  # subprocess calls
            r"__builtins__",
            r"__file__",
            r"__name__",
            r"\.\.\/",  # path traversal patterns
            r"SELECT.*FROM.*WHERE",  # SQL injection patterns
        ]

        self.circuit_breaker = {
            "max_validation_time": 5.0,  # seconds
            "max_model_size": 1e9,  # bytes
            "max_parameter_count": 1e8,  # parameters
        }

        self.max_validation_errors = 100
        self.validation_timeout = 60.0

        self._custom_rules: Dict[str, Callable] = {}
        self._context: Dict[str, Any] = {}

    def validate(self, model_config: Union[Dict[str, Any], Any]) -> ValidationResult:
        """Validate a model configuration."""
        start_time = time.time()
        errors = []
        warnings = []

        try:
            # Handle ParseResult objects
            if hasattr(model_config, "sections") and hasattr(model_config, "metadata"):
                # This is a ParseResult object
                config_dict = {}
                original_parse_result = model_config

                # Extract sections
                if hasattr(model_config, "sections") and model_config.sections:
                    config_dict.update(model_config.sections)

                # Extract metadata
                if hasattr(model_config, "metadata") and model_config.metadata:
                    config_dict["metadata"] = model_config.metadata

                # Extract name from AST or metadata
                if (
                    hasattr(model_config, "ast")
                    and model_config.ast
                    and hasattr(model_config.ast, "name")
                ):
                    config_dict["name"] = model_config.ast.name
                elif "name" in getattr(model_config, "metadata", {}):
                    config_dict["name"] = model_config.metadata["name"]

                # Create a proper model object for the result
                validated_model = type(
                    "Model",
                    (),
                    {"name": config_dict.get("name", "Unknown Model"), "config": config_dict},
                )()

                model_config = config_dict
            elif not isinstance(model_config, dict):
                errors.append("Model configuration must be a dictionary or ParseResult")
                return ValidationResult(is_valid=False, errors=errors)

            # Validate metadata (required fields)
            metadata = model_config.get("metadata", {})
            if not metadata.get("name"):
                errors.append("Missing required metadata field: name")

            # Validate architecture section
            architecture_section = model_config.get("architecture", {})
            if not architecture_section:
                errors.append("Missing required field: architecture")
            else:
                # Validate architecture type
                arch_type = architecture_section.get("type", "")
                if arch_type not in self.allowed_architectures:
                    errors.append(
                        f"Invalid architecture: {arch_type}. "
                        f"Allowed: {', '.join(self.allowed_architectures)}"
                    )

                # Validate activation function
                activation = architecture_section.get("activation")
                if activation and activation not in self.allowed_activations:
                    errors.append(
                        f"Invalid activation: {activation}. "
                        f"Allowed: {', '.join(self.allowed_activations)}"
                    )

                # Validate dimensions
                hidden_dim = architecture_section.get("hidden_dim")
                if hidden_dim is not None:
                    if not isinstance(hidden_dim, int) or hidden_dim <= 0:
                        errors.append("hidden_dim must be a positive integer")

                layers_count = architecture_section.get("layers")
                if layers_count is not None:
                    if not isinstance(layers_count, int) or layers_count <= 0:
                        errors.append("layers must be a positive integer")

            # Validate parameters section
            parameters = model_config.get("parameters", {})
            param_errors = self._validate_hyperparameters(parameters)
            errors.extend(param_errors)

            # Validate active inference configuration
            active_inference = model_config.get("active_inference", {})
            if active_inference:
                ai_errors = self._validate_active_inference_config(active_inference)
                errors.extend(ai_errors)

            # Generate warnings for memory, performance, and numerical issues
            architecture_warnings = self._check_architecture_warnings(
                architecture_section, parameters
            )
            warnings.extend(architecture_warnings)

            # Security validation
            security_errors = self._validate_security(model_config)
            errors.extend(security_errors)

            # Check for circular dependencies
            if "dependencies" in model_config:
                circular_errors = self._check_circular_dependencies(model_config["dependencies"])
                errors.extend(circular_errors)

            # Validate cross-references
            cross_ref_errors = self._validate_cross_references(model_config)
            errors.extend(cross_ref_errors)

            # Check for definitions circular dependencies (separate from regular dependencies)
            if "definitions" in model_config:
                definitions_errors = self._check_definitions_circular_dependencies(
                    model_config["definitions"]
                )
                errors.extend(definitions_errors)

            # Validate metadata
            if "metadata" in model_config:
                metadata_errors = self._validate_metadata(model_config["metadata"])
                errors.extend(metadata_errors)

            # Apply custom rules
            for rule_name, rule_func in self._custom_rules.items():
                try:
                    # Pass the original ParseResult if available, otherwise processed config
                    rule_input = (
                        original_parse_result
                        if "original_parse_result" in locals()
                        else model_config
                    )
                    rule_result = rule_func(rule_input)
                    if isinstance(rule_result, list):
                        errors.extend(rule_result)
                    elif isinstance(rule_result, str):
                        errors.append(rule_result)
                except Exception as e:
                    warnings.append(f"Custom rule {rule_name} failed: {str(e)}")

            # Check validation time
            elapsed_time = time.time() - start_time
            if elapsed_time > self.circuit_breaker["max_validation_time"]:
                warnings.append(
                    f"Validation took {elapsed_time:.2f}s, "
                    f"exceeding limit of {self.circuit_breaker['max_validation_time']}s"
                )

            # Limit errors
            if len(errors) > self.max_validation_errors:
                errors = errors[: self.max_validation_errors]
                errors.append(f"... and {len(errors) - self.max_validation_errors} more errors")

            # Create result
            is_valid = len(errors) == 0

            # Get the model object if it was a ParseResult
            model_obj = None
            if is_valid:
                if "validated_model" in locals():
                    model_obj = validated_model
                else:
                    model_obj = model_config

            # Extract metadata for result
            architecture_section = model_config.get("architecture", {})
            arch_type = architecture_section.get("type", "Unknown")
            layers_count = architecture_section.get("layers", 0)

            # Create metadata with context if available
            result_metadata = {
                "validation_time": elapsed_time,
                "architecture": arch_type,
                "num_layers": layers_count if isinstance(layers_count, int) else 0,
                "num_parameters": self._estimate_parameters(model_config),
            }

            # Include validation context if set
            if self._context:
                result_metadata["context"] = self._context.copy()

            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                model=model_obj,
                metadata=result_metadata,
            )

        except Exception as e:
            logger.error(f"Validation failed with exception: {e}")
            return ValidationResult(is_valid=False, errors=[f"Validation failed: {str(e)}"])

    def _validate_layer(self, layer: Dict[str, Any], index: int) -> List[str]:
        """Validate a single layer configuration."""
        errors = []

        if not isinstance(layer, dict):
            errors.append(f"Layer {index} must be a dictionary")
            return errors

        # Check layer type
        layer_type = layer.get("type", "")
        if not layer_type:
            errors.append(f"Layer {index} missing type")

        # Validate activation
        if "activation" in layer:
            activation = layer["activation"]
            if activation not in self.allowed_activations:
                errors.append(
                    f"Layer {index} has invalid activation: {activation}. "
                    f"Allowed: {', '.join(self.allowed_activations)}"
                )

        # Validate dimensions
        if "input_dim" in layer:
            input_dim = layer["input_dim"]
            if not isinstance(input_dim, int) or input_dim <= 0:
                errors.append(f"Layer {index} input_dim must be positive integer")

        if "output_dim" in layer:
            output_dim = layer["output_dim"]
            if not isinstance(output_dim, int) or output_dim <= 0:
                errors.append(f"Layer {index} output_dim must be positive integer")

        # Validate layer-specific parameters
        if layer_type == "attention" and "num_heads" in layer:
            num_heads = layer["num_heads"]
            if not isinstance(num_heads, int) or num_heads <= 0:
                errors.append(f"Layer {index} num_heads must be positive integer")

            if "output_dim" in layer and layer["output_dim"] % num_heads != 0:
                errors.append(f"Layer {index} output_dim must be divisible by num_heads")

        return errors

    def _validate_hyperparameters(self, hyperparams: Dict[str, Any]) -> List[str]:
        """Validate hyperparameters."""
        errors = []

        for param, (min_val, max_val) in self.parameter_constraints.items():
            if param in hyperparams:
                value = hyperparams[param]

                if not isinstance(value, (int, float)):
                    errors.append(f"{param} must be numeric")
                elif value < min_val or value > max_val:
                    errors.append(f"{param} value {value} outside range [{min_val}, {max_val}]")

        return errors

    def _validate_active_inference_config(self, active_inference: Dict[str, Any]) -> List[str]:
        """Validate Active Inference configuration."""
        errors = []

        # Check required fields and their values
        required_ai_fields = ["num_states", "num_observations", "num_actions"]
        for field in required_ai_fields:
            if field in active_inference:
                value = active_inference[field]
                if not isinstance(value, int) or value <= 0:
                    errors.append(f"Active Inference {field} must be a positive integer")

        return errors

    def _check_architecture_warnings(
        self, architecture: Dict[str, Any], parameters: Dict[str, Any]
    ) -> List[str]:
        """Check for architecture-related warnings."""
        warnings: List[str] = []

        if not architecture:
            return warnings

        # Memory constraint warnings
        layers = architecture.get("layers", 0)
        hidden_dim = architecture.get("hidden_dim", 0)
        batch_size = parameters.get("batch_size", 32)

        if layers > 50:
            warnings.append(f"High layer count ({layers}) may cause memory issues")
        if hidden_dim > 5000:
            warnings.append(f"Large hidden dimension ({hidden_dim}) may cause memory issues")
        if batch_size > 512:
            warnings.append(f"Large batch size ({batch_size}) may cause memory issues")

        # Performance implications
        arch_type = architecture.get("type", "")
        if arch_type == "GAT":
            num_heads = architecture.get("num_heads", 1)
            if num_heads > 16:
                warnings.append(f"High attention head count ({num_heads}) may impact performance")
        if layers > 8:
            warnings.append(f"Deep network ({layers} layers) may have slow training")

        # Numerical stability warnings
        gradient_clip = parameters.get("gradient_clip")
        if gradient_clip and gradient_clip > 1e5:
            warnings.append(
                f"Very large gradient clip ({gradient_clip}) may cause numerical instability"
            )
        eps = parameters.get("eps")
        if eps and eps < 1e-10:
            warnings.append(f"Very small epsilon ({eps}) may cause numerical instability")

        # Layer compatibility warnings
        if arch_type == "GCN" and architecture.get("edge_features"):
            warnings.append("GCN does not typically use edge features - consider GraphSAGE or GAT")

        return warnings

    def _validate_security(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration for security issues."""
        errors = []

        # Convert config to string for pattern matching
        config_str = str(config)

        for pattern in self.security_patterns:
            if re.search(pattern, config_str, re.IGNORECASE):
                errors.append(f"Security violation: forbidden pattern '{pattern}' detected")

        # Check for suspicious keys
        suspicious_keys = {"__", "exec", "eval", "compile", "import"}
        for key in self._get_all_keys(config):
            for suspicious in suspicious_keys:
                if suspicious in key.lower():
                    errors.append(f"Suspicious key detected: {key}")

        return errors

    def _check_circular_dependencies(self, dependencies: Dict[str, List[str]]) -> List[str]:
        """Check for circular dependencies."""
        errors = []

        def has_cycle(node: str, visited: Set[str], rec_stack: Set[str]) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in dependencies.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        for node in dependencies:
            if node not in visited:
                if has_cycle(node, visited, rec_stack):
                    errors.append(f"Circular dependency detected involving {node}")

        return errors

    def _validate_metadata(self, metadata: Dict[str, Any]) -> List[str]:
        """Validate metadata fields."""
        errors = []

        # Check for required metadata fields
        required_metadata = ["version", "created_at"]
        for field in required_metadata:
            if field not in metadata:
                errors.append(f"Missing required metadata field: {field}")

        # Validate version format
        if "version" in metadata:
            version = metadata["version"]
            if not re.match(r"^\d+\.\d+\.\d+$", str(version)):
                errors.append(f"Invalid version format: {version}")

        # Validate timestamps
        if "created_at" in metadata:
            created_at = metadata["created_at"]
            try:
                # Simple timestamp validation
                if not isinstance(created_at, (int, float, str)):
                    errors.append("created_at must be a timestamp")
            except Exception:
                errors.append("Invalid created_at timestamp")

        return errors

    def _estimate_parameters(self, config: Dict[str, Any]) -> int:
        """Estimate number of parameters in the model."""
        total_params = 0

        layers = config.get("layers", [])
        for i, layer in enumerate(layers):
            if isinstance(layer, dict):
                input_dim = layer.get("input_dim", 0)
                output_dim = layer.get("output_dim", 0)

                # Simple estimation
                if input_dim and output_dim:
                    # Weights + bias
                    total_params += input_dim * output_dim + output_dim

                # Additional parameters for specific layer types
                if layer.get("type") == "attention":
                    num_heads = layer.get("num_heads", 1)
                    # Query, Key, Value projections
                    total_params += 3 * input_dim * output_dim

        return total_params

    def _get_all_keys(self, obj: Any, prefix: str = "") -> List[str]:
        """Get all keys from nested dictionary."""
        keys = []

        if isinstance(obj, dict):
            for key, value in obj.items():
                full_key = f"{prefix}.{key}" if prefix else key
                keys.append(full_key)
                keys.extend(self._get_all_keys(value, full_key))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                keys.extend(self._get_all_keys(item, f"{prefix}[{i}]"))

        return keys

    def _collect_variable_definitions(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Collect variable definitions from configuration."""
        variables = {}

        # Look for variables section
        if "variables" in config:
            variables.update(config["variables"])

        # Look for definitions in metadata
        if "metadata" in config and "variables" in config["metadata"]:
            variables.update(config["metadata"]["variables"])

        return variables

    def add_custom_rule(
        self,
        rule_func: Callable[[Dict[str, Any]], Optional[Union[str, List[str]]]],
        name: Optional[str] = None,
    ) -> None:
        """Add a custom validation rule."""
        if name is None:
            name = f"custom_rule_{len(self._custom_rules)}"
        self._custom_rules[name] = rule_func
        logger.info(f"Added custom validation rule: {name}")

    def set_context(self, context: Dict[str, Any]) -> None:
        """Set validation context for custom rules."""
        self._context = context

    def _validate_cross_references(self, config: Dict[str, Any]) -> List[str]:
        """Validate cross-references in node features."""
        errors = []

        # Check node_features section for undefined references
        if "node_features" in config:
            node_features = config["node_features"]
            if isinstance(node_features, dict) and "features" in node_features:
                features = node_features["features"]
                if isinstance(features, list):
                    # Collect all variable definitions
                    defined_variables = self._collect_variable_definitions(config)

                    # Check each feature reference
                    for feature in features:
                        if isinstance(feature, str) and feature not in defined_variables:
                            errors.append(f"Undefined reference in node features: {feature}")

        return errors

    def _check_definitions_circular_dependencies(self, definitions: Dict[str, Any]) -> List[str]:
        """Check for circular dependencies in definitions section."""
        errors = []

        def has_cycle(node: str, visited: set, rec_stack: set) -> bool:
            visited.add(node)
            rec_stack.add(node)

            # Get dependencies for this node
            node_def = definitions.get(node, {})
            depends_on = node_def.get("depends_on", [])
            if isinstance(depends_on, str):
                depends_on = [depends_on]
            elif not isinstance(depends_on, list):
                depends_on = []

            for neighbor in depends_on:
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        for node in definitions:
            if node not in visited:
                if has_cycle(node, visited, rec_stack):
                    errors.append(f"Circular dependency detected in definitions involving {node}")

        return errors
