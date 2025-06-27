"""
Module for FreeAgentics Active Inference implementation.
"""

import logging
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Set

from .model import GMNModel
from .parser import GMNParser

"""
GNN Validator Module
Type checking and syntax validation for GNN models.
Validates types like Real[0, 100], H3Cell[resolution=7], List[Observation], Distribution[State]
"""
logger = logging.getLogger(__name__)


class ValidationConstraints:
    """Constants for validation rules"""

    MAX_NAME_LENGTH = 100
    MIN_NAME_LENGTH = 1
    MAX_EQUATION_LENGTH = 1000
    MAX_VARIABLE_COUNT = 100
    MAX_CONNECTION_COUNT = 500
    MAX_PREFERENCE_WEIGHTS = 10
    VALID_NAME_PATTERN = "^[a-zA-Z_][a-zA-Z0-9_]*$"
    MAX_FILE_SIZE_MB = 10
    MAX_PROCESSING_TIME_SECONDS = 30


@dataclass
class ValidationError:
    """Represents a validation error"""

    field: str
    message: str
    severity: str = "error"
    error_code: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of validation"""

    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]
    processing_time: float = 0.0

    def add_error(self, field: str, message: str, error_code: Optional[str] = None) -> None:
        """Add an error to the result"""
        self.errors.append(ValidationError(field, message, "error", error_code))
        self.is_valid = False

    def add_warning(self, field: str, message: str, error_code: Optional[str] = None) -> None:
        """Add a warning to the result"""
        self.warnings.append(ValidationError(field, message, "warning", error_code))


class CircuitBreaker:
    """Circuit breaker pattern for handling repeated failures"""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60) -> None:
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.is_open = False

    def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute function with circuit breaker protection"""
        if self.is_open:
            if (
                self.last_failure_time is not None
                and time.time() - self.last_failure_time > self.timeout
            ):
                self.is_open = False
                self.failure_count = 0
            else:
                raise Exception("Circuit breaker is open")
        try:
            result = func(*args, **kwargs)
            self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.is_open = True
                logger.error(f"Circuit breaker opened after {self.failure_count} failures")
            raise e


def validate_input(func: Callable) -> Callable:
    """Decorator for input validation on methods"""

    @wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not args and (not kwargs):
            raise ValueError(f"{func.__name__} called with no arguments")
        return func(self, *args, **kwargs)

    return wrapper


@contextmanager
def safe_gnn_processing(file_path: Optional[str] = None) -> Generator[List[Any], None, None]:
    """Context manager for safe GNN processing with resource cleanup"""
    start_time = time.time()
    resources: List[Any] = []
    try:
        if file_path:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"GNN file not found: {file_path}")
            if not path.suffix == ".md" or not path.stem.endswith(".gnn"):
                raise ValueError(f"Invalid GNN file extension: {file_path}")
            if path.stat().st_size > ValidationConstraints.MAX_FILE_SIZE_MB * 1024 * 1024:
                raise ValueError(f"File too large: {path.stat().st_size / 1024 / 1024:.2f}MB")
        yield resources
    except Exception as e:
        logger.error(f"Error in GNN processing: {e}")
        raise
    finally:
        processing_time = time.time() - start_time
        if processing_time > ValidationConstraints.MAX_PROCESSING_TIME_SECONDS:
            logger.warning(f"Processing took too long: {processing_time:.2f}s")
        for resource in resources:
            if hasattr(resource, "close"):
                resource.close()


class GMNValidator:
    """
    Validator for GMN models with comprehensive input validation.
    Ensures:
    - Valid type definitions
    - Consistent variable references
    - Proper constraint specifications
    - Equation validity
    - Preference function correctness
    - Input sanitization and edge case handling
    """

    VALID_TYPES = {
        "Real",
        "Integer",
        "Boolean",
        "String",
        "H3Cell",
        "AgentID",
        "Resource",
        "Action",
        "List",
        "Set",
        "Dict",
        "Distribution",
        "Observation",
        "State",
        "Belief",
        "Goal",
        "Timestamp",
        "Duration",
        "Message",
        "TradeOffer",
    }
    BUILTIN_FUNCTIONS = {
        "bayesian_update",
        "softmax",
        "sigmoid",
        "tanh",
        "exp",
        "log",
        "sqrt",
        "abs",
        "min",
        "max",
        "sum",
        "mean",
        "variance",
        "normalize",
        "argmin",
        "argmax",
        "clip",
        "round",
    }

    def __init__(self) -> None:
        self.defined_variables: set[str] = set()
        self.referenced_variables: set[str] = set()
        self.circuit_breaker = CircuitBreaker()

    @validate_input
    def validate(self, model: GMNModel) -> ValidationResult:
        """
        Validate a GNN model with comprehensive checks.
        Args:
            model: The GNNModel to validate
        Returns:
            ValidationResult with errors and warnings
        """
        start_time = time.time()
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        try:
            with safe_gnn_processing():
                if not isinstance(model, GMNModel):
                    result.add_error("model", "Invalid model type", "INVALID_TYPE")  # type: ignore[unreachable]
                    return result  # type: ignore[unreachable]
                self._validate_basic_structure(model, result)
                self._validate_state_space(model.state_space, result)
                self._validate_observations(model.observations, result)
                self._validate_connections(model.connections, result)
                self._validate_update_equations(model.update_equations, result)
                self._validate_preferences(model.preferences, result)
                self._validate_references(result)
                self._validate_consistency(model, result)
                self._validate_security(model, result)
        except Exception as e:
            logger.error(f"Validation error: {e}")
            result.add_error("validation", f"Validation failed: {str(e)}", "VALIDATION_FAILED")
        finally:
            result.processing_time = time.time() - start_time
        return result

    def _validate_basic_structure(self, model: GMNModel, result: ValidationResult) -> None:
        """Validate basic model structure with edge cases"""
        if not model.name:
            result.add_error("model.name", "Model name is required", "MISSING_NAME")
        elif not isinstance(model.name, str):
            result.add_error(
                "model.name", "Model name must be a string", "INVALID_NAME_TYPE"
            )  # type: ignore[unreachable]
        elif len(model.name) > ValidationConstraints.MAX_NAME_LENGTH:
            result.add_error(
                "model.name",
                f"Model name too long (max {ValidationConstraints.MAX_NAME_LENGTH})",
                "NAME_TOO_LONG",
            )
        elif not re.match(ValidationConstraints.VALID_NAME_PATTERN, model.name):
            result.add_error(
                "model.name",
                "Model name contains invalid characters",
                "INVALID_NAME_FORMAT",
            )
        if not model.description:
            result.add_warning(
                "model.description",
                "Model description is recommended",
                "MISSING_DESCRIPTION",
            )
        elif isinstance(model.description, str) and len(model.description) > 1000:
            result.add_warning("model.description", "Description is very long", "LONG_DESCRIPTION")
        if not model.state_space:
            result.add_error("model.state_space", "State space is required", "MISSING_STATE_SPACE")

    def _validate_state_space(self, state_space: Dict[str, Any], result: ValidationResult) -> None:
        """Validate state space definitions with comprehensive checks"""
        if not state_space:
            result.add_error("state_space", "State space cannot be empty", "EMPTY_STATE_SPACE")
            return
        if not isinstance(state_space, dict):
            result.add_error(  # type: ignore[unreachable]
                "state_space",
                "State space must be a dictionary",
                "INVALID_STATE_SPACE_TYPE",
            )
            return
        if len(state_space) > ValidationConstraints.MAX_VARIABLE_COUNT:
            result.add_error(
                "state_space",
                f"Too many state variables (max {ValidationConstraints.MAX_VARIABLE_COUNT})",
                "TOO_MANY_VARIABLES",
            )
        for var_name, type_def in state_space.items():
            if not isinstance(var_name, str):
                result.add_error(  # type: ignore[unreachable]
                    f"state_space.{var_name}",
                    "Variable name must be a string",
                    "INVALID_VAR_NAME_TYPE",
                )
                continue
            if not re.match(ValidationConstraints.VALID_NAME_PATTERN, var_name):
                result.add_error(
                    f"state_space.{var_name}",
                    f"Invalid variable name: {var_name}",
                    "INVALID_VAR_NAME",
                )
                continue
            if len(var_name) > ValidationConstraints.MAX_NAME_LENGTH:
                result.add_error(
                    f"state_space.{var_name}",
                    "Variable name too long",
                    "VAR_NAME_TOO_LONG",
                )
                continue
            self.defined_variables.add(var_name)
            self._validate_type_definition(f"state_space.{var_name}", type_def, result)

    def _validate_observations(
        self, observations: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Validate observation space definitions"""
        if not observations:
            result.add_warning("observations", "No observations defined", "NO_OBSERVATIONS")
            return
        if not isinstance(observations, dict):
            result.add_error(  # type: ignore[unreachable]
                "observations",
                "Observations must be a dictionary",
                "INVALID_OBSERVATIONS_TYPE",
            )
            return
        for obs_name, type_def in observations.items():
            self.defined_variables.add(obs_name)
            if not isinstance(obs_name, str):
                result.add_error(  # type: ignore[unreachable]
                    f"observations.{obs_name}",
                    "Observation name must be a string",
                    "INVALID_OBS_NAME_TYPE",
                )
                continue
            if not re.match(ValidationConstraints.VALID_NAME_PATTERN, obs_name):
                result.add_error(
                    f"observations.{obs_name}",
                    f"Invalid observation name: {obs_name}",
                    "INVALID_OBS_NAME",
                )
            self._validate_type_definition(f"observations.{obs_name}", type_def, result)

    def _validate_type_definition(
        self, field: str, type_def: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Validate a type definition with edge cases"""
        if not isinstance(type_def, dict):
            result.add_error(field, f"Invalid type definition: {type_def}", "INVALID_TYPE_DEF")  # type: ignore[unreachable]
            return
        base_type = type_def.get("type")
        constraints = type_def.get("constraints")
        if not base_type:
            result.add_error(field, "Type definition missing 'type' field", "MISSING_TYPE")
            return
        if base_type not in self.VALID_TYPES:
            result.add_error(field, f"Unknown type: {base_type}", "UNKNOWN_TYPE")
            return
        if base_type == "Real" and constraints:
            self._validate_real_constraints(field, constraints, result)
        elif base_type == "Integer" and constraints:
            self._validate_integer_constraints(field, constraints, result)
        elif base_type == "H3Cell" and constraints:
            self._validate_h3cell_constraints(field, constraints, result)
        elif base_type in ["List", "Set"] and constraints:
            self._validate_collection_constraints(field, constraints, result)
        elif base_type == "Distribution" and constraints:
            self._validate_distribution_constraints(field, constraints, result)

    def _validate_real_constraints(
        self, field: str, constraints: Any, result: ValidationResult
    ) -> None:
        """Validate Real type constraints with edge cases"""
        if not isinstance(constraints, dict):
            result.add_error(field, "Constraints must be a dictionary", "INVALID_CONSTRAINTS_TYPE")
            return
        if "min" in constraints and "max" in constraints:
            min_val = constraints["min"]
            max_val = constraints["max"]
            if not isinstance(min_val, (int, float)):
                result.add_error(field, f"Invalid min value: {min_val}", "INVALID_MIN_TYPE")
            if not isinstance(max_val, (int, float)):
                result.add_error(field, f"Invalid max value: {max_val}", "INVALID_MAX_TYPE")
            if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                if min_val > max_val:
                    result.add_error(field, f"Min ({min_val}) > Max ({max_val})", "INVALID_RANGE")
                if min_val == max_val:
                    result.add_warning(field, f"Min equals Max ({min_val})", "SINGLE_VALUE_RANGE")
            if isinstance(min_val, float):
                if not -1e308 < min_val < 1e308:
                    result.add_error(field, "Min value out of float range", "MIN_OUT_OF_RANGE")
            if isinstance(max_val, float):
                if not -1e308 < max_val < 1e308:
                    result.add_error(field, "Max value out of float range", "MAX_OUT_OF_RANGE")
        else:
            result.add_warning(field, "Real type should specify min and max", "MISSING_BOUNDS")

    def _validate_integer_constraints(
        self, field: str, constraints: Any, result: ValidationResult
    ) -> None:
        """Validate Integer type constraints"""
        self._validate_real_constraints(field, constraints, result)
        if isinstance(constraints, dict):
            min_val = constraints.get("min")
            max_val = constraints.get("max")
            if isinstance(min_val, float) and min_val != int(min_val):
                result.add_error(field, "Integer min must be whole number", "NON_INTEGER_MIN")
            if isinstance(max_val, float) and max_val != int(max_val):
                result.add_error(field, "Integer max must be whole number", "NON_INTEGER_MAX")

    def _validate_h3cell_constraints(
        self, field: str, constraints: Any, result: ValidationResult
    ) -> None:
        """Validate H3Cell constraints"""
        if not isinstance(constraints, dict):
            result.add_error(
                field,
                "H3Cell constraints must be a dictionary",
                "INVALID_H3_CONSTRAINTS",
            )
            return
        if "resolution" in constraints:
            try:
                res = int(constraints["resolution"])
                if res < 0 or res > 15:
                    result.add_error(
                        field,
                        f"H3 resolution must be 0-15, got {res}",
                        "INVALID_H3_RESOLUTION",
                    )
            except (ValueError, TypeError):
                result.add_error(
                    field,
                    f"Invalid resolution: {constraints['resolution']}",
                    "INVALID_RESOLUTION_TYPE",
                )
        else:
            result.add_warning(field, "H3Cell should specify resolution", "MISSING_H3_RESOLUTION")

    def _validate_collection_constraints(
        self, field: str, constraints: Any, result: ValidationResult
    ) -> None:
        """Validate List/Set constraints"""
        if not isinstance(constraints, dict):
            result.add_error(
                field,
                "Collection constraints must be a dictionary",
                "INVALID_COLLECTION_CONSTRAINTS",
            )
            return
        element_type = constraints.get("element_type")
        if element_type:
            if element_type not in self.VALID_TYPES and element_type not in self.defined_variables:
                result.add_warning(
                    field,
                    f"Unknown element type: {element_type}",
                    "UNKNOWN_ELEMENT_TYPE",
                )
        if "max_size" in constraints:
            try:
                max_size = int(constraints["max_size"])
                if max_size < 0:
                    result.add_error(field, "max_size must be non-negative", "NEGATIVE_MAX_SIZE")
                elif max_size > 10000:
                    result.add_warning(field, "Very large max_size specified", "LARGE_MAX_SIZE")
            except (ValueError, TypeError):
                result.add_error(field, "max_size must be an integer", "INVALID_MAX_SIZE_TYPE")

    def _validate_distribution_constraints(
        self, field: str, constraints: Any, result: ValidationResult
    ) -> None:
        """Validate Distribution constraints"""
        if not isinstance(constraints, dict):
            result.add_error(
                field,
                "Distribution constraints must be a dictionary",
                "INVALID_DIST_CONSTRAINTS",
            )
            return
        element_type = constraints.get("element_type")
        if element_type:
            if element_type not in self.VALID_TYPES and element_type not in self.defined_variables:
                result.add_warning(
                    field,
                    f"Unknown distribution over: {element_type}",
                    "UNKNOWN_DIST_TYPE",
                )

    def _validate_connections(
        self, connections: List[Dict[str, Any]], result: ValidationResult
    ) -> None:
        """Validate connections between nodes"""
        if not isinstance(connections, list):
            result.add_error(  # type: ignore[unreachable]
                "connections", "Connections must be a list", "INVALID_CONNECTIONS_TYPE"
            )
            return
        if len(connections) > ValidationConstraints.MAX_CONNECTION_COUNT:
            result.add_error(
                "connections",
                f"Too many connections (max {ValidationConstraints.MAX_CONNECTION_COUNT})",
                "TOO_MANY_CONNECTIONS",
            )
        seen_connections = set()
        for i, conn in enumerate(connections):
            if not isinstance(conn, dict):
                result.add_error(  # type: ignore[unreachable]
                    f"connections[{i}]",
                    "Connection must be a dictionary",
                    "INVALID_CONNECTION_TYPE",
                )
                continue
            source = conn.get("source")
            target = conn.get("target")
            conn_type = conn.get("type")
            if not all([source, target, conn_type]):
                result.add_error(
                    f"connections[{i}]",
                    "Connection must have source, target, and type",
                    "INCOMPLETE_CONNECTION",
                )
                continue
            if not isinstance(source, str) or not isinstance(target, str):
                result.add_error(
                    f"connections[{i}]",
                    "Source and target must be strings",
                    "INVALID_CONNECTION_FIELDS",
                )
                continue
            if source == target:
                result.add_warning(
                    f"connections[{i}]",
                    f"Self-loop detected: {source} -> {target}",
                    "SELF_LOOP",
                )
            conn_key = (source, target)
            if conn_key in seen_connections:
                result.add_warning(
                    f"connections[{i}]",
                    f"Duplicate connection: {source} -> {target}",
                    "DUPLICATE_CONNECTION",
                )
            seen_connections.add(conn_key)
            self.referenced_variables.add(source)
            self.referenced_variables.add(target)

    def _validate_update_equations(
        self, equations: Dict[str, str], result: ValidationResult
    ) -> None:
        """Validate update equations with security checks"""
        if not isinstance(equations, dict):
            result.add_error(  # type: ignore[unreachable]
                "update_equations",
                "Update equations must be a dictionary",
                "INVALID_EQUATIONS_TYPE",
            )
            return
        for var_name, equation in equations.items():
            if not isinstance(equation, str):
                result.add_error(  # type: ignore[unreachable]
                    f"update_equations.{var_name}",
                    "Equation must be a string",
                    "INVALID_EQUATION_TYPE",
                )
                continue
            if len(equation) > ValidationConstraints.MAX_EQUATION_LENGTH:
                result.add_error(
                    f"update_equations.{var_name}",
                    "Equation too long",
                    "EQUATION_TOO_LONG",
                )
                continue
            if var_name not in self.defined_variables:
                result.add_warning(
                    f"update_equations.{var_name}",
                    f"Updating undefined variable: {var_name}",
                    "UNDEFINED_UPDATE_VAR",
                )
            self._validate_equation(f"update_equations.{var_name}", equation, result)

    def _validate_equation(self, field: str, equation: str, result: ValidationResult) -> None:
        """Validate an equation for syntax, references, and security"""
        dangerous_patterns = [
            "__",
            "exec",
            "eval",
            "import",
            "open",
            "file",
            "subprocess",
            "os.",
            "sys.",
            "globals",
            "locals",
        ]
        for pattern in dangerous_patterns:
            if pattern in equation:
                result.add_error(
                    field,
                    f"Potentially dangerous pattern '{pattern}' in equation",
                    "DANGEROUS_PATTERN",
                )
        var_pattern = re.compile("\\b([a-zA-Z_][a-zA-Z0-9_]*)\\b")
        for match in var_pattern.finditer(equation):
            token = match.group(1)
            if token in self.BUILTIN_FUNCTIONS:
                continue
            if token in ["True", "False", "None", "pi", "e"]:
                continue
            self.referenced_variables.add(token)
        paren_count = equation.count("(") - equation.count(")")
        if paren_count != 0:
            result.add_error(field, "Unbalanced parentheses in equation", "UNBALANCED_PARENS")
        bracket_count = equation.count("[") - equation.count("]")
        if bracket_count != 0:
            result.add_error(field, "Unbalanced brackets in equation", "UNBALANCED_BRACKETS")

    def _validate_preferences(self, preferences: Dict[str, Any], result: ValidationResult) -> None:
        """Validate preference functions"""
        if not preferences:
            result.add_warning(
                "preferences",
                "No preferences defined (required for Active Inference)",
                "NO_PREFERENCES",
            )
            return
        if not isinstance(preferences, dict):
            result.add_error(  # type: ignore[unreachable]
                "preferences",
                "Preferences must be a dictionary",
                "INVALID_PREFERENCES_TYPE",
            )
            return
        for pref_name, pref_def in preferences.items():
            if not pref_name.endswith("_pref"):
                result.add_warning(
                    f"preferences.{pref_name}",
                    "Preference names should end with '_pref'",
                    "INVALID_PREF_NAME",
                )
            if not isinstance(pref_def, dict):
                result.add_error(
                    f"preferences.{pref_name}",
                    "Preference must be a dictionary",
                    "INVALID_PREF_TYPE",
                )
                continue
            input_type = pref_def.get("input")
            output_type = pref_def.get("output")
            if not input_type:
                result.add_error(
                    f"preferences.{pref_name}",
                    "Preference must specify input type",
                    "MISSING_PREF_INPUT",
                )
            if not output_type:
                result.add_error(
                    f"preferences.{pref_name}",
                    "Preference must specify output type",
                    "MISSING_PREF_OUTPUT",
                )
            details = pref_def.get("details", [])
            if not details:
                result.add_warning(
                    f"preferences.{pref_name}",
                    "Preference should specify weighted components",
                    "NO_PREF_DETAILS",
                )
            elif (
                isinstance(details, list)
                and len(details) > ValidationConstraints.MAX_PREFERENCE_WEIGHTS
            ):
                result.add_warning(
                    f"preferences.{pref_name}",
                    f"Too many preference weights (max {ValidationConstraints.MAX_PREFERENCE_WEIGHTS})",
                    "TOO_MANY_WEIGHTS",
                )

    def _validate_references(self, result: ValidationResult) -> None:
        """Cross-validate variable references"""
        undefined = self.referenced_variables - self.defined_variables
        allowed_undefined = {
            "movement_cost",
            "resource_gain",
            "decay_factor",
            "learning_rate",
            "planning_horizon",
            "temperature",
        }
        for var in undefined:
            if var not in allowed_undefined:
                result.add_warning(
                    "references",
                    f"Reference to undefined variable: {var}",
                    "UNDEFINED_REFERENCE",
                )

    def _validate_consistency(self, model: GMNModel, result: ValidationResult) -> None:
        """Validate overall model consistency"""
        if not model.state_space and (not model.observations):
            result.add_error(
                "model",
                "Model must define either state space or observations",
                "EMPTY_MODEL",
            )
        for var in model.update_equations:
            if var not in model.state_space and var not in model.observations:
                result.add_warning(
                    "update_equations",
                    f"Update equation for non-existent variable: {var}",
                    "ORPHANED_EQUATION",
                )
        if model.connections:
            self._check_circular_dependencies(model.connections, result)

    def _check_circular_dependencies(
        self, connections: List[Dict[str, Any]], result: ValidationResult
    ) -> None:
        """Check for circular dependencies in connections"""
        graph: Dict[str, List[str]] = {}
        for conn in connections:
            source = conn.get("source")
            target = conn.get("target")
            if source and target:
                if source not in graph:
                    graph[source] = []
                graph[source].append(target)
        visited = set()
        rec_stack = set()

        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(node)
            return False

        for node in graph:
            if node not in visited:
                if has_cycle(node):
                    result.add_warning(
                        "connections",
                        "Circular dependency detected",
                        "CIRCULAR_DEPENDENCY",
                    )
                    break

    def _validate_security(self, model: GMNModel, result: ValidationResult) -> None:
        """Validate model for security issues"""
        all_strings = []
        all_strings.append(model.name)
        all_strings.append(model.description)
        for eq in model.update_equations.values():
            if isinstance(eq, str):
                all_strings.append(eq)
        suspicious_patterns = [
            "<script",
            "javascript:",
            "onclick",
            "\\$\\{",
            "\\{\\{",
            "\\\\x[0-9a-fA-F]{2}",
            "\\\\u[0-9a-fA-F]{4}",
        ]
        for string in all_strings:
            if not string:
                continue
            for pattern in suspicious_patterns:
                if re.search(pattern, string, re.IGNORECASE):
                    result.add_error(
                        "security",
                        f"Suspicious pattern detected: {pattern}",
                        "SECURITY_RISK",
                    )


if __name__ == "__main__":
    edge_case_gnns = [
        "\n# Model: EmptyModel\n",
        "\n# Model: DangerousModel\n## State Space\nx: Real[0, 100]\n## Update Equations\nx = __import__('os').system('ls')\n",
        "\n# Model: CircularModel\n## State Space\na: Real[0, 1]\nb: Real[0, 1]\n## Connections\na -> b: depends\nb -> a: depends\n",
        "\n# Model: InvalidTypes\n## State Space\nx: InvalidType[0, 100]\ny: Real[min, max]\n",
    ]
    parser = GMNParser()
    validator = GMNValidator()
    for i, gnn_content in enumerate(edge_case_gnns):
        print(f"\n--- Testing edge case {i + 1} ---")
        try:
            with safe_gnn_processing():
                model = parser.parse_content(gnn_content)
                result = validator.validate(model)
                print(f"Valid: {result.is_valid}")
                print(f"Processing time: {result.processing_time:.3f}s")
                print(f"Errors: {len(result.errors)}")
                for error in result.errors:
                    print(f"  - [{error.error_code}] {error.field}: {error.message}")
                print(f"Warnings: {len(result.warnings)}")
                for warning in result.warnings:
                    print(f"  - [{warning.error_code}] {warning.field}: {warning.message}")
        except Exception as e:
            print(f"Exception caught: {e}")
