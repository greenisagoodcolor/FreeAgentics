"""Additional classes for GMN parser to support test compatibility."""

from typing import Any, Dict, List


class GMNValidationError(Exception):
    """Exception raised for GMN validation errors."""

    pass


class GMNSchemaValidator:
    """Validator for GMN schema."""

    def __init__(self):
        self.errors = []

    def validate(self, spec: Dict[str, Any]) -> bool:
        """Validate GMN specification."""
        self.errors = []

        # Check required sections
        if isinstance(spec, dict):
            if "nodes" not in spec:
                self.errors.append("Missing 'nodes' section")

            if "edges" not in spec:
                self.errors.append("Missing 'edges' section")
        else:
            self.errors.append("Specification must be a dictionary")

        return len(self.errors) == 0

    def get_errors(self) -> List[str]:
        """Get validation errors."""
        return self.errors
