"""Mock classes for GMN parser tests.

These mocks are used to fix import errors in GMN parser tests
until the actual classes are implemented.
"""


class GMNSpecification:
    """Mock GMN specification class."""

    def __init__(self, data=None):
        self.data = data or {}


class GMNToPyMDPConverter:
    """Mock GMN to PyMDP converter."""

    def __init__(self):
        pass

    def convert(self, specification):
        """Mock conversion method."""
        return {}


class GMNValidationError(Exception):
    """Mock GMN validation error."""

    pass


class GMNHierarchicalParser:
    """Mock GMN hierarchical parser."""

    def __init__(self):
        pass

    def parse(self, data):
        """Mock parse method."""
        return GMNSpecification(data)


class GMNIntegrationValidator:
    """Mock GMN integration validator."""

    def __init__(self):
        pass

    def validate(self, specification):
        """Mock validate method."""
        return True
