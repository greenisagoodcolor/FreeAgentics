"""Type stubs for pythonjsonlogger library."""

import logging
from typing import Any, Dict

class JsonFormatter(logging.Formatter):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def add_fields(
        self,
        log_record: Dict[str, Any],
        record: logging.LogRecord,
        message_dict: Dict[str, Any],
    ) -> None: ...

# Module-level access
jsonlogger = JsonFormatter
