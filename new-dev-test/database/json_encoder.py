"""Custom JSON encoder for database models."""

import json
from datetime import datetime
from typing import Any
from uuid import UUID


class DatabaseJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime and UUID objects."""

    def default(self, obj: Any) -> Any:
        """Convert non-serializable objects to JSON-serializable formats."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, UUID):
            return str(obj)
        return super().default(obj)


def dumps(obj: Any) -> str:
    """Serialize object to JSON string with custom encoder."""
    return json.dumps(obj, cls=DatabaseJSONEncoder)


def loads(json_str: str) -> Any:
    """Deserialize JSON string."""
    return json.loads(json_str)
