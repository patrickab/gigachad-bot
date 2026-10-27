"""The stored JSON byte format, shared by ``JsonDocs`` and atomic multi-document writes."""

import json
from typing import Any


def dump_json(data: dict[str, Any] | list[Any]) -> str:
    """Serialize *data* in the stored format: 2-space indent, unescaped unicode, trailing newline."""
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"
