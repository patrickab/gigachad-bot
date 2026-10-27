"""Extract JSON from chatty LLM responses."""

import json
from typing import Any


def extract_json_from_llm(text: str) -> dict[str, Any]:
    """Strip fences, then parse the first JSON object, ignoring any prose around it.

    Raises ValueError (or its subclass `json.JSONDecodeError`) when no object parses.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    if start < 0:
        raise ValueError("No JSON object found in LLM response")
    payload, _end = json.JSONDecoder().raw_decode(stripped, start)
    return payload
