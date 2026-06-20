"""Extract JSON from chatty LLM responses."""

import json
from typing import Any


def extract_json_from_llm(text: str) -> dict[str, Any]:
    """Strip fences, find balanced braces, parse JSON.

    Handles both ```json fenced blocks and trailing prose after the closing brace.
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

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(stripped)):
        c = stripped[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(stripped[start : i + 1])

    return json.loads(stripped[start:])
