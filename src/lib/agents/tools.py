from typing import Any, Dict, List

from pydantic import BaseModel, Field

# --- Tool Schemas ---

class ReadFile(BaseModel):
    """
    Reads a specific section of a file. Returns content with line numbers.
    Use this to inspect code before editing.
    """
    path: str = Field(..., description="The relative path to the file")
    start_line: int = Field(1, description="The line number to start reading from (1-based)")
    end_line: int = Field(100, description="The line number to end reading at (1-based)")

class EditFile(BaseModel):
    """
    Replaces lines in a file with new content.
    Auto-lints before saving to prevent syntax errors.
    """
    path: str = Field(..., description="The relative path to the file")
    start_line: int = Field(..., description="The line number to start replacing (1-based)")
    end_line: int = Field(..., description="The line number to end replacing (1-based, inclusive)")
    new_content: str = Field(..., description="The new code to insert (can be multiple lines)")

class RunShell(BaseModel):
    """
    Executes a shell command in the sandbox.
    Use this to run tests, git commands, or file operations.
    """
    command: str = Field(..., description="The bash command to run")

class ListDir(BaseModel):
    """
    Lists files in a directory.
    """
    path: str = Field(".", description="The directory path to list")

class SearchCode(BaseModel):
    """
    Searches for a string pattern in the codebase using grep.
    Ignores binary files and hidden directories (.git, .venv).
    """
    query: str = Field(..., description="The string or regex to search for")
    dir: str = Field(".", description="The directory to search in")

# --- Schema Generator ---

def get_aci_tools() -> List[Dict[str, Any]]:
    """
    Returns the OpenAI-compatible tool definitions.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": ReadFile.__doc__.strip(),
                "parameters": ReadFile.model_json_schema()
            }
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": EditFile.__doc__.strip(),
                "parameters": EditFile.model_json_schema()
            }
        },
        {
            "type": "function",
            "function": {
                "name": "run_shell",
                "description": RunShell.__doc__.strip(),
                "parameters": RunShell.model_json_schema()
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_dir",
                "description": ListDir.__doc__.strip(),
                "parameters": ListDir.model_json_schema()
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_code",
                "description": SearchCode.__doc__.strip(),
                "parameters": SearchCode.model_json_schema()
            }
        }
    ]