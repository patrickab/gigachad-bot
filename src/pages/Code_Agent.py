import json
import shlex
from typing import Any, Dict, List, Type

from llm_baseclient.client import LLMClient
from pydantic import BaseModel, Field
import streamlit as st

from lib.streamlit_helper import model_selector, render_messages
from lib.utils.logger import get_logger
from src.lib.agents.docker_sandbox import DockerSandbox

logger = get_logger()

# --- TOOL DEFINITIONS (Pydantic) ---


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


class SearchCode(BaseModel):
    """
    Searches for a string pattern in the codebase.
    """

    query: str = Field(..., description="The string to search for")
    dir: str = Field(".", description="The directory to search in")


class ListDir(BaseModel):
    """
    Lists files in a directory.
    """

    path: str = Field(".", description="The directory path to list")


def _model_to_tool_schema(model: Type[BaseModel], name: str) -> Dict[str, Any]:
    """Converts a Pydantic model to an OpenAI/LiteLLM compatible tool definition."""
    schema = model.model_json_schema()
    return {
        "type": "function",
        "function": {
            "name": name or model.__name__,
            "description": model.__doc__.strip() if model.__doc__ else "",
            "parameters": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            },
        },
    }


# --- CORE LOGIC ---


class CodeAgentTools:
    """
    The Runtime implementation of the Agent-Computer Interface (ACI).
    Maps abstract Pydantic tools to concrete DockerSandbox commands.
    """

    def __init__(self, sandbox: DockerSandbox) -> None:
        self.sandbox = sandbox
        logger.debug("ACI Tools: CodeAgentTools initialized with sandbox: %s", sandbox)
        # Map tool names to Pydantic models and internal methods
        self.tool_registry = {
            "read_file": (ReadFile, self._read_file),
            "edit_file": (EditFile, self._edit_file),
            "run_shell": (RunShell, self._run_shell),
            "search_code": (SearchCode, self._search_code),
            "list_dir": (ListDir, self._list_dir),
        }

    def get_definitions(self) -> List[Dict[str, Any]]:
        """Exposes the schemas to the LLM Client in OpenAI format."""
        definitions = []
        for name, (model, _) in self.tool_registry.items():
            definitions.append(_model_to_tool_schema(model, name=name))
        return definitions

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Dispatcher: Routes the tool name to the actual Python method.
        """
        logger.info("ACI Tools: Executing tool '%s' with args: %s", tool_name, arguments)

        if tool_name not in self.tool_registry:
            error_msg = f"ACI Tools: Tool '{tool_name}' not found"
            logger.warning(error_msg)
            return f"Error: {error_msg}"

        model_cls, method = self.tool_registry[tool_name]

        try:
            # Validate arguments using Pydantic
            validated_args = model_cls(**arguments)
            # Execute method with unpacked arguments
            result = method(**validated_args.model_dump())
            logger.debug("ACI Tools: Tool '%s' executed successfully", tool_name)
            return result
        except Exception as e:
            error_msg = f"ACI Tools: Error executing {tool_name}: {e!s}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    # --- Tool Implementations ---

    def _read_file(self, path: str, start_line: int = 1, end_line: int = 100) -> str:
        logger.debug("ACI Tools: Reading file '%s' lines %d-%d", path, start_line, end_line)
        try:
            content = self.sandbox.files.read(path)
            lines = content.splitlines()

            # Handle 1-based indexing
            start_idx = max(0, start_line - 1)
            end_idx = min(len(lines), end_line)

            selected_lines = lines[start_idx:end_idx]

            # Add line numbers for the LLM
            output = []
            for i, line in enumerate(selected_lines):
                output.append(f"{start_idx + i + 1}: {line}")

            if not output:
                logger.info("ACI Tools: File read returned empty content for '%s'", path)
                return "File is empty or range is invalid."

            logger.debug("ACI Tools: Successfully read %d lines from '%s'", len(output), path)
            return "\n".join(output)
        except Exception as e:
            error_msg = f"ACI Tools: Error reading file: {e}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    def _edit_file(self, path: str, start_line: int, end_line: int, new_content: str) -> str:
        logger.info("ACI Tools: Editing file '%s' from line %d to %d", path, start_line, end_line)
        try:
            # 1. Read original
            content = self.sandbox.files.read(path)
            lines = content.splitlines()

            # 2. Prepare Slicing (1-based to 0-based)
            start_idx = max(0, start_line - 1)
            end_idx = end_line

            # 3. Apply Edit in Memory
            new_lines_list = new_content.splitlines()

            # Safety Check: Are we extending the file?
            if start_idx > len(lines):
                error_msg = f"ACI Tools: Start line {start_line} is beyond end of file ({len(lines)} lines)"
                logger.warning(error_msg)
                return f"Error: {error_msg}"

            # Reconstruct content
            final_lines = lines[:start_idx] + new_lines_list + lines[end_idx:]
            final_content = "\n".join(final_lines)

            # 4. Write to Temp File
            temp_path = f"{path}.temp_lint"
            self.sandbox.files.write(temp_path, final_content)
            logger.debug("ACI Tools: Wrote temp file for linting: %s", temp_path)

            # 5. Auto-Lint (Syntax Check)
            lint_cmd = f"python3 -m py_compile {shlex.quote(temp_path)}"
            proc = self.sandbox.commands.run(lint_cmd)

            if proc.exit_code != 0:
                self.sandbox.commands.run(f"rm {temp_path}")
                error_msg = f"ACI Tools: Edit Rejected: Syntax Error in generated code.\n{proc.stderr}"
                logger.warning(error_msg)
                return f"❌ {error_msg}"

            # 6. Commit Change
            self.sandbox.commands.run(f"mv {temp_path} {path}")
            logger.info("ACI Tools: Successfully edited file '%s'", path)
            return "✅ Success: File edited and syntax verified."

        except Exception as e:
            error_msg = f"ACI Tools: Error editing file: {e}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    def _search_code(self, query: str, dir: str = ".") -> str:
        logger.debug("ACI Tools: Searching for '%s' in directory '%s'", query, dir)
        ignore_dirs = "{.git,.venv,venv,__pycache__,node_modules,.mypy_cache}"
        cmd = f"grep -rnH -I --exclude-dir={ignore_dirs} {shlex.quote(query)} {shlex.quote(dir)} | head -n 200"
        proc = self.sandbox.commands.run(cmd)
        if proc.exit_code != 0 and not proc.stdout:
            return "No matches found."
        return proc.stdout

    def _list_dir(self, path: str = ".") -> str:
        logger.debug("ACI Tools: Listing directory contents for '%s'", path)
        proc = self.sandbox.commands.run(f"ls -F {shlex.quote(path)}")
        if proc.exit_code != 0:
            return f"Error: {proc.stderr}"
        lines = proc.stdout.splitlines()
        filtered = [line for line in lines if not line.startswith("__") and not line.startswith(".git")]
        if len(filtered) > 50:
            return "\n".join(filtered[:50]) + "\n... (Output truncated)"
        return "\n".join(filtered)

    def _run_shell(self, command: str) -> str:
        logger.info("ACI Tools: Running shell command: %s", command)
        proc = self.sandbox.commands.run(command)
        result = f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}\nExit Code: {proc.exit_code}"
        return result


# --- AGENT LOGIC ---


class CodeAgent:
    """
    Autonomous Agent that uses LLMClient and CodeAgentTools to solve tasks.
    Implements the Think -> Act -> Observe loop.
    """

    def __init__(self, tools: CodeAgentTools) -> None:
        self.client = LLMClient()
        self.tools = tools
        self.system_prompt = (
            "You are an expert software engineer. "
            "You have access to a Linux environment and python tools. "
            "Always verify your code changes by running tests or scripts. "
            "Do not hallucinate file contents; read them first."
        )
        self.messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]

    def run(self, task: str, max_steps: int = 10):
        """Executes the agent loop for a given task."""
        self.messages.append({"role": "user", "content": task})

        step = 0
        while step < max_steps:
            step += 1
            yield {"type": "status", "content": f"Step {step}/{max_steps}: Thinking..."}

            # 1. Call LLM
            try:
                response = self.client.chat(
                    model=st.session_state.selected_model,
                    messages=self.messages,
                    tools=self.tools.get_definitions(),
                    tool_choice="auto",
                )
            except Exception as e:
                yield {"type": "error", "content": f"LLM Error: {e}"}
                break

            # Handle LiteLLM/OpenAI response format
            # Assuming response is a ModelResponse object or similar dict-like structure
            message = response.choices[0].message
            self.messages.append(message)  # Add assistant thought to history

            # 2. Check for Tool Calls
            if not message.tool_calls:
                # Agent finished or just talking
                yield {"type": "response", "content": message.content}
                if message.content and "TASK_COMPLETE" in message.content:  # Simple stop condition
                    break
                # If no tool calls and no explicit stop, we might prompt again or stop.
                # For this loop, we assume if it doesn't call a tool, it's waiting for user or done.
                break

            # 3. Execute Tools
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args_str = tool_call.function.arguments
                call_id = tool_call.id

                yield {"type": "tool_call", "name": func_name, "args": func_args_str}

                try:
                    args = json.loads(func_args_str)
                    result = self.tools.execute(func_name, args)
                except json.JSONDecodeError:
                    result = "Error: Invalid JSON arguments provided."

                yield {"type": "tool_result", "content": result}

                # 4. Feed Observation back to LLM
                self.messages.append({"role": "tool", "tool_call_id": call_id, "name": func_name, "content": str(result)})


# --- STREAMLIT INTERFACE ---


def main() -> None:
    st.set_page_config(page_title="Agent-in-a-Box", layout="wide")

    with st.sidebar:
        model_selector(key="code_agent")
        st.subheader("🔧 Sandbox Configuration")
        github_url = st.text_input("GitHub Repository URL")

        if github_url and "agent" not in st.session_state:
            try:
                sandbox = DockerSandbox(github_url=github_url)
                tools = CodeAgentTools(sandbox)
                client = LLMClient()
                st.session_state.agent = CodeAgent(client, tools)
                st.success("Agent Initialized")
            except Exception as e:
                st.error(f"Failed to initialize agent: {e}")
                logger.critical(f"Agent Initialization Failed: {e}")
        else:
            st.warning("Provide a valid GitHub URL to initialize the agent.")
            return

        debug_mode = st.toggle("Debug Mode", value=False)
        if st.button("Reset Agent"):
            st.session_state.pop("agent", None)
            st.session_state.messages = []
            st.rerun()

    with st.expander("Agent Messages", expanded=debug_mode):
        message_container = st.container()
        render_messages(message_container, client=client)

    with st._bottom():
        prompt = st.chat_input("Assign a task to the agent...")

    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run Agent
        with st.chat_message("assistant"):
            container = st.empty()
            full_response = ""

            # Stream agent steps
            agent = st.session_state.agent
            for event in agent.run(prompt):
                if event["type"] == "status":
                    with st.status(event["content"], expanded=True):
                        pass  # Just showing status
                elif event["type"] == "tool_call":
                    with st.expander(f"🛠️ Executing: {event['name']}"):
                        st.code(event["args"], language="json")
                elif event["type"] == "tool_result":
                    with st.expander("📄 Result", expanded=debug_mode):
                        st.code(event["content"])
                elif event["type"] == "response":
                    full_response = event["content"]
                    container.markdown(full_response)
                elif event["type"] == "error":
                    st.error(event["content"])


if __name__ == "__main__":
    main()
