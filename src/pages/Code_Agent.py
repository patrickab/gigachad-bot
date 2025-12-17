import json
import shlex
from typing import Any, Dict, Generator, List

from llm_baseclient.client import LLMClient
from pydantic import BaseModel, Field
import streamlit as st

from lib.agents.docker_sandbox import DockerSandbox
from lib.streamlit_helper import model_selector
from lib.utils.logger import get_logger

logger = get_logger()


class AgentTool(BaseModel):
    """Base class unifiying schema, execution, and error handling."""

    @classmethod
    def definition(cls) -> Dict[str, Any]:
        schema = cls.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": cls.__name__,
                "description": (cls.__doc__ or "").strip(),
                "parameters": {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                },
            },
        }

    def run(self, sandbox: DockerSandbox) -> str:
        raise NotImplementedError

    def safe_run(self, sandbox: DockerSandbox) -> str:
        """Executes run() with centralized error handling and logging."""
        try:
            return self.run(sandbox)
        except Exception as e:
            logger.error(f"Tool Error ({self.__class__.__name__}): {e}", exc_info=True)
            return f"Error executing {self.__class__.__name__}: {e}"


# --- CONCRETE TOOLS (Logic) ---


class ReadFile(AgentTool):
    """Reads a file section with line numbers. Use before editing."""

    path: str = Field(..., description="Relative path to file")
    start_line: int = Field(1, description="Start line (1-based)")
    end_line: int = Field(100, description="End line (1-based)")

    def run(self, sandbox: DockerSandbox) -> str:
        content = sandbox.files.read(self.path)
        lines = content.splitlines()
        start = max(0, self.start_line - 1)
        end = min(len(lines), self.end_line)

        output = [f"{start + i + 1}: {line}" for i, line in enumerate(lines[start:end])]
        return "\n".join(output) if output else "File is empty or range is invalid."


class EditFile(AgentTool):
    """Replaces lines in a file. Auto-lints before saving."""

    path: str = Field(..., description="Relative path")
    start_line: int = Field(..., description="Start line (1-based)")
    end_line: int = Field(..., description="End line (1-based, inclusive)")
    new_content: str = Field(..., description="New content")

    def run(self, sandbox: DockerSandbox) -> str:
        lines = sandbox.files.read(self.path).splitlines()
        start_idx = max(0, self.start_line - 1)

        if start_idx > len(lines):
            return f"Error: Start line {self.start_line} > file length ({len(lines)})"

        final_lines = lines[:start_idx] + self.new_content.splitlines() + lines[self.end_line :]
        temp_path = f"{self.path}.temp_lint"
        sandbox.files.write(temp_path, "\n".join(final_lines))

        # Verify syntax using the container's python
        cmd = f"python3 -m py_compile {shlex.quote(temp_path)}"
        if (proc := sandbox.commands.run(cmd)).exit_code != 0:
            sandbox.commands.run(f"rm {temp_path}")
            return f"❌ Edit Rejected: Syntax Error.\n{proc.stderr}"

        sandbox.commands.run(f"mv {temp_path} {self.path}")
        return "✅ Success: File edited and syntax verified."


class RunShell(AgentTool):
    """Executes a shell command."""

    command: str = Field(..., description="Bash command")

    def run(self, sandbox: DockerSandbox) -> str:
        proc = sandbox.commands.run(self.command)
        return f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}\nExit Code: {proc.exit_code}"


class SearchCode(AgentTool):
    """Searches for a string pattern in the codebase."""

    query: str = Field(..., description="The string to search for")
    dir: str = Field(".", description="The directory to search in")

    def run(self, sandbox: DockerSandbox) -> str:
        ignore = "{.git,.venv,venv,__pycache__,node_modules,.mypy_cache}"
        cmd = f"grep -rnH -I --exclude-dir={ignore} {shlex.quote(self.query)} {shlex.quote(self.dir)} | head -n 200"
        proc = sandbox.commands.run(cmd)
        return proc.stdout or "No matches found."


class ListDir(AgentTool):
    """Lists files in a directory."""

    path: str = Field(".", description="The directory path to list")

    def run(self, sandbox: DockerSandbox) -> str:
        proc = sandbox.commands.run(f"ls -F {shlex.quote(self.path)}")
        if proc.exit_code != 0:
            return f"Error: {proc.stderr}"

        lines = [line for line in proc.stdout.splitlines() if not line.startswith((".", "__"))]
        return "\n".join(lines[:50]) + ("\n... (Truncated)" if len(lines) > 50 else "")


class CodeAgentTools:
    """Runtime ACI: Dispatches Pydantic tools to the DockerSandbox."""

    def __init__(self, sandbox: DockerSandbox) -> None:
        self.sandbox = sandbox
        # Auto-discover tools inheriting from AgentTool
        self.registry = {cls.__name__: cls for cls in AgentTool.__subclasses__()}
        logger.debug("ACI Tools: Initialized with %d tools", len(self.registry))

    def get_definitions(self) -> List[Dict[str, Any]]:
        return [t.definition() for t in self.registry.values()]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_cls := self.registry.get(tool_name):
            return tool_cls(**arguments).safe_run(self.sandbox)
        return f"Error: Tool '{tool_name}' not found"


class CodeAgent:
    """Autonomous Agent implementing Think -> Act -> Observe loop."""

    def __init__(self, repo_url: str, branch: str = "main") -> None:
        self.client = LLMClient()
        self.sandbox = DockerSandbox(repo_url=repo_url, branch=branch)
        self.tools = CodeAgentTools(self.sandbox)
        self.tool_definitions = self.tools.get_definitions()

        tool_list = ", ".join(self.tools.registry.keys())
        self.system_prompt = (
            "You are an expert software engineer working in a sandboxed environment.\n"
            f"TOOLS: You have access to: {tool_list}.\n"
            "PROTOCOL:\n"
            "1. EXPLORE: Always ListDir and ReadFile before editing.\n"
            "2. VERIFY: Always create a reproduction script or test case before fixing.\n"
            "3. EDIT: Use EditFile with precise line numbers.\n"
            "4. TEST: confirm fix with tests.\n"
            "5. DONE: Output 'TASK_COMPLETE'."
        )

        if not self.client.messages:
            self.client.messages.append({"role": "system", "content": self.system_prompt})

    def run(self, task: str, model: str, max_steps: int = 15) -> Generator[Dict[str, Any], None, None]:
        """Executes the agent loop, yielding UI events."""
        response = self.client.chat(model=model, user_msg=task, tools=self.tool_definitions, stream=False)

        for step in range(1, max_steps + 1):
            message = response.choices[0].message

            if message.tool_calls:
                yield {"type": "status", "content": f"Step {step}: Agent is using tools..."}

                for tool_call in message.tool_calls:
                    yield {"type": "tool_call", "name": tool_call.function.name, "args": tool_call.function.arguments}

                    try:
                        args = json.loads(tool_call.function.arguments)
                        result = self.tools.execute(tool_call.function.name, args)
                    except json.JSONDecodeError:
                        result = "Error: Invalid JSON arguments."
                        yield {"type": "error", "content": result}

                    yield {"type": "tool_result", "content": result}
                    self.client.add_tool_result(tool_call_id=tool_call.id, output=result)

                response = self.client.chat(model=model, user_msg=None, tools=self.tool_definitions, stream=False)
            else:
                yield {"type": "response", "content": message.content}
                if message.content and "TASK_COMPLETE" in message.content:
                    break
                break


def main() -> None:
    st.set_page_config(page_title="Agent-in-a-Box", layout="wide")

    with st.sidebar:
        model_selector(key="code_agent")
        st.subheader("🔧 Sandbox Configuration")
        repo_url = st.text_input("GitHub Repository URL")

        if not repo_url:
            st.warning("Provide a valid GitHub URL to initialize the agent.")
            return

        if repo_url and "code_agent" not in st.session_state:
            st.session_state.code_agent = CodeAgent(repo_url=repo_url, branch="main")
            st.success("Agent Initialized")
            st.rerun()

        debug_mode = st.toggle("Debug Mode", value=False)
        if st.button("Reset Agent"):
            st.session_state.pop("code_agent", None)
            st.rerun()

    with st._bottom:
        prompt = st.chat_input("Assign a task to the agent...")

    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            container = st.container()
            status_container = container.empty()
            response_placeholder = container.empty()

            code_agent: CodeAgent = st.session_state.code_agent

            try:
                for event in code_agent.run(task=prompt, model=st.session_state.selected_model):
                    if event["type"] == "status":
                        status_container.status(event["content"], state="running")
                    elif event["type"] == "tool_call":
                        with container.expander(f"🛠️ Executing: {event['name']}"):
                            st.code(event["args"], language="json")
                    elif event["type"] == "tool_result":
                        with container.expander("📄 Result", expanded=debug_mode):
                            st.code(event["content"])
                    elif event["type"] == "response":
                        status_container.empty()
                        response_placeholder.markdown(event["content"])
                    elif event["type"] == "error":
                        container.error(event["content"])
            except Exception as e:
                st.error(f"Runtime Error: {e}")


if __name__ == "__main__":
    main()
