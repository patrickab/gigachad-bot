from abc import ABC, abstractmethod
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar

import git
from pydantic import BaseModel, Field
import streamlit as st

from lib.streamlit_helper import model_selector


class AgentCommand(BaseModel, ABC):
    """Base command model for external agent processes.

    Input:
        executable: str
           - preinstalled CLI executable name
        workspace: Path
           - absolute or relative filesystem path
           - must exist before execution
        args: list[str]
           - ordered CLI arguments
           - no executable included
        env_vars: dict[str, str] | None
           - environment variable overrides
           - merged over os.environ
    """

    executable: str = Field(..., description="CLI executable name")
    workspace: Path = Field(default_factory=Path.cwd, description="Agent workspace directory")
    args: List[str] = Field(default_factory=list, description="CLI arguments excluding executable")
    env_vars: Optional[Dict[str, str]] = Field(default=None, description="Environment variable overrides")

    def _snake_to_kebab(self, s: str) -> str:
        """Convert snake_case to kebab-case."""
        return s.replace("_", "-")

    def construct_args(self) -> List[str]:
        """Construct aider argument list."""
        # This model dump will include all class values of the subclass
        # The fields is a dict of all class attributes intended for CLI
        fields = self.model_dump(exclude={"executable", "workspace","args", "env_vars"})
        self.args = [
            item for k, v in fields.items() for item in ([f"--{self._snake_to_kebab(k)}"] + ([] if isinstance(v, bool) else [str(v)]))
        ]


TCommand = TypeVar("TCommand", bound=AgentCommand)


class CodeAgent(ABC, Generic[TCommand]):
    """Generic base class for Code Agents.

    Input:
        repo_url: str
           - HTTPS or SSH git URL
           - points to accessible repository
        branch: str
           - existing or new branch name
           - non-empty string

    Output:
        instance: CodeAgent
           - initialized workspace path
           - ready for command execution

    Side Effects:
        - creates ~/agent_sandbox directory
        - clones or updates git repository
        - may install Python dependencies (if requirements.txt or pyproject.toml present)
    """

    def __init__(self, repo_url: str, branch: str) -> None:
        self.repo_url: str = repo_url
        self.branch: str = branch
        self.path_agent_workspace: Path = self._setup_workspace(repo_url, branch)

    def _setup_workspace(self, repo_url: str, branch: str) -> Path:
        """Setup agent workspace: clone, checkout branch, install deps.

        Input:
            repo_url: str
               - HTTPS or SSH git URL
               - parseable by gitpython
            branch: str
               - target branch name
               - used for checkout or creation

        Output:
            workspace_path: Path
               - absolute path to repo root
               - guaranteed to exist on success

        Side Effects:
            - creates ~/agent_sandbox/<repo_name>
            - runs optional dependency installation
        """
        sandbox_root: Path = Path.home() / "agent_sandbox"
        sandbox_root.mkdir(parents=True, exist_ok=True)

        repo_name: str = repo_url.rstrip("/").split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]

        workspace: Path = sandbox_root / repo_name

        if workspace.exists() and (workspace / ".git").exists():
            subprocess.run(
                ["git", "-C", str(workspace), "pull"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            subprocess.run(
                ["git", "clone", "--branch", branch, repo_url, str(workspace)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

        self._try_install_dependencies(workspace)
        return workspace

    def _try_install_dependencies(self, workspace: Path) -> None:
        """
        Best-effort dependency installation; non-fatal on failure.
        Assumes `uv` installed on system PATH.

        Input:
            workspace: Path
               - repository root path
               - must exist
        """
        requirements: Path = workspace / "requirements.txt"
        pyproject: Path = workspace / "pyproject.toml"

        cmd: Optional[List[str]] = None
        if requirements.is_file():
            cmd = ["uv", "pip", "install", "-r", str(requirements)]
        elif pyproject.is_file():
            cmd = ["uv", "pip", "install", "-e", str(workspace)]

        if not cmd:
            return

        try:
            subprocess.run(
                cmd,
                cwd=workspace,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            # ignore installation failures; agent expected to handle env issues
            return

    def _execute_agent_command(self, command: TCommand) -> subprocess.Popen[bytes]:
        """Execute the agent command in its workspace using subprocess.

        Input:
            command: TCommand
               - fully configured agent command

        Output:
            process: subprocess.Popen[bytes]
               - running child process handle
               - stdout/stderr inherited by parent

        Side Effects:
            - starts external process
        """
        full_env: Dict[str, str] = dict(os.environ)
        # NOTE: disable for now
        # if command.env_vars:
        #    full_env.update(command.env_vars)

        full_cmd: List[str] = [command.executable, *command.construct_args()]

        try:
            process = subprocess.Popen(full_cmd, cwd=str(command.workspace), env=full_env)
        except OSError as exc:
            st.error(f"Failed to start agent process: {exc}")
            raise

        return process

    @abstractmethod
    def run(self, task: str, command: TCommand) -> None:
        """Combines task with command according to agent-specific syntax.

        Input:
            task: str
               - natural language instruction
               - non-empty string
            command: TCommand
               - agent-specific command model
               - mutated with task context

        Side Effects:
            - may start external processes
        """
        raise NotImplementedError

    @abstractmethod
    def ui_define_command(self) -> TCommand:
        """Define the agent command with Streamlit UI.

        Input:
            None

        Output:
            command: TCommand
               - fully configured command model
               - ready for execution
        """
        raise NotImplementedError

    def get_diff(self) -> str:
        """Return git diff for workspace."""
        result = subprocess.run(
            ["git", "-C", str(self.path_agent_workspace), "diff"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout


class AiderCommand(AgentCommand):
    """Aider-specific command definition."""

    model: str = Field(..., description="Architect LLM identifier")
    editor_model: str = Field(..., description="Editor LLM identifier")
    reasoning_effort: Literal["low", "medium", "high"] = Field(default="high", description="Reasoning effort")
    edit_format: Literal["diff", "whole", "udiff"] = Field(default="diff", description="Edit format")
    no_commit: bool = Field(default=True, description="Disable git commits")
    map_tokens: Literal[1024, 2048, 4096, 8192] = Field(default=4096, description="Context map tokens")

class Aider(CodeAgent[AiderCommand]):
    """Aider Code Agent."""

    def ui_define_command(self) -> AiderCommand:
        """Define the Aider command with Streamlit UI."""
        st.markdown("# Model Control")
        with st.expander("", expanded=True):
            reasoning_effort = st.selectbox("Reasoning effort", ["low", "medium", "high"], index=2, key="aider_reasoning_effort")
            model_architect = model_selector(key="code_agent_architect")
            model_editor = model_selector(key="code_agent_editor")

        st.markdown("---")
        st.markdown("# Advanced Command Control")
        with st.expander("", expanded=False):
            edit_format = st.selectbox("Select Edit Format", ["diff", "whole", "udiff"], index=0, key="aider_edit_format")
            map_tokens = st.selectbox("Context map tokens", [1024, 2048, 4096, 8192], index=0, key="aider_map_tokens")
            no_commit = st.toggle("Disable git commits (--no-commit)", value=True, key="aider_no_commit")

            extra_flags_str = st.text_input(
                "Extra aider flags (advanced)",
                value="",
                help="Raw aider flags, space-separated; do not include models or edit-format here.",
                key="aider_extra_flags",
            )
            extra_flags = shlex.split(extra_flags_str) if extra_flags_str.strip() else []

        command = AiderCommand(
            executable="aider",
            workspace=self.path_agent_workspace,
            args=extra_flags,
            model=model_architect,
            editor_model=model_editor,
            reasoning_effort=reasoning_effort,
            edit_format=edit_format,
            map_tokens=map_tokens,
            no_commit=no_commit,
        )

        return command

    def run(self, task: str, command: AiderCommand) -> None:
        """Executes the aider agent with task and UI feedback."""
        if not task.strip():
            st.warning("Task is empty; nothing to run.")
            return

        command.args.extend(["--message", task])

        with st.spinner("Starting aider in terminal or browser; interact there."):
            process = self._execute_agent_command(command)

        st.info(
            "Aider started. Interact with it in your terminal or configured UI. Return here after completion to view the git diff."
        )
        st.caption(f"Spawned process PID: {process.pid}")


# Agent registry: dynamic discovery for extensible multi-agent support
agent_subclasses = CodeAgent.__subclasses__()
agent_subclass_dict = {cls.__name__: cls for cls in agent_subclasses}
agent_subclass_names = list(agent_subclass_dict.keys())


@st.cache_resource
def get_agent(agent_type: str, repo_url: str, branch: str) -> CodeAgent[Any]:
    """Instantiate and cache agent instance."""
    agent_cls: type[CodeAgent[Any]] = agent_subclass_dict[agent_type]
    return agent_cls(repo_url, branch)

def get_remote_branches(repo_url: str) -> list[str]:
    """Extract branch names from remote repository."""
    return [
        ref.split("\t")[1].replace("refs/heads/", "")
        for ref in git.Git().ls_remote("--heads", repo_url).splitlines()
    ]

def main() -> None:
    """Streamlit entrypoint for multi-agent code workspace."""
    st.set_page_config(page_title="Agent-in-a-Box", layout="wide")

    with st.sidebar:
        if "selected_agent" not in st.session_state:
            st.markdown("## Agent Controls")
            selected_agent_name: str = st.selectbox("Select Code Agent", options=agent_subclass_names, key="code_agent_selector")
            repo_url: str = st.text_input("GitHub Repository URL", value="https://github.com/patrickab/gigachad-bot", key="repo_url")
            if repo_url:
                if "branches" not in st.session_state:
                    st.session_state.branches = get_remote_branches(repo_url)

                branch = st.selectbox("Select Branch", options=st.session_state.branches, index=0, key="branch_selector")

            def init_agent() -> None:
                """Initialize and store agent in session state."""
                selected_agent: CodeAgent[Any] = get_agent(agent_type=selected_agent_name, repo_url=repo_url, branch=branch)
                st.session_state.selected_agent = selected_agent

            if repo_url and branch:
                st.button("Initialize Agent", key="init_agent_button", on_click=init_agent)

        else:
            # Agent in session state - configure command
            selected_agent: CodeAgent[Any] = st.session_state.selected_agent
            command: AgentCommand = selected_agent.ui_define_command()
            if st.button("Clear Agent", key="clear_agent_button"):
                del st.session_state.selected_agent

    with st._bottom:
        task: Optional[str] = st.chat_input("Assign a task to the agent...")

    if task:
        with st.chat_message("user"):
            st.markdown(task)

        with st.chat_message("assistant"):
            selected_agent.run(task=task, command=command)
            diff: str = selected_agent.get_diff()
            if diff:
                st.markdown("### Git Diff")
                st.code(diff, language="diff")
            else:
                st.info("No changes detected in git diff.")


if __name__ == "__main__":
    main()
