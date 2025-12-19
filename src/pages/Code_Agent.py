from abc import ABC, abstractmethod
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any, Dict, Generic, List, Optional, TypeVar

import git
from pydantic import BaseModel, Field
import streamlit as st

from lib.streamlit_helper import model_selector


class AgentCommand(BaseModel, ABC):
    """Base command model for external agent processes.

    Input:
        executable: str
           - preinstalled CLI executable name
           - non-empty string
        workspace: Path
           - absolute or relative filesystem path
           - must exist before execution
        args: list[str]
           - ordered CLI arguments
           - no executable included
        env_vars: dict[str, str] | None
           - environment variable overrides
           - merged over os.environ

    Output:
        None

    Errors:
        ValidationError
           - invalid field types
    """

    executable: str = Field(..., description="CLI executable name")
    workspace: Path = Field(default_factory=Path.cwd, description="Agent workspace directory")
    args: List[str] = Field(default_factory=list, description="CLI arguments excluding executable")
    env_vars: Optional[Dict[str, str]] = Field(default=None, description="Environment variable overrides")

    @abstractmethod
    def construct_args(self) -> List[str]:
        """Construct full commandline arguments."""

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

    Errors:
        git.GitCommandError
           - clone or checkout failures
        OSError
           - filesystem permission issues

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

        Errors:
            git.GitCommandError
               - clone or checkout failures
            OSError
               - directory creation failures

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

        try:
            if workspace.exists() and (workspace / ".git").exists():
                repo = git.Repo(workspace)
                repo.remotes.origin.set_url(repo_url)
                repo.remotes.origin.fetch()
            else:
                repo = git.Repo.clone_from(repo_url, workspace)

            target_head = repo.heads[branch] if branch in repo.heads else repo.create_head(branch)
            target_head.checkout()

            try:
                repo.remotes.origin.pull(branch)
            except git.GitCommandError:
                st.warning(f"Git pull failed for branch '{branch}'. Using local state.")

            self._maybe_install_dependencies(workspace)
        except git.GitCommandError as exc:
            st.error(f"Git operation failed: {exc}")
            raise
        except OSError as exc:
            st.error(f"Workspace setup failed: {exc}")
            raise

        return workspace

    def _maybe_install_dependencies(self, workspace: Path) -> None:
        """Best-effort dependency installation; non-fatal on failure.

        Input:
            workspace: Path
               - repository root path
               - must exist

        Output:
            None

        Errors:
            None

        Side Effects:
            - may run pip install commands
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
               - fully constructed args
               - workspace directory set

        Output:
            process: subprocess.Popen[bytes]
               - running child process handle
               - stdout/stderr inherited by parent

        Errors:
            OSError
               - executable not found
               - spawn failure

        Side Effects:
            - starts external process
        """
        full_env: Dict[str, str] = dict(os.environ)
        if command.env_vars:
            full_env.update(command.env_vars)

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

        Output:
            None

        Errors:
            Implementation-defined

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

        Errors:
            None

        Side Effects:
            - renders Streamlit widgets
        """
        raise NotImplementedError

    def get_diff(self) -> str:
        """Return git diff for workspace."""
        try:
            repo = git.Repo(self.path_agent_workspace)
            diff_text: str = repo.git.diff()
            return diff_text
        except git.GitCommandError as exc:
            st.error(f"Failed to compute git diff: {exc}")
            return ""


class AiderCommand(AgentCommand):
    """Aider-specific command definition."""

    model_architect: str = Field(..., description="Architect LLM identifier")
    model_editor: str = Field(..., description="Editor LLM identifier")
    edit_format: str = Field(default="diff", description="Edit format: diff|whole|udiff")
    no_commit: bool = Field(default=True, description="Disable git commits")
    map_tokens: int = Field(default=4096, description="Context map tokens")
    extra_flags: List[str] = Field(default_factory=list, description="Raw aider flags")

    def construct_args(self) -> List[str]:
        """Construct aider argument list."""
        base: List[str] = list(self.args)
        base.extend(
            [
                "--architect-model",
                self.model_architect,
                "--editor-model",
                self.model_editor,
                "--edit-format",
                self.edit_format,
                "--map-tokens",
                str(self.map_tokens),
            ]
        )
        if self.no_commit:
            base.append("--no-commit")
        if self.extra_flags:
            base.extend(self.extra_flags)
        self.args = base
        return base


class AiderCodeAgent(CodeAgent[AiderCommand]):
    """Autonomous Aider Code Agent."""

    def ui_define_command(self) -> AiderCommand:
        """Define the Aider command with Streamlit UI."""
        command = AiderCommand(executable="aider", args=[], workspace=self.path_agent_workspace)

        st.markdown("## Select Architect Model")
        command.model_architect = model_selector(key="code_agent_architect")

        st.markdown("## Select Editor Model")
        command.model_editor = model_selector(key="code_agent_editor")

        st.markdown("---")
        st.markdown("## Edit Format")
        command.edit_format = st.selectbox(
            "Select Edit Format",
            options=["diff", "whole", "udiff"],
            index=0,
            key="aider_edit_format",
        )

        st.markdown("## Map Tokens")
        command.map_tokens = st.selectbox(
            "Context map tokens",
            options=[1024, 2048, 4096, 8192],
            index=2,
            key="aider_map_tokens",
        )

        st.markdown("## Flags")
        command.no_commit = st.checkbox(
            "Disable git commits (--no-commit)",
            value=True,
            key="aider_no_commit",
        )

        extra_flags_str: str = st.text_input(
            "Extra aider flags (advanced)",
            value="",
            help="Raw aider flags, space-separated; do not include models or edit-format here.",
            key="aider_extra_flags",
        )
        command.extra_flags = shlex.split(extra_flags_str) if extra_flags_str.strip() else []

        st.markdown("## Environment")
        ollama_base: str = st.text_input(
            "OLLAMA_API_BASE (optional)",
            value=os.environ.get("OLLAMA_API_BASE", ""),
            key="aider_ollama_api_base",
        )

        env_vars: Dict[str, str] = {}
        if ollama_base.strip():
            env_vars["OLLAMA_API_BASE"] = ollama_base.strip()
        command.env_vars = env_vars or None

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
            "Aider started. Interact with it in your terminal or configured UI. "
            "Return here after completion to view the git diff."
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

def main() -> None:
    """Streamlit entrypoint for multi-agent code workspace."""
    st.set_page_config(page_title="Agent-in-a-Box", layout="wide")

    with st.sidebar:
        st.markdown("## Repository")
        repo_url: str = st.text_input("GitHub Repository URL", key="repo_url")
        branch: str = st.text_input("Branch", value="main", key="branch")
        st.markdown("---")

        if not repo_url or not branch:
            st.warning("Please provide both Repository URL and Branch to proceed.")
            st.stop()

        st.markdown("## Agent")
        selected_agent_name: str = st.selectbox("Select Code Agent", options=agent_subclass_names, key="code_agent_selector")

        selected_agent: CodeAgent[Any] = get_agent(agent_type=selected_agent_name, repo_url=repo_url, branch=branch)
        command: AgentCommand = selected_agent.ui_define_command()

    with st._bottom:
        task: Optional[str] = st.chat_input("Assign a task to the agent...")

    if task:
        with st.chat_message("user"):
            st.markdown(task)

        with st.chat_message("assistant"):
            selected_agent.run(task=task, command=command)  # type: ignore[arg-type]
            diff: str = selected_agent.get_diff()
            if diff:
                st.markdown("### Git Diff")
                st.code(diff, language="diff")
            else:
                st.info("No changes detected in git diff.")


if __name__ == "__main__":
    main()
