from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Generic, List, Optional, TypeVar

import streamlit as st
import os

from lib.streamlit_helper import model_selector
from pydantic import BaseModel
from abc import abstractmethod, ABC

class AgentCommand(BaseModel, ABC):
    executable: str # commandline executable (assume preinstalled)
    workspace: Path = Path.cwd()  # path to agent workspace (automatically setup by agent) / default current working directory
    args: List[str] = [] # optional list of commandline arguments
    env_vars: Optional[dict[str, str]] = None  # optional environment variables

    @abstractmethod
    def construct_args(self) -> List[str]:
        """Construct the full commandline arguments for the agent command."""
        ...

TCommand = TypeVar("TCommand", bound=AgentCommand)


class CodeAgent(ABC, Generic[TCommand]):
    """Generic base class for Code Agents."""
    def __init__(self, repo_url: str, branch: str) -> None:
        self.repo_url = repo_url
        self.branch = branch
        self.path_agent_workspace = self._setup_workspace(repo_url, branch)

    def _setup_workspace(self, repo_url: str, branch: str) -> Path:
        """Setup agent workspace - clone, branch, etc."""
        # TODO: Implement workspace setup using gitpython library. For now undockerized repository clone to operate on.
        # Clone to ~/agent_sandbox/<repo_name> using appropriate git clone command.
        # Handle both HTTPS and SSH URLs. Create/switch to branch as needed. User must be able to git push from agent workspace if SSH selected.
        # Try dependency installation if requirements.txt or pyproject.toml found. If fails, pass silently & let agent handle it.
        return Path.home() / "agent_sandbox" / <repo_name>  # Placeholder repo name must be inferred from git repository

    def _execute_agent_command(self, command: TCommand) -> None:
        """Execute the agent command in its workspace using subprocess."""
        # TODO: Use st.spinner direct the user to commandline or agent web UI instead of streaming output in Streamlit
        # blocks UI until command completes.
        subprocess.run(
            [command.executable, *command.args],
            capture_output=False,
            cwd=command.workspace,
            env={**os.environ, **(command.env_vars or {})}, # TODO: ensure that this is correct syntax
        )

    @abstractmethod
    def run(self, task: str, command: TCommand) -> None:
        """Combines task with command according to agent-specific syntax."""
        ...

    @abstractmethod
    def ui_define_command(self) -> TCommand:
        """Define the agent command with Streamlit UI."""
        ...

    def get_diff(self) -> str:
        # TODO: get git diff from the agent workspace & return as string
        return "Diff placeholder"


class AiderCommand(AgentCommand):
    """Aider-specific command definition."""
    command_executable: str = "aider"

    model_architect: str
    model_editor: str
    edit_format: str = "diff" | "whole" | "udiff"
    no_commit: bool = True

    def construct_args(self) -> List[str]:
        """
        Construct the full commandline arguments for aider.
        User-friendly way to set common flags.
        """
        args = [
            "--architect-model", self.model_architect,
            "--editor-model", self.model_editor,
            "--edit-format", self.edit_format,
        ]
        if self.no_commit:
            args.append("--no-commit")

        self.args = self.args + args


class AiderCodeAgent(CodeAgent[AiderCommand]):
    """Autonomous Aider Code Agent."""

    def ui_define_command(self) -> AiderCommand:
        """Define the Aider command with Streamlit UI."""
        command = AiderCommand(
            executable="aider",
            args=[],
            workspace=self.path_agent_workspace,
        )
        st.markdown("## Select Architect Model")
        command.model_architect = model_selector(key="code_agent_architect")
        st.markdown("## Select Editor Model")
        command.model_editor = model_selector(key="code_agent_editor")
        st.markdown("---")
        st.markdown("## Select Edit Format")
        command.edit_format = st.selectbox(
            "Select Edit Format",
            options=["diff", "whole", "udiff"],
        )

        st.markdown("## Select Flags for Aider Command")
        # TODO: Create user-friendly Streamlit controls for common aider flags:
        # - Selectbox for (map-tokens: 1024-8192 in multiples of powers of 2)
        # TODO: use st.multiselect for --no-commit, --browser, --no-stream, --edit-format diff
        # TODO: think about +5 most relevant flags for aider command & add them to multiselect

        return command

    def run(self, task: str, command: AiderCommand) -> None:
        """Executes the agent loop, yielding UI events."""
        # TODO: add task to command accordig to aider syntax
        # TODO: Prepare environment variables (e.g. OLLAMA_API_BASE)
        # Pass them to _execute_agent_command via command.env_vars
        command.args.extend(["--message", task])
        command.construct_args()
        self._execute_agent_command(command)



# Get subclasses and their names
agent_subclasses = CodeAgent.__subclasses__()

# Map name -> class
agent_subclass_dict = {cls.__name__: cls for cls in agent_subclasses}
agent_subclass_names = list(agent_subclass_dict.keys())

@st.cache_resource
def get_agent(agent_type: str, repo_url: str, branch: str):
    """Getter method to avoid re-instantiation on every interaction."""
    agent_cls: type[CodeAgent] = agent_subclass_dict[agent_type]
    return agent_cls(repo_url, branch)

def main() -> None:
    st.set_page_config(page_title="Agent-in-a-Box", layout="wide")

    with st.sidebar:

        st.markdown("## Select Repository to work on")
        st.session_state.repo_url = st.text_input("GitHub Repository URL")
        st.session_state.branch = st.text_input("Branch", value="main")
        st.markdown("---")
        if not st.session_state.repo_url or not st.session_state.branch:
            st.warning("Please provide both Repository URL and Branch to proceed.")
            st.stop()

        # Store selected class name
        st.session_state.selected_agent = st.selectbox(
            "Select Code Agent",
            options=agent_subclass_names,
            key="code_agent_selector"
        )

        # Retrieve and instantiate the actual class
        selected_agent = get_agent(
            agent_type=st.session_state.selected_agent,
            repo_url=st.session_state.repo_url,
            branch=st.session_state.branch,
        )
        command = selected_agent.ui_define_command()

    with st._bottom:
        task = st.chat_input("Assign a task to the agent...")

    if task:
        with st.chat_message("user"):
            st.markdown(task)

        with st.chat_message("assistant"):
            selected_agent.run(task=task, command=command)
            diff = selected_agent.get_diff()
            st.markdown(diff, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
