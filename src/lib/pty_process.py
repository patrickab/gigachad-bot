"""A child process on a pseudo-terminal: bytes in and out, resizes, and a hangup on close."""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import os
from pathlib import Path
import signal
import struct
import subprocess
import termios

_READ_CHUNK = 65536
_SHUTDOWN_SECONDS = 2


def set_terminal_size(fd: int, rows: int, cols: int) -> None:
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))


def claim_controlling_terminal() -> None:
    """Make the PTY the child's controlling terminal so resizes deliver SIGWINCH."""
    fcntl.ioctl(0, termios.TIOCSCTTY, 0)


def terminal_environment() -> dict[str, str]:
    """The backend's environment for a terminal program, true color on."""
    environment = {**os.environ, "TERM": "xterm-256color", "COLORTERM": "truecolor"}
    environment.pop("NVIM", None)  # set when the backend itself runs inside a Neovim terminal
    return environment


class PtyProcess:
    """One process in its own session on a PTY.

    Examples
    --------
    >>> shell = PtyProcess.spawn(["bash", "-l"], cwd=Path.home(), rows=40, cols=120)
    >>> shell.write_input(b"ls\\r")
    >>> while (output := await shell.read_output()) is not None: ...
    >>> await shell.close()
    """

    def __init__(self, process: subprocess.Popen, terminal_fd: int) -> None:
        self._process = process
        self._terminal_fd = terminal_fd
        self._output: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._loop = asyncio.get_running_loop()
        self._loop.add_reader(terminal_fd, self._read_terminal)

    @classmethod
    def spawn(cls, argv: list[str], *, cwd: Path, rows: int, cols: int) -> PtyProcess:
        """Start *argv* in *cwd* on a new PTY of *rows* x *cols*. Call from a running event loop."""
        terminal_fd, child_fd = os.openpty()
        set_terminal_size(terminal_fd, rows, cols)
        try:
            process = subprocess.Popen(
                argv,
                stdin=child_fd,
                stdout=child_fd,
                stderr=child_fd,
                cwd=cwd,
                env=terminal_environment(),
                start_new_session=True,
                preexec_fn=claim_controlling_terminal,
            )
        except BaseException:
            os.close(terminal_fd)
            raise
        finally:
            os.close(child_fd)
        return cls(process, terminal_fd)

    async def read_output(self) -> bytes | None:
        """Next output chunk, None once the terminal has closed."""
        return await self._output.get()

    def write_input(self, data: bytes) -> None:
        """Forward keystrokes, dropped once the terminal has closed."""
        view = memoryview(data)
        while view and self._terminal_fd >= 0:
            view = view[os.write(self._terminal_fd, view) :]

    def resize(self, rows: int, cols: int) -> None:
        if self._terminal_fd >= 0:
            set_terminal_size(self._terminal_fd, rows, cols)

    async def wait(self) -> int:
        return await asyncio.to_thread(self._process.wait)

    async def close(self) -> None:
        """Hang up the process group (an interactive shell ignores SIGTERM), kill it when it outlives the grace period."""
        if self._process.poll() is None:
            self._signal_group(signal.SIGHUP)
            try:
                await asyncio.to_thread(self._process.wait, _SHUTDOWN_SECONDS)
            except subprocess.TimeoutExpired:
                self._signal_group(signal.SIGKILL)
                await asyncio.to_thread(self._process.wait)
        self._stop_reading()

    def _signal_group(self, signal_number: signal.Signals) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self._process.pid, signal_number)

    def _read_terminal(self) -> None:
        try:
            data = os.read(self._terminal_fd, _READ_CHUNK)
        except OSError:  # EIO once the child side of the PTY is closed
            data = b""
        if data:
            self._output.put_nowait(data)
        else:
            self._stop_reading()

    def _stop_reading(self) -> None:
        if self._terminal_fd < 0:
            return
        self._loop.remove_reader(self._terminal_fd)
        os.close(self._terminal_fd)
        self._terminal_fd = -1
        self._output.put_nowait(None)
