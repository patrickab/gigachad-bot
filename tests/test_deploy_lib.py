"""deploy/lib.sh's process helpers against real processes."""

from pathlib import Path
import subprocess

from tests.shell import REPOSITORY


def _gone(pid: int) -> bool:
    stat = Path(f"/proc/{pid}/stat")
    try:
        return stat.read_text().rsplit(")", 1)[1].split()[0] == "Z"
    except FileNotFoundError:
        return True


def test_stop_tree_kills_a_tree_ignoring_term_within_one_grace_period():
    script = f"""
source {REPOSITORY / "deploy/lib.sh"}
bash -c 'trap "" TERM; sleep 60 & wait' >/dev/null 2>&1 &
parent=$!
until child=$(pgrep -P "$parent"); do sleep 0.05; done
echo "$parent $child"
start=$EPOCHREALTIME
stop_tree 2 "$parent"
echo "$start $EPOCHREALTIME"
"""
    done = subprocess.run(["bash", "-c", script], text=True, capture_output=True, timeout=30, check=True)
    pids, times = done.stdout.splitlines()
    parent, child = map(int, pids.split())
    start, end = map(float, times.split())

    assert 1.5 < end - start < 3.5
    assert _gone(parent) and _gone(child)
    assert str(child) in done.stderr
