"""Exercise run-learnharness.sh against local Git repositories and fake npm/node/next/systemd: checkout, overlay, rebuilds,
data snapshots, the sandboxed unit and its environment."""

from http.server import BaseHTTPRequestHandler, HTTPServer
import os
from pathlib import Path
import re
import shutil
import socket
import stat
import subprocess
import threading

from tests.conftest import git
from tests.shell import REPOSITORY, environment_with, fake_command, run_script

REF = "v0.4.0"
SHELL_ORIGINS = "http://127.0.0.1:2999 http://localhost:2999"
# npm records each command with the provider key it saw, `ci` installs the fake next and tsc, `run` (the build) writes
# BUILD_ID first like Next and fails while FAIL_BUILD exists or the upstream commit holds build-broken. node packs a chat-ui
# whose content is CHAT_UI_CONTENT.
FAKE_NPM = """echo "$1 openai=${{OPENAI_API_KEY:-none}}" >>"{log}"
mkdir -p node_modules/.bin
if [[ $1 == ci ]]; then
    cp "{next}" node_modules/.bin/next
    cp "{tsc}" node_modules/.bin/tsc
    touch node_modules/.package-lock.json
fi
if [[ $1 == run ]]; then
    mkdir -p .next && touch .next/BUILD_ID
    [[ ! -e "{fail}" && ! -e build-broken ]]
fi
"""
# The overlay checks record where they ran and fail on an upstream commit holding types-broken or tests-broken.
FAKE_TSC = """#!/usr/bin/env bash
echo "tsc $PWD" >>"{log}"
[[ ! -e types-broken ]]
"""
FAKE_NODE_TEST = """if [[ $1 == --test ]]; then
    echo "test $PWD $*" >>"{log}"
    [[ ! -e tests-broken ]]
fi
"""
# Records its environment and arguments in .next (outside the rebuild check), then listens on its port for a moment.
FAKE_NEXT = """#!/usr/bin/env bash
env >"$PWD/.next/served-env.txt"
echo "$*" >"$PWD/.next/served-args.txt"
exec python3 -c 'import socket, sys, time; server = socket.create_server(("127.0.0.1", int(sys.argv[1]))); time.sleep(0.6)' \\
    "${@: -1}"
"""
FAKE_NODE = """if [[ $1 == packages/chat-ui/pack.mjs ]]; then
    mkdir -p "$3" && echo "${CHAT_UI_CONTENT:-}" >"$3/chat-ui.tgz" && echo "$3/chat-ui.tgz"
fi
"""
# systemd-run records its arguments and runs the command in the working directory, systemctl records its calls and reports
# every unit inactive unless UNIT_ACTIVE is set.
FAKE_SYSTEMD_RUN = """printf '%s\\n' "$@" >"{log}"
while [[ $1 != -- ]]; do
    if [[ $1 == --working-directory=* ]]; then cd -- "${{1#*=}}"; fi
    shift
done
shift
exec "$@"
"""
FAKE_SYSTEMCTL = 'echo "$*" >>"{log}"\nif [[ $* == *is-active* ]]; then [[ -n ${{UNIT_ACTIVE:-}} ]]; fi\n'
FAKE_FLATPAK = 'printf \'%s\\n\' "FLATPAK_USER_DIR=$FLATPAK_USER_DIR" "$@"\n'
ALLOWED_ENVIRONMENT = {
    "PATH", "HOME", "LANG", "NODE_ENV", "LH_DATA_DIR", "LH_AUTH_FILE", "TMPDIR", "CHROME_PATH", "BRAVE_API_KEY",
    "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS", "CUSTOM_OPENAI_API_KEY", "GIGACHAD_LEARNHARNESS_FRAME_ANCESTORS",
}
# The project venv's python records each generator call and answers `master-key` with the proxy's key.
FAKE_PYTHON = """echo "$* home=$GIGACHAD_LEARNHARNESS_HOME proxy=$GIGACHAD_LITELLM_PROXY_HOME src=$PYTHONPATH" >>"$WRITE_CONFIG_LOG"
[[ $* != *master-key* ]] || cat "$GIGACHAD_LITELLM_PROXY_HOME/master.key"
"""
MASTER_KEY = "sk-master"
# What the session model freeze wraps in upstream's components/model-picker.tsx.
MODEL_PICKER = """export function ModelControl({ chatId }: { chatId?: string }) {
  window.addEventListener("lh:kbd-open-model-picker", open);
  window.addEventListener("lh:kbd-cycle-thinking", cycle);
}
"""
# The launcher's `upstream_markers` (file|text): what it requires of an upstream ref.
LAUNCHER_MARKERS = re.findall(r"^    '([^'|]+)\|([^']+)'$", (REPOSITORY / "run-learnharness.sh").read_text(), re.MULTILINE)


def upstream_files() -> dict[str, str]:
    """An upstream with every marker the launcher requires, the freeze's events included."""
    files = {"components/model-picker.tsx": MODEL_PICKER}
    for path, marker in LAUNCHER_MARKERS:
        files[path] = files.get(path, "") + marker + "\n"
    return files


def seams_preimages(patch: str) -> dict[str, str]:
    """Upstream files the seams patch applies to: each hunk's context and removed lines at its line, filler elsewhere."""
    files: dict[str, list[str]] = {}
    without_final_newline: set[str] = set()
    lines: list[str] = []
    path = ""
    in_hunk = False
    previous = ""
    for line in patch.splitlines():
        if line.startswith("--- "):
            in_hunk = False
        elif line.startswith("+++ "):
            path = line.split(" ", 1)[1].split("/", 1)[1]
            lines = files.setdefault(path, [])
        elif match := re.match(r"@@ -(\d+)", line):
            lines.extend("// upstream" for _ in range(int(match[1]) - 1 - len(lines)))
            in_hunk = True
        elif in_hunk and line.startswith("\\"):
            if previous[:1] in (" ", "-"):
                without_final_newline.add(path)
        elif in_hunk and line[:1] in (" ", "-", ""):
            lines.append(line[1:])
        previous = line
    return {path: "\n".join(lines) + ("" if path in without_final_newline else "\n") for path, lines in files.items()}


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class Launcher:
    """A copy of the launcher with its inputs, an upstream repository, fake tools and the logs they write."""

    def __init__(self, tmp_path: Path):
        self.tmp = tmp_path
        self.root = tmp_path / "gigachad-bot"
        (self.root / "deploy").mkdir(parents=True)
        (self.root / "src" / "frontend").mkdir(parents=True)
        shutil.copy(REPOSITORY / "run-learnharness.sh", self.root)
        shutil.copy(REPOSITORY / "deploy" / "lib.sh", self.root / "deploy")
        shutil.copytree(REPOSITORY / "learnharness-overlay", self.root / "learnharness-overlay")
        (self.root / ".env.local").write_text("SECRET=1\n")
        # The repository's .venv/bin would otherwise shadow the fake node.
        (self.root / ".venv" / "bin").mkdir(parents=True)
        fake_command(self.root / ".venv" / "bin", "python", FAKE_PYTHON)

        self.source = tmp_path / "source"
        git(tmp_path, "init", "--initial-branch=main", str(self.source))
        for path, content in seams_preimages(self.overlay("seams.patch").read_text()).items():
            (self.source / path).parent.mkdir(parents=True, exist_ok=True)
            (self.source / path).write_text(content)
        (self.source / "package-lock.json").write_text("{}\n")
        for path, content in upstream_files().items():
            (self.source / path).parent.mkdir(parents=True, exist_ok=True)
            (self.source / path).write_text(content)
        self.commit("Fixture", REF)

        self.user_home = tmp_path / "user"
        self.user_home.mkdir()
        (self.user_home / ".secrets").write_text("export OPENAI_API_KEY=provider-secret\n")
        binaries = tmp_path / "bin"
        binaries.mkdir()
        self.npm_log = tmp_path / "npm.log"
        self.fail_build = tmp_path / "fail-build"
        self.systemd_run_log = tmp_path / "systemd-run.log"
        self.systemctl_log = tmp_path / "systemctl.log"
        self.checks_log = tmp_path / "checks.log"
        (binaries / "fake-next").write_text(FAKE_NEXT)
        (binaries / "fake-next").chmod(0o755)
        (binaries / "fake-tsc").write_text(FAKE_TSC.format(log=self.checks_log))
        (binaries / "fake-tsc").chmod(0o755)
        fake_command(
            binaries,
            "npm",
            FAKE_NPM.format(log=self.npm_log, next=binaries / "fake-next", tsc=binaries / "fake-tsc", fail=self.fail_build),
        )
        fake_command(binaries, "node", FAKE_NODE + FAKE_NODE_TEST.format(log=self.checks_log))
        fake_command(binaries, "systemd-run", FAKE_SYSTEMD_RUN.format(log=self.systemd_run_log))
        fake_command(binaries, "systemctl", FAKE_SYSTEMCTL.format(log=self.systemctl_log))
        fake_command(binaries, "flatpak", FAKE_FLATPAK)

        self.checkout = tmp_path / "nested" / "checkout"
        self.worktree = tmp_path / "worktree"
        # A quote and a space: the Chrome wrapper and the unit properties must keep the path whole.
        self.home = tmp_path / "learn harness's home"
        self.proxy = tmp_path / "proxy"
        self.proxy.mkdir()
        (self.proxy / "master.key").write_text(f"{MASTER_KEY}\n")
        self.write_config_log = tmp_path / "write-config.log"
        self.port = free_port()
        self.unit = f"gigachad-learnharness-{self.port}"
        self.environment = environment_with(
            binaries,
            HOME=str(self.user_home),
            GIGACHAD_LEARNHARNESS_REPO=str(self.checkout),
            GIGACHAD_LEARNHARNESS_DIR=str(self.worktree),
            GIGACHAD_LEARNHARNESS_REF=REF,
            GIGACHAD_LEARNHARNESS_HOME=str(self.home),
            GIGACHAD_LITELLM_PROXY_HOME=str(self.proxy),
            WRITE_CONFIG_LOG=str(self.write_config_log),
            GIGACHAD_LEARNHARNESS_BIND=f"127.0.0.1:{self.port}",
            OPENAI_API_KEY="provider-secret",
            GEMINI_API_KEY="provider-secret",
            LH_DESKTOP_SERVER="1",
            BRAVE_API_KEY="brave-key",
            XDG_RUNTIME_DIR="/run/user/1000",
            DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/1000/bus",
            GIT_CONFIG_COUNT="1",
            GIT_CONFIG_KEY_0=f"url.{self.source.as_uri()}.insteadOf",
            GIT_CONFIG_VALUE_0="https://github.com/amosblomqvist/learnharness.git",
        )
        for name in ("XDG_DATA_HOME", "GIGACHAD_LEARNHARNESS_FRAME_ANCESTORS", "UNIT_ACTIVE"):
            self.environment.pop(name, None)

    def overlay(self, name: str) -> Path:
        return self.root / "learnharness-overlay" / name

    def commit(self, message: str, tag: str) -> str:
        git(self.source, "add", ".")
        git(self.source, "commit", "-m", message)
        git(self.source, "tag", tag)
        return git(self.source, "rev-parse", "HEAD")

    def release(self, tag: str, **files: str) -> str:
        """Tag a new upstream commit holding *files* (names with `_` for `-`)."""
        (self.source / "README.md").write_text(f"release {tag}\n")
        for name, content in files.items():
            (self.source / name.replace("_", "-")).write_text(content)
        return self.commit(f"Release {tag}", tag)

    def run(self) -> subprocess.CompletedProcess[str]:
        return run_script(self.root / "run-learnharness.sh", self.environment, timeout=30)

    def select(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(self.root / "run-learnharness.sh"), "--select"],
            env=self.environment, text=True, capture_output=True, timeout=30, check=False,
        )

    def launch(self) -> subprocess.CompletedProcess[str]:
        result = self.run()
        assert result.returncode == 0, result.stdout + result.stderr
        return result

    def npm_calls(self) -> list[str]:
        return self.npm_log.read_text().splitlines() if self.npm_log.exists() else []

    def builds(self) -> int:
        return sum(call.startswith("run ") for call in self.npm_calls())

    def served_environment(self) -> dict[str, str]:
        lines = (self.worktree / ".next" / "served-env.txt").read_text().splitlines()
        return dict(line.split("=", 1) for line in lines if "=" in line)

    def checks(self) -> list[str]:
        """Where each overlay check ran: tsc and the overlay tests, in the check checkout."""
        if not self.checks_log.exists():
            return []
        return [line.split()[0] + " " + line.split()[1] for line in self.checks_log.read_text().splitlines()]

    def served_commit(self) -> str:
        return (self.home / "served-commit").read_text()

    def snapshots(self) -> list[Path]:
        snapshots = self.home / "snapshots"
        return sorted(snapshots.iterdir()) if snapshots.exists() else []


def test_launcher_serves_overlay_in_a_sandboxed_unit_from_an_allowlisted_environment(tmp_path):
    launcher = Launcher(tmp_path)
    home = launcher.home

    result = launcher.launch()

    first_commit = git(launcher.source, "rev-parse", "HEAD")
    worktree = launcher.worktree
    assert git(worktree, "rev-parse", "HEAD") == first_commit
    assert "<CustomRoot />" in (worktree / "app" / "layout.tsx").read_text()
    assert "withCustom(nextConfig)" in (worktree / "next.config.ts").read_text()
    assert (worktree / "custom" / "root.tsx").read_bytes() == launcher.overlay("root.tsx").read_bytes()
    served_arguments = (worktree / ".next" / "served-args.txt").read_text().split()
    assert served_arguments == ["start", "--hostname", "127.0.0.1", "--port", str(launcher.port)]
    assert f"Starting LearnHarness on 127.0.0.1:{launcher.port} (unit {launcher.unit})" in result.stdout

    # The unit: a named transient user service with the user's and gigachad's secrets unreadable, stopped on exit.
    unit_arguments = launcher.systemd_run_log.read_text().splitlines()
    options = unit_arguments[: unit_arguments.index("--")]
    assert {"--user", f"--unit={launcher.unit}", "--collect", "--pipe", f"--working-directory={worktree}"} <= set(options)
    hidden = {
        option.removeprefix('InaccessiblePaths="-').removesuffix('"') for option in options if option.startswith("InaccessiblePaths=")
    }
    user = launcher.user_home
    expected_hidden = {
        ".secrets", ".ssh", ".omp", ".gnupg", ".config/gh", ".aws", ".netrc", ".git-credentials", ".local/share/gigachad-bot/litellm-proxy",
        ".claude", ".claude.json", ".codex", ".config/opencode", ".local/share/opencode", ".local/share/keyrings",
        ".password-store", ".docker", ".kube", ".config/gcloud", ".azure", ".npmrc", ".pypirc", ".var/app",
        ".config/google-chrome", ".mozilla",
    }
    assert hidden == {str(user / name) for name in expected_hidden} | {str(launcher.proxy), str(launcher.root / ".env.local")}
    assert "UMask=0077" in options
    command = unit_arguments[unit_arguments.index("--") + 1 :]
    assert command[:2] == ["env", "-i"]
    systemctl_calls = launcher.systemctl_log.read_text().splitlines()
    assert systemctl_calls == [f"--user --quiet is-active {launcher.unit}", f"--user stop {launcher.unit}"]

    served = launcher.served_environment()
    assert set(served) - {"PWD", "SHLVL", "_", "OLDPWD"} == ALLOWED_ENVIRONMENT
    assert served["HOME"] == str(home / "home")
    assert served["LH_DATA_DIR"] == str(home / "data")
    assert served["LH_AUTH_FILE"] == str(home / "auth.json")
    assert served["DBUS_SESSION_BUS_ADDRESS"] == "unix:path=/run/user/1000/bus"
    assert served["NODE_ENV"] == "production"
    assert served["BRAVE_API_KEY"] == "brave-key"
    assert served["CUSTOM_OPENAI_API_KEY"] == MASTER_KEY
    assert served["GIGACHAD_LEARNHARNESS_FRAME_ANCESTORS"] == SHELL_ORIGINS
    assert stat.S_IMODE((home).stat().st_mode) == 0o700

    # Demos are files in TMPDIR, which the Chrome wrapper shares with the flatpak sandbox (its own /tmp is private).
    assert served["TMPDIR"] == str(home / "tmp")
    chrome = Path(served["CHROME_PATH"])
    assert os.access(chrome, os.X_OK)
    flatpak = subprocess.run([str(chrome), "--headless", "a b"], env=launcher.environment, capture_output=True, text=True, check=True)
    assert flatpak.stdout.splitlines() == [
        f"FLATPAK_USER_DIR={user / '.local' / 'share' / 'flatpak'}",
        "run",
        f"--filesystem={home / 'tmp'}",
        "com.google.Chrome",
        "--headless",
        "a b",
    ]

    # Installs and the build never see a provider key.
    assert launcher.npm_calls() == ["ci openai=none", "install openai=none", "run openai=none"]
    # The proxy's master key is read through the generator, which creates it when the proxy launcher has not yet.
    generator_calls = launcher.write_config_log.read_text()
    assert generator_calls == f"-m lib.learnharness master-key home={home} proxy={launcher.proxy} src={launcher.root / 'src'}\n"
    # LearnHarness listened: its data now belongs to this commit.
    assert (home / "served-commit").read_text() == first_commit
    assert launcher.snapshots() == []


def test_launcher_rebuilds_exactly_when_an_input_changed_and_after_a_failed_build(tmp_path):
    launcher = Launcher(tmp_path)
    launcher.launch()
    assert launcher.builds() == 1

    launcher.launch()
    assert launcher.npm_calls() == ["ci openai=none", "install openai=none", "run openai=none"]

    launcher.overlay("root.tsx").write_text(launcher.overlay("root.tsx").read_text() + "// edited\n")
    launcher.launch()
    assert launcher.builds() == 2

    launcher.overlay("theme/tokens.css").unlink()
    launcher.launch()
    assert launcher.builds() == 3
    assert not (launcher.worktree / "custom" / "theme" / "tokens.css").exists()

    launcher.environment["CHAT_UI_CONTENT"] = "chat-ui 2"
    launcher.launch()
    assert launcher.builds() == 4
    assert launcher.npm_calls()[-2] == "install openai=none"

    launcher.environment["GIGACHAD_LEARNHARNESS_FRAME_ANCESTORS"] = "https://gigachad.tail.ts.net"
    launcher.launch()
    assert launcher.builds() == 5

    # Next wrote BUILD_ID before failing: the next start still rebuilds.
    launcher.fail_build.touch()
    launcher.overlay("root.tsx").write_text(launcher.overlay("root.tsx").read_text() + "// broken\n")
    assert launcher.run().returncode != 0
    assert (launcher.worktree / ".next" / "BUILD_ID").exists()
    launcher.fail_build.unlink()
    launcher.launch()
    assert launcher.builds() == 7
    launcher.launch()
    assert launcher.builds() == 7


def test_launcher_snapshots_data_and_credentials_once_per_move_to_another_commit(tmp_path):
    launcher = Launcher(tmp_path)
    launcher.launch()
    first_commit = git(launcher.source, "rev-parse", "HEAD")
    home = launcher.home

    # Same ref again: nothing to snapshot, the patched worktree stays as it is.
    (home / "data" / "chats").mkdir(parents=True)
    (home / "data" / "chats" / "c1.json").write_text("{}")
    (home / "auth.json").write_text('{"k": 1}')
    (home / "chatgpt-credentials.json").write_text('{"c": 1}')
    launcher.launch()
    assert launcher.snapshots() == []
    assert "<CustomRoot />" in (launcher.worktree / "app" / "layout.tsx").read_text()

    # A new ref the clone has not fetched yet: fetched, and the data copied aside under the commit that wrote it.
    (launcher.source / "README.md").write_text("next release\n")
    second_commit = launcher.commit("Next release", "v0.5.0")
    launcher.environment["GIGACHAD_LEARNHARNESS_REF"] = "v0.5.0"
    launcher.launch()
    assert git(launcher.worktree, "rev-parse", "HEAD") == second_commit
    assert "<CustomRoot />" in (launcher.worktree / "app" / "layout.tsx").read_text()
    (snapshot,) = launcher.snapshots()
    assert snapshot.name.startswith(f"{first_commit}-")
    assert (snapshot / "data" / "chats" / "c1.json").read_text() == "{}"
    assert (snapshot / "auth.json").read_text() == '{"k": 1}'
    assert (snapshot / "chatgpt-credentials.json").read_text() == '{"c": 1}'
    assert (home / "data" / "chats" / "c1.json").exists()
    assert (home / "served-commit").read_text() == second_commit

    # A release the seams no longer apply to fails before serving: retries keep one snapshot and the served commit.
    (launcher.source / "app" / "layout.tsx").write_text("export default function RootLayout() {}\n")
    launcher.commit("Layout rewrite", "v0.6.0")
    launcher.environment["GIGACHAD_LEARNHARNESS_REF"] = "v0.6.0"
    assert launcher.run().returncode != 0
    assert launcher.run().returncode != 0
    assert [path.name.split("-")[0] for path in launcher.snapshots()].count(second_commit) == 1
    assert (home / "served-commit").read_text() == second_commit

    # A ref nowhere: named in the error.
    launcher.environment["GIGACHAD_LEARNHARNESS_REF"] = "v9.9.9"
    missing = launcher.run()
    assert missing.returncode != 0
    assert "ref v9.9.9 not found" in missing.stderr


def test_launcher_refuses_a_ref_whose_model_picker_lacks_what_the_freeze_wraps(tmp_path):
    launcher = Launcher(tmp_path)
    picker = launcher.source / "components" / "model-picker.tsx"
    picker.write_text(picker.read_text().replace("lh:kbd-cycle-thinking", "lh:kbd-thinking-next"))
    launcher.commit("Renamed event", "v0.5.0")
    launcher.environment["GIGACHAD_LEARNHARNESS_REF"] = "v0.5.0"

    refused = launcher.run()

    assert refused.returncode != 0
    assert "lacks lh:kbd-cycle-thinking, the learn-mode overlay needs updating" in refused.stderr
    assert launcher.npm_calls() == []
    assert not (launcher.home / "served-commit").exists()


def test_launcher_refuses_a_ref_lacking_a_marker_the_overlay_or_adapter_relies_on(tmp_path):
    assert len(LAUNCHER_MARKERS) > 30
    launcher = Launcher(tmp_path)
    chat = launcher.source / "components" / "chat.tsx"
    chat.write_text(chat.read_text().replace('data-testid="plan-chip"', 'data-testid="plan-button"'))
    launcher.commit("Renamed plan chip", "v0.5.0")
    launcher.environment["GIGACHAD_LEARNHARNESS_REF"] = "v0.5.0"

    refused = launcher.run()

    assert refused.returncode != 0
    assert 'lacks data-testid="plan-chip", the learn-mode overlay needs updating' in refused.stderr
    assert launcher.npm_calls() == []
    assert not (launcher.home / "served-commit").exists()


class _Serving(BaseHTTPRequestHandler):
    """A LearnHarness already on the port, its CSP from the server's `csp` attribute."""

    def do_HEAD(self):
        self.send_response(200)
        if self.server.csp:
            self.send_header("Content-Security-Policy", self.server.csp)
        self.end_headers()

    def log_message(self, *args):
        pass


def test_launcher_reuses_a_running_learnharness_without_side_effects(tmp_path):
    launcher = Launcher(tmp_path)
    server = HTTPServer(("127.0.0.1", launcher.port), _Serving)
    server.csp = f"frame-ancestors 'self' {SHELL_ORIGINS}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        reused = launcher.launch()
        assert reused.stdout == f"Reusing the LearnHarness already listening on 127.0.0.1:{launcher.port}\n"
        assert reused.stderr == ""

        server.csp = None
        unframed = launcher.launch()
        assert "refuses frames from http://127.0.0.1:2999" in unframed.stderr
    finally:
        server.shutdown()
        server.server_close()
    assert not launcher.checkout.exists()
    assert not launcher.worktree.exists()
    assert not launcher.home.exists()
    assert launcher.npm_calls() == []
    assert not launcher.systemd_run_log.exists()

    # Its unit is still starting: reused too.
    launcher.environment["UNIT_ACTIVE"] = "1"
    starting = launcher.launch()
    assert starting.stdout == f"Reusing the LearnHarness unit {launcher.unit}, still starting on 127.0.0.1:{launcher.port}\n"
    assert not launcher.checkout.exists()


def test_launcher_rejects_a_directory_that_is_not_a_learnharness_checkout(tmp_path):
    launcher = Launcher(tmp_path)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "keep").write_text("untouched")
    launcher.environment["GIGACHAD_LEARNHARNESS_REPO"] = str(unrelated)

    rejected = launcher.run()

    assert rejected.returncode != 0
    assert "not a LearnHarness Git checkout" in rejected.stderr
    assert (unrelated / "keep").read_text() == "untouched"
    assert not launcher.worktree.exists()


def auto_launcher(tmp_path: Path) -> tuple[Launcher, str]:
    """A launcher following upstream's releases, having served the first one, and that release's commit."""
    launcher = Launcher(tmp_path)
    del launcher.environment["GIGACHAD_LEARNHARNESS_REF"]
    first = git(launcher.source, "rev-parse", "HEAD")
    launcher.launch()
    assert launcher.served_commit() == first
    check_dir = f"{launcher.worktree}-check"
    assert launcher.checks() == [f"tsc {check_dir}", f"test {check_dir}"]
    return launcher, first


def test_auto_update_serves_a_newer_release_that_passes_the_checks_and_checks_it_once(tmp_path):
    launcher, first = auto_launcher(tmp_path)
    assert launcher.builds() == 1
    second = launcher.release("v0.5.0")

    # redeploy.sh asks which commit a start would serve: the checks run once, the start reuses their result.
    selected = launcher.select()
    assert selected.returncode == 0, selected.stderr
    assert selected.stdout == f"{second}\n"
    assert len(launcher.checks()) == 4
    assert git(Path(f"{launcher.worktree}-check"), "rev-parse", "HEAD") == second
    assert git(launcher.worktree, "rev-parse", "HEAD") == first

    updated = launcher.launch()
    assert git(launcher.worktree, "rev-parse", "HEAD") == second
    assert "<CustomRoot />" in (launcher.worktree / "app" / "layout.tsx").read_text()
    assert launcher.served_commit() == second
    assert f"LearnHarness update: v0.4.0 ({first[:7]}) -> v0.5.0 ({second[:7]})\n" in updated.stderr
    assert len(launcher.checks()) == 4
    assert launcher.builds() == 2

    # An unchanged upstream costs only the fetch: no check, no build, no update line.
    again = launcher.launch()
    assert len(launcher.checks()) == 4
    assert launcher.builds() == 2
    assert "LearnHarness update" not in again.stderr and "not updated" not in again.stderr


def test_auto_update_keeps_the_last_working_release_when_a_newer_one_fails_a_check_or_its_build(tmp_path):
    launcher, first = auto_launcher(tmp_path)
    launcher.release("v0.5.0", types_broken="")

    kept = launcher.launch()
    assert git(launcher.worktree, "rev-parse", "HEAD") == first
    assert launcher.served_commit() == first
    (line,) = [line for line in kept.stderr.splitlines() if "not updated" in line]
    assert line.startswith(f"LearnHarness not updated, serving v0.4.0 ({first[:7]}): v0.5.0 failed tsc (log: ")
    assert Path(line.split("(log: ")[1].rstrip(")")).exists()
    assert launcher.builds() == 1
    # The failure is kept for this overlay: the next start does not check it again.
    checks = len(launcher.checks())
    assert "v0.5.0 failed tsc" in launcher.launch().stderr
    assert len(launcher.checks()) == checks

    # Passing its checks, failing its build: back to the last working release, which serves, and not tried again.
    (launcher.source / "types-broken").unlink()
    third = launcher.release("v0.6.0", build_broken="")
    fallback = launcher.launch()
    assert f"LearnHarness not updated, serving v0.4.0 ({first[:7]}): v0.6.0 ({third[:7]}) failed to build" in fallback.stderr
    assert git(launcher.worktree, "rev-parse", "HEAD") == first
    assert launcher.served_commit() == first
    assert launcher.builds() == 3
    retried = launcher.launch()
    assert "v0.6.0 (" in retried.stderr and "failed to build" in retried.stderr
    assert launcher.builds() == 3


def test_auto_update_keeps_the_last_working_release_when_offline(tmp_path):
    launcher, first = auto_launcher(tmp_path)
    launcher.release("v0.5.0")
    launcher.source.rename(tmp_path / "unreachable")

    offline = launcher.launch()

    assert git(launcher.worktree, "rev-parse", "HEAD") == first
    assert launcher.served_commit() == first
    assert f"LearnHarness not updated, serving v0.4.0 ({first[:7]}): fetching origin's tags failed" in offline.stderr
    assert len(launcher.checks()) == 2


def test_a_pinned_ref_and_the_opt_out_are_never_updated(tmp_path):
    launcher = Launcher(tmp_path)
    launcher.launch()
    first = git(launcher.source, "rev-parse", "HEAD")
    launcher.release("v0.5.0")

    pinned = launcher.launch()
    assert git(launcher.worktree, "rev-parse", "HEAD") == first
    assert launcher.served_commit() == first
    assert launcher.checks() == []
    assert "LearnHarness update" not in pinned.stderr
    assert git(launcher.checkout, "tag", "--list", "v0.5.0") == ""

    del launcher.environment["GIGACHAD_LEARNHARNESS_REF"]
    launcher.environment["GIGACHAD_AUTO_UPDATE"] = "0"
    opted_out = launcher.launch()
    assert git(launcher.worktree, "rev-parse", "HEAD") == first
    assert f"LearnHarness not updated, serving v0.4.0 ({first[:7]}): auto-update is off (GIGACHAD_AUTO_UPDATE=0)" in opted_out.stderr
    assert launcher.checks() == []
    assert git(launcher.checkout, "tag", "--list", "v0.5.0") == ""


def test_auto_update_snapshots_the_data_before_switching(tmp_path):
    launcher, first = auto_launcher(tmp_path)
    home = launcher.home
    (home / "data" / "chats").mkdir(parents=True)
    (home / "data" / "chats" / "c1.json").write_text("{}")
    (home / "auth.json").write_text('{"k": 1}')
    second = launcher.release("v0.5.0")

    launcher.launch()

    assert launcher.served_commit() == second
    (snapshot,) = launcher.snapshots()
    assert snapshot.name.startswith(f"{first}-")
    assert (snapshot / "data" / "chats" / "c1.json").read_text() == "{}"
    assert (snapshot / "auth.json").read_text() == '{"k": 1}'
