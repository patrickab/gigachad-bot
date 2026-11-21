/**
 * Commands of the pipeline (hooks, gates, git, the reviewer child), each in its own process group.
 *
 * - Consumes an argv and a cwd
 * - Provides `CommandRunner`: `run` starts commands that `cancel()` kills as whole groups, `cleanup` starts the ones a
 *   cancel must not stop
 * - Memory per command is bounded: the end of the output, and stdout only as far as the caller asks for it
 * - `pi.exec` is not used: it reports a killed command as exit 0 and kills only the direct child
 */
import { spawn } from "node:child_process";

const KILL_GRACE_MS = 5000;
/** How long the pipes of a command that has exited may still deliver output. */
const EXIT_DRAIN_MS = 1000;
/** Enough for any tail the pipeline shows, small enough to hold for every command. */
const OUTPUT_KEEP_CHARS = 64_000;
/** A git command that would open an editor (`rebase --continue`, `commit` without `-m`) takes the message as it is: no command has a TTY. */
const NO_EDITOR_ENV = { GIT_EDITOR: "true", GIT_SEQUENCE_EDITOR: "true" };

export interface CommandOptions {
  cwd: string;
  /** Added to the process environment. */
  env?: Record<string, string>;
  timeoutMs?: number;
  /** How much of stdout to keep from its start. Without it only `output` holds the end. */
  maxStdoutChars?: number;
}

export interface CommandResult {
  /** Null when a signal ended the command or it could not start. */
  exitCode: number | null;
  killed: boolean;
  timedOut: boolean;
  durationMs: number;
  /** Stdout up to `maxStdoutChars`, empty when none was asked for. */
  stdout: string;
  /** True when stdout was longer than what `stdout` holds. */
  stdoutTruncated: boolean;
  /** The end of stdout and stderr in arrival order. */
  output: string;
}

export type Exec = (argv: string[], options: CommandOptions) => Promise<CommandResult>;

/** Starts `argv` in its own process group. `kills`, when given, holds the kill of the command while it runs. */
function spawnCommand(argv: string[], options: CommandOptions, kills?: Set<() => void>): Promise<CommandResult> {
  return new Promise((resolve) => {
    const started = Date.now();
    const keep = options.maxStdoutChars ?? 0;
    let stdout = "";
    let stdoutTruncated = false;
    let output = "";
    let killed = false;
    let timedOut = false;
    let finished = false;
    const child = spawn(argv[0], argv.slice(1), {
      cwd: options.cwd,
      env: { ...process.env, ...NO_EDITOR_ENV, ...options.env },
      detached: true,
      stdio: ["ignore", "pipe", "pipe"],
    });
    // Decoded by the stream, so a multi-byte character split over two chunks stays whole.
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    const signalGroup = (signal: NodeJS.Signals) => {
      if (child.pid === undefined) return;
      try {
        process.kill(-child.pid, signal);
      } catch {
        // the group is already gone
      }
    };
    let forceKill: ReturnType<typeof setTimeout> | undefined;
    let drain: ReturnType<typeof setTimeout> | undefined;
    const kill = () => {
      if (killed) return;
      killed = true;
      signalGroup("SIGTERM");
      forceKill = setTimeout(() => signalGroup("SIGKILL"), KILL_GRACE_MS);
    };
    const timeout =
      options.timeoutMs === undefined
        ? undefined
        : setTimeout(() => {
            timedOut = true;
            kill();
          }, options.timeoutMs);
    const finish = (exitCode: number | null) => {
      if (finished) return;
      finished = true;
      clearTimeout(timeout);
      clearTimeout(forceKill);
      clearTimeout(drain);
      kills?.delete(kill);
      // Members of a killed group that ignored SIGTERM do not outlive the command.
      if (killed) signalGroup("SIGKILL");
      child.stdout.destroy();
      child.stderr.destroy();
      resolve({ exitCode, killed, timedOut, durationMs: Date.now() - started, stdout, stdoutTruncated, output });
    };
    kills?.add(kill);
    child.stdout.on("data", (chunk: string) => {
      const room = keep - stdout.length;
      if (room > 0) stdout += chunk.slice(0, room);
      if (chunk.length > room) stdoutTruncated = true;
      output = (output + chunk).slice(-OUTPUT_KEEP_CHARS);
    });
    child.stderr.on("data", (chunk: string) => {
      output = (output + chunk).slice(-OUTPUT_KEEP_CHARS);
    });
    child.on("error", (error) => {
      output += `${argv[0]}: ${error.message}\n`;
      finish(null);
    });
    // `close` waits for every holder of the pipes. A command that left a daemon behind them has ended at `exit`:
    // what is still in the pipes gets a short drain, then the command counts as ended.
    child.on("exit", (code) => {
      drain = setTimeout(() => finish(code), EXIT_DRAIN_MS);
    });
    child.on("close", (code) => finish(code));
  });
}

export class CommandRunner {
  private readonly kills = new Set<() => void>();
  private cancelRequested = false;

  get cancelled(): boolean {
    return this.cancelRequested;
  }

  /** Kills every running command group. Until `reset()`, later `run` commands do not start and report as killed. */
  cancel(): void {
    this.cancelRequested = true;
    for (const kill of this.kills) kill();
  }

  reset(): void {
    this.cancelRequested = false;
  }

  run: Exec = (argv, options) => {
    if (this.cancelRequested) {
      return Promise.resolve({ exitCode: null, killed: true, timedOut: false, durationMs: 0, stdout: "", stdoutTruncated: false, output: "" });
    }
    return spawnCommand(argv, options, this.kills);
  };

  /**
   * Runs a command that `cancel()` neither prevents nor kills: cleanup that has to happen after a cancel
   * (`git rebase --abort`, bringing an index back in line), and short steps that must not be cut in half.
   */
  cleanup: Exec = (argv, options) => spawnCommand(argv, options);
}
