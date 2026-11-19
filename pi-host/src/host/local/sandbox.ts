/**
 * The `sandboxed` spawn strategy (C4, C5): Pi in an Agents-in-a-box container through `agent-sandbox pi-rpc`, with the
 * same stdio, stdin-EOF and SIGTERM contract as a local `pi --mode rpc` child.
 *
 * - Consumes: the `agent-sandbox` CLI, the gateway-only token file (C1), Pi's session paths on this machine
 * - Provides: `spawnSandboxed`
 *
 * The session file stays host-side (M1 lock, catalog): the container bind-mounts the session's directory and Pi stores
 * a new session there through `--session-dir`. The child gets no `-e` extensions (C5) and no model credentials.
 */
import { mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import { loadToken } from "../../auth.ts";
import type { PiChild } from "../index.ts";
import { sandboxEnv } from "./env.ts";
import { cwdSessionDir, sessionsRoot } from "./paths.ts";

export type SandboxOptions = {
  agentSandboxBin: string;
  sandboxProfile: string;
  gatewayPort: number;
  gatewayTokenPath: string;
};

/** Host directory of the child's session: the resumed file's directory, else Pi's directory for `cwd`. */
function sessionDirOf(cwd: string, piArgs: string[]): string {
  const at = piArgs.indexOf("--session");
  const sessionPath = at >= 0 ? piArgs[at + 1] : undefined;
  return sessionPath ? dirname(sessionPath) : cwdSessionDir(sessionsRoot(), cwd);
}

/** `agent-sandbox pi-rpc` argv (C4) running `piArgs` in `cwd`, the session directory passed to both. */
function sandboxArgv(options: SandboxOptions, cwd: string, piArgs: string[]): string[] {
  const sessionDir = sessionDirOf(cwd, piArgs);
  return [
    options.agentSandboxBin,
    "pi-rpc",
    "--profile",
    options.sandboxProfile,
    "--cwd",
    cwd,
    "--session-dir",
    sessionDir,
    "--gateway-port",
    String(options.gatewayPort),
    "--",
    ...piArgs,
    "--session-dir",
    sessionDir,
  ];
}

/** Start a sandboxed child; its session directory exists first, so the bind mount never creates it. */
export async function spawnSandboxed(options: SandboxOptions, cwd: string, piArgs: string[]): Promise<PiChild> {
  await mkdir(sessionDirOf(cwd, piArgs), { recursive: true });
  const gatewayToken = await loadToken(options.gatewayTokenPath);
  return Bun.spawn(sandboxArgv(options, cwd, piArgs), {
    cwd,
    env: sandboxEnv(gatewayToken),
    stdin: "pipe",
    stdout: "pipe",
    stderr: "pipe",
  });
}
