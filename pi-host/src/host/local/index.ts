/**
 * `LocalHost`: the `Host` of the machine pi-host runs on (machine id `local`).
 *
 * - Consumes: the Pi binary, the `agent-sandbox` CLI, pi-host's environment, the repo extensions dir, Pi's session
 *   files and sidecars, git in a requested cwd
 * - Provides: `LocalHost`, the only place that spawns processes, touches or watches files or takes `flock`s
 *
 * Spawn strategies (plan D5): `coding` and `document` run the Pi binary directly, the profile shapes its arguments
 * (`supervisor/profiles.ts`) and both load the repo extensions (C6) unless those arguments carry `--no-extensions`.
 * Names in `<extensionsDir>/exclude.json` (package source, extension name or path) are kept out of both: pi-host then
 * resolves Pi's own extension set and passes it as `--no-extensions` plus explicit `-e` paths.
 * Both always load pi-host's `extensions/gateway-provider.ts` with the gateway URL and the gateway-only token in their
 * environment: the provider `gateway` serves models of providers the child has no key for, keys stay in pi-host.
 * `sandboxed` runs the same arguments through `agent-sandbox pi-rpc` (`sandbox.ts`, C4, C5), which keeps the same
 * stdio, stdin-EOF and SIGTERM contract.
 */
import type { Dirent } from "node:fs";
import { readdir, readFile, stat } from "node:fs/promises";
import { homedir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { DefaultPackageManager, SettingsManager } from "@earendil-works/pi-coding-agent";
import { loadToken } from "../../auth.ts";
import type {
  DeleteResult,
  ForkRequest,
  ForkResult,
  Host,
  LockResult,
  MoveResult,
  PiChild,
  PrivateState,
  SearchMatch,
  SessionMatchPage,
  SessionEntries,
  SessionMeta,
  SpawnRequest,
} from "../index.ts";
import { childEnv, gatewayEnv } from "./env.ts";
import { listFiles } from "./files.ts";
import { forkSession } from "./fork.ts";
import { tryLockFile, type FileLock } from "./lock.ts";
import { spawnSandboxed } from "./sandbox.ts";
import { searchSessions, sessionMatches } from "./search.ts";
import { deleteSession, findSession, listSessions, readEntries, moveSession, sessionMeta } from "./sessions.ts";
import { readState, writeState } from "./state.ts";
import { watchSessions } from "./watch.ts";

export type LocalHostOptions = {
  /** Pi executable (C4). */
  piBin: string;
  /** Extra variable names children inherit beyond the M6 allowlist. */
  extraChildEnv: string[];
  /** Repo extensions dir, each top-level extension loaded into unrestricted local children (C6). */
  extensionsDir: string;
  /** `agent-sandbox` CLI running `sandboxed` children (C5). */
  agentSandboxBin: string;
  /** AiB profile of `sandboxed` children. */
  sandboxProfile: string;
  /** pi-host's own port, the completion gateway sandboxed children reach. */
  gatewayPort: number;
  /** The gateway-only token file (C1), the only credential a sandboxed child gets. */
  gatewayTokenPath: string;
};

/** True for a C6 extension entry: a `*.ts` file (not `*.test.ts` or `*.d.ts`) or a directory holding `index.ts`. */
async function isExtension(dir: string, entry: Dirent): Promise<boolean> {
  if (entry.isFile()) return entry.name.endsWith(".ts") && !/\.(test|d)\.ts$/.test(entry.name);
  if (!entry.isDirectory()) return false;
  return (await stat(join(dir, entry.name, "index.ts")).catch(() => undefined))?.isFile() ?? false;
}

/** `-e <abs path>` for each extension in `dir`, sorted, listed per spawn; none when `dir` is missing. */
async function extensionArgs(dir: string): Promise<string[]> {
  const entries = await readdir(dir, { withFileTypes: true }).catch((error: NodeJS.ErrnoException) => {
    if (error.code === "ENOENT") return [];
    throw error;
  });
  const kept = await Promise.all(entries.map(async (entry) => ((await isExtension(dir, entry)) ? entry.name : undefined)));
  return kept
    .filter((name): name is string => name !== undefined)
    .sort()
    .flatMap((name) => ["-e", join(dir, name)]);
}

/** Pi's built-in extensions (Pi 1.5.2 `settings.md`, "Resources"). */
const BUILTIN_EXTENSIONS = ["mcp", "llama.cpp", "codemode", "tool-search"];

/** Names in `<dir>/exclude.json`, extensions children skip; none when the file is missing. */
async function excludedNames(dir: string): Promise<Set<string>> {
  const text = await readFile(join(dir, "exclude.json"), "utf8").catch((error: NodeJS.ErrnoException) => {
    if (error.code === "ENOENT") return "[]";
    throw error;
  });
  const names: unknown = JSON.parse(text);
  if (!Array.isArray(names) || !names.every((name) => typeof name === "string")) {
    throw new Error(`${join(dir, "exclude.json")} must be a JSON array of extension names`);
  }
  return new Set(names);
}

/** An extension's names: its package source (`npm:pi-foo`) and its own name (file stem, or dir of an `index.*`). */
function extensionNames(path: string, source: string): string[] {
  const file = basename(path).replace(/\.[cm]?[jt]s$/, "");
  return [source, path, file === "index" ? basename(dirname(path)) : file];
}

/**
 * The extensions Pi would load for `cwd` minus `excluded`, as `--no-extensions` plus `-e` each, so excluded ones are
 * never imported (faster child start). Project-scope extensions are dropped: an RPC child cannot grant project trust.
 */
async function filteredExtensionArgs(cwd: string, excluded: Set<string>): Promise<string[]> {
  const agentDir = process.env.PI_CODING_AGENT_DIR || join(homedir(), ".pi/agent");
  const settingsManager = SettingsManager.create(cwd, agentDir);
  const manager = new DefaultPackageManager({ cwd, agentDir, settingsManager, builtinExtensions: BUILTIN_EXTENSIONS });
  const { extensions } = await manager.resolve(async () => "skip");
  const kept = extensions.filter(
    ({ path, enabled, metadata }) =>
      enabled && metadata.scope === "user" && !extensionNames(path, metadata.source).some((name) => excluded.has(name)),
  );
  return ["--no-extensions", ...kept.flatMap(({ path }) => ["-e", path])];
}

/** pi-host's own extension registering the provider `gateway` in local children. */
export const GATEWAY_EXTENSION = resolve(import.meta.dir, "../../../extensions/gateway-provider.ts");

export class LocalHost implements Host {
  readonly machine = "local";
  private readonly locks = new Map<string, FileLock>();

  constructor(private readonly options: LocalHostOptions) {}

  async piVersion(): Promise<string> {
    const child = Bun.spawn([this.options.piBin, "--version"], {
      env: childEnv(this.options.extraChildEnv),
      stdout: "pipe",
      stderr: "pipe",
    });
    const [output, code] = await Promise.all([new Response(child.stdout).text(), child.exited]);
    if (code !== 0) throw new Error(`${this.options.piBin} --version exited ${code}: "${output.trim()}"`);
    return output.trim().split(/\s+/).pop() ?? "";
  }

  async spawn({ profile, cwd, piArgs }: SpawnRequest): Promise<PiChild> {
    if (profile === "sandboxed") return spawnSandboxed(this.options, cwd, piArgs);
    const restricted = piArgs.includes("--no-extensions");
    const excluded = restricted ? new Set<string>() : await excludedNames(this.options.extensionsDir);
    const extensions = restricted ? [] : await extensionArgs(this.options.extensionsDir);
    if (excluded.size > 0) extensions.unshift(...(await filteredExtensionArgs(cwd, excluded)));
    const gateway = gatewayEnv(this.options.gatewayPort, await loadToken(this.options.gatewayTokenPath));
    return Bun.spawn([this.options.piBin, ...piArgs, "-e", GATEWAY_EXTENSION, ...extensions], {
      cwd,
      env: { ...childEnv(this.options.extraChildEnv), ...gateway },
      stdin: "pipe",
      stdout: "pipe",
      stderr: "pipe",
    });
  }

  async lockSession(sessionPath: string): Promise<LockResult> {
    let lock: FileLock | undefined;
    try {
      lock = tryLockFile(sessionPath);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return { ok: false, error: "missing" };
      throw error;
    }
    if (!lock) return { ok: false, error: "busy" };
    const lockId = crypto.randomUUID();
    this.locks.set(lockId, lock);
    return { ok: true, lockId };
  }

  async unlockSession(lockId: string): Promise<void> {
    this.locks.get(lockId)?.release();
    this.locks.delete(lockId);
  }

  async isDirectory(path: string): Promise<boolean> {
    return (await stat(path).catch(() => undefined))?.isDirectory() ?? false;
  }

  listSessions(cwd: string | null, archived: boolean): Promise<SessionMeta[]> {
    return listSessions(cwd, archived);
  }

  sessionMeta(sessionPath: string): Promise<SessionMeta | undefined> {
    return sessionMeta(sessionPath);
  }

  findSession(sessionId: string): Promise<string | undefined> {
    return findSession(sessionId);
  }

  readEntries(sessionPath: string): Promise<SessionEntries> {
    return readEntries(sessionPath);
  }

  moveSession(sessionPath: string, toArchive: boolean): Promise<MoveResult> {
    return moveSession(sessionPath, toArchive);
  }

  deleteSession(sessionPath: string): Promise<DeleteResult> {
    return deleteSession(sessionPath);
  }

  forkSession(request: ForkRequest): Promise<ForkResult> {
    return forkSession(request);
  }

  readState(sessionPath: string): Promise<PrivateState> {
    return readState(sessionPath);
  }

  writeState(sessionPath: string, state: PrivateState): Promise<void> {
    return writeState(sessionPath, state);
  }

  listFiles(cwd: string): Promise<string[]> {
    return listFiles(cwd);
  }

  async searchSessions(query: string, limit: number): Promise<SearchMatch[]> {
    const metas = await listSessions(null, false);
    return searchSessions(query, limit, metas.filter((meta) => meta.factory === undefined));
  }

  sessionMatches(sessionPath: string, query: string, limit: number): Promise<SessionMatchPage | undefined> {
    return sessionMatches(sessionPath, query, limit);
  }

  watchSessions(onChange: (sessionPaths: string[]) => void): () => void {
    return watchSessions(onChange);
  }
}
