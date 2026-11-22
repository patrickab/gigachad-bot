/**
 * The machine seam (M23): every machine-local action of pi-host goes through one `Host`.
 *
 * - Consumes: nothing; `LocalHost` (`./local/index.ts`) implements it for this machine
 * - Provides: `Host`, the data types its operations take and return, `PiChild`, `Profile`
 *
 * Operations (fixed list, no generic exec or file read):
 * - `piVersion`: version the machine's Pi binary reports
 * - `spawn`: start a `pi --mode rpc` child for a profile through its spawn strategy (the Pi binary, or an AiB container
 *   for `sandboxed`, D5)
 * - `lockSession` / `unlockSession`: the M1 lock on a session file
 * - `isDirectory`: check a working directory before the per-cwd operations
 * - `listSessions`, `sessionMeta`, `findSession`: session listing per cwd or across all cwds, header and entry
 *   metadata, lookup by id
 * - `moveSession`: archive or restore a session file and its sidecar
 * - `deleteSession`: remove a session file and its sidecar
 * - `forkSession`: write a new session file holding one branch of another (D11, Pi's fork and clone)
 * - `readState` / `writeState`: the private state sidecar (M19, M20)
 * - `listFiles`: `@` completion candidates of a cwd (M7)
 * - `searchSessions`: full-text search over live session transcripts (M24)
 * - `watchSessions`: changed session files of the live root, debounced (M25)
 *
 * Inputs and outputs are plain serializable data, so a future `RemoteHost` proxies each operation over a link.
 * Two exceptions: `PiChild`, a narrow stdio stream type a transport can carry as a byte pipe, and the
 * `watchSessions` listener, a stream of plain `string[]` messages a transport carries as frames.
 */

/**
 * `coding`: the user's Pi; `document`: the user's Pi with a workbench chat's prompt, restricted on request (D4, D9,
 * `supervisor/profiles.ts`);
 * `sandboxed`: `coding` arguments in an Agents-in-a-box container that reaches models only through the gateway (C5).
 */
export type Profile = "coding" | "document" | "sandboxed";
export const PROFILES: readonly Profile[] = ["coding", "document", "sandboxed"];

export type SpawnRequest = {
  profile: Profile;
  cwd: string;
  /** Arguments after the binary, starting with `--mode rpc`. */
  piArgs: string[];
};

/** A Pi child's stdio, exit and signals; a `Bun.Subprocess` locally, a proxied pipe remotely. */
export interface PiChild {
  readonly stdin: { write(data: string): unknown; flush(): unknown; end(): unknown };
  readonly stdout: AsyncIterable<Uint8Array>;
  readonly stderr: AsyncIterable<Uint8Array>;
  readonly exited: Promise<number>;
  kill(signal: "SIGTERM" | "SIGKILL"): void;
}

/** A session file's entries (pi's `session-format.md` lines after the header) in append order, the last one the leaf. */
export type SessionEntries = { entries: Record<string, unknown>[]; leafId: string | null };

/** A held lock is addressed by `lockId`; `busy` when another holder has the file. */
export type LockResult = { ok: true; lockId: string } | { ok: false; error: "busy" | "missing" };

export const FACTORY_STATES = ["running", "waiting_approve", "blocked", "merging", "done", "cancelled"] as const;

/** A factory task's state as its session file last recorded it (a `factory-state` custom entry, `at` its timestamp). */
export type FactoryState = { state: (typeof FACTORY_STATES)[number]; round: number; reason: string; at: string };

/** What a session file says about itself plus where it lives. */
export type SessionMeta = {
  sessionId: string;
  /** Absolute path as listed. */
  sessionPath: string;
  /** `sessionPath` with symlinks resolved, the key live sessions are deduplicated by. */
  realPath: string;
  cwd: string;
  createdAt: string;
  modifiedAt: string;
  firstMessage: string;
  messageCount: number;
  /** Context size after the last answer, 0 before one. */
  tokens: number;
  parentSession?: string;
  archived: boolean;
  /** Present when the session is a factory task: its last `factory-state` entry in file order. */
  factory?: FactoryState;
  /** What the session's models used, present once one answered: see `SessionUsage`. */
  usage?: SessionUsage;
};

/**
 * Tokens and dollars of a session's assistant answers plus the reviews a factory task recorded (`factory-review`, run
 * in a child of its own). `input` counts cache reads and writes too: everything the models read.
 */
export type SessionUsage = { input: number; output: number; cost: number };

/** `profile` marks the profile a session was started with (G5); absent means `coding`. */
export type PrivateState = { revision: number; pinned: boolean; data: Record<string, unknown>; profile?: Profile };

export type MoveResult = { ok: true; path: string } | { ok: false; error: "locked" | "exists" | "not_in_root" | "not_found" };

export type DeleteResult = { ok: true } | { ok: false; error: "locked" | "not_found" };

/** The entry a fork ends at (inclusive) in a source session file; the source's leaf makes a clone. */
export type ForkRequest = { sessionPath: string; entryId: string };

/** Path of the new session file, or why there is none (`not_found`: no source session, `no_entry`: no such entry). */
export type ForkResult = { ok: true; path: string } | { ok: false; error: "not_found" | "no_entry" };

/** One session matching a search: where the best hit sits and the text around it. */
export type SearchMatch = {
  sessionPath: string;
  /** Id of the entry holding the hit, empty when the file has no entry ids. */
  entryId: string;
  snippet: string;
  score: number;
};

export interface Host {
  /** Machine id in every `SessionInfo`, `local` for this machine. */
  readonly machine: string;
  piVersion(): Promise<string>;
  spawn(request: SpawnRequest): Promise<PiChild>;
  lockSession(sessionPath: string): Promise<LockResult>;
  unlockSession(lockId: string): Promise<void>;
  isDirectory(path: string): Promise<boolean>;
  /** Sessions whose header cwd is `cwd` (every cwd when null), newest first. */
  listSessions(cwd: string | null, archived: boolean): Promise<SessionMeta[]>;
  /** Undefined when the file is missing or not a Pi session. */
  sessionMeta(sessionPath: string): Promise<SessionMeta | undefined>;
  /** Path of the session with Pi id `sessionId`, live root first. */
  findSession(sessionId: string): Promise<string | undefined>;
  /** Every entry of the session file in append order, the last one the leaf; empty when the file is unreadable. */
  readEntries(sessionPath: string): Promise<SessionEntries>;
  /** Move between the live and archive roots under the session's lock. */
  moveSession(sessionPath: string, toArchive: boolean): Promise<MoveResult>;
  /** Remove the session file and its sidecar under the session's lock. */
  deleteSession(sessionPath: string): Promise<DeleteResult>;
  /**
   * New session in the live root of the source's cwd: new id, `parentSession` = source path, the entries from the
   * root to `entryId`. The source file stays unchanged.
   */
  forkSession(request: ForkRequest): Promise<ForkResult>;
  /** Defaults when there is no sidecar; other read errors throw. */
  readState(sessionPath: string): Promise<PrivateState>;
  writeState(sessionPath: string, state: PrivateState): Promise<void>;
  /** Relative paths of the cwd's candidate files. */
  listFiles(cwd: string): Promise<string[]>;
  /** Live sessions whose transcript (user and assistant text) matches `query`, best first, newest first on ties. */
  searchSessions(query: string, limit: number): Promise<SearchMatch[]>;
  /**
   * Call `onChange` with the session files changed in the live root, at most once per debounce window; an empty
   * list means the changed files are unknown (the watch overflowed). Returns the function that stops watching.
   */
  watchSessions(onChange: (sessionPaths: string[]) => void): () => void;
}
