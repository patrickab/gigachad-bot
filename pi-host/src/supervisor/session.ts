/**
 * One live Pi session: a `pi --mode rpc` child, its attached clients and the state replayed on attach and resync.
 *
 * - Consumes: the child's JSONL stdout and stderr (Pi `rpc.md`), commands from attached clients, the session lock
 * - Provides: `LiveSession` with attach and resync (snapshot + replay, M2), client commands (M21), extension UI state
 *   (M3), liveness (M17), live status and orderly close
 *
 * Snapshot consistency: the replay buffer is copied in the same synchronous step that handles the `get_entries`
 * response, so the entries plus the replayed records are exactly what Pi had emitted up to that point. Records after
 * it are queued for the client until its `attached` or `resync` frame is out. `get_state` is sent after that point,
 * so the state is never older than the entries; the queued records bring it forward.
 *
 * A client not yet attached gets its frames held from attach start: at the snapshot point the ones the snapshot
 * reproduces (entries, replay, UI state) are dropped, the rest (responses to its own commands, transient UI requests)
 * go out after the replay.
 */
import type { PiChild } from "../host/index.ts";
import { withMediaHashes } from "../media.ts";
import type { SessionLock } from "./lock.ts";
import type { ProfileSpec } from "./profiles.ts";

/** Anything a frame can be sent to, a `ServerWebSocket` in production. */
export interface FrameSink {
  /** Bun semantics: 0 means the frame was dropped (socket closed or over its backpressure limit). */
  send(data: string): unknown;
  close?(code?: number, reason?: string): void;
}

export type PiRecord = { type?: unknown; id?: unknown; [field: string]: unknown };
export type LiveStatus = { running: boolean; waiting: boolean; attached: number };
export type SessionEntry = { id: string; parentId: string | null; type: string; [field: string]: unknown };

/** Ids of commands pi-host sends itself; their responses never reach clients. */
const INTERNAL_ID_PREFIX = "pi-host:";
/** An extension keeps its idle session alive by setting a non-empty status under this key (M17). */
export const LIVENESS_STATUS_KEY = "session-liveness";
/** pi-subagents answers `/subagents-inspect-rpc` with one emit-then-retract widget on this key, the client reads it live. */
const TRANSIENT_WIDGET_KEY = "subagent-inspect";
const DIALOG_METHODS = new Set(["select", "confirm", "input", "editor"]);
/** Commands that would move the child to another session file, breaking M1 locking and id addressing. */
const SESSION_SWITCHING_COMMANDS = new Set(["new_session", "switch_session", "fork", "clone"]);
/** High-rate records that cannot create the session file, skipped by the late-lock check. */
const STREAMING_RECORDS = new Set(["message_update", "tool_execution_update", "bash_execution_update"]);
const REQUEST_TIMEOUT_MS = 30_000;
/** An RPC answer slower than this is logged with its command type, so a stalled start names its step. */
const SLOW_REQUEST_MS = 5_000;
const CLOSE_GRACE_MS = 10_000;
/** After the child exits, how long its stdout and stderr may still drain (a descendant can hold them open). */
const READER_DRAIN_MS = 1_000;
/** Replay buffer bound; the oldest records go first once it is exceeded. */
const REPLAY_MAX_BYTES = 16 * 1024 * 1024;
const STDERR_TAIL_LINES = 20;
/** Close code for a client that dropped a frame: it reconnects and resyncs. */
const CLOSE_TRY_AGAIN_LATER = 1013;

type BufferedRecord = { line: string; type: string; toolCallId?: string };
type Dialog = { record: PiRecord; timer?: Timer };
type UiState = { dialogs: PiRecord[]; statuses: Record<string, string>; widgets: Record<string, string[]>; queue: Queue };
type Queue = { steering: unknown[]; followUp: unknown[] };
type EntriesData = { entries: SessionEntry[]; leafId: string | null };
type PendingSnapshot = {
  /** The snapshot frame without `state`, set at the snapshot point. */
  frame?: Record<string, unknown>;
  replay?: string[];
  /** Frames for a client not yet attached, from attach start to the snapshot point; `covered` ones the snapshot reproduces. */
  early?: { frame: string; covered: boolean }[];
  /** Frames held until the snapshot frame and the replay are out. */
  queued: string[];
};

/** Root-to-leaf path through `entries` by `parentId`. */
export function branchOf(entries: SessionEntry[], leafId: string | null): SessionEntry[] {
  const byId = new Map(entries.map((entry) => [entry.id, entry]));
  const path: SessionEntry[] = [];
  for (let id = leafId; id !== null; ) {
    const entry = byId.get(id);
    if (!entry) break;
    byId.delete(id);
    path.push(entry);
    id = entry.parentId;
  }
  return path.reverse();
}

/**
 * Entries to append after `cursor`, given the `get_entries` tail after it: the leaf's ancestors back to the cursor.
 * Undefined when the cursor is not an ancestor of the leaf or a compaction follows it (M2 replace rule).
 */
export function appendAfter(cursor: string, tail: SessionEntry[], leafId: string | null): SessionEntry[] | undefined {
  const byId = new Map(tail.map((entry) => [entry.id, entry]));
  const path: SessionEntry[] = [];
  let id = leafId;
  while (id !== null && id !== cursor) {
    const entry = byId.get(id);
    if (!entry || entry.type === "compaction") return undefined;
    byId.delete(id);
    path.push(entry);
    id = entry.parentId;
  }
  return id === cursor ? path.reverse() : undefined;
}

/** `{type:"pi"}` frame around a raw stdout record, spliced so the record stays byte-identical (M16). */
function piFrame(sessionId: string, line: string): string {
  return `{"type":"pi","sessionId":${JSON.stringify(sessionId)},"record":${line}}`;
}

/** Envelope frame for a session. */
export function envelope(type: string, sessionId: string, fields: Record<string, unknown> = {}): string {
  return JSON.stringify({ type, sessionId, ...fields });
}

/**
 * Add `streamId` and `sequence` to a serialized frame (M16). A frame always ends with its object's `}`, so the
 * fields are spliced in front of it and the raw Pi record inside stays untouched. `streamId` is null for errors
 * about no live session.
 */
export function stamp(frame: string, streamId: string | null, sequence: number): string {
  return `${frame.slice(0, -1)},"streamId":${JSON.stringify(streamId)},"sequence":${sequence}}`;
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function entriesData(data: unknown): EntriesData {
  const record = typeof data === "object" && data !== null ? (data as Record<string, unknown>) : {};
  return {
    entries: asArray(record.entries) as SessionEntry[],
    leafId: typeof record.leafId === "string" ? record.leafId : null,
  };
}

export class LiveSession {
  sessionId = "";
  sessionPath = "";
  readonly clients = new Set<FrameSink>();
  /** New per child: the session identity never changes within one child (switching commands are refused). */
  readonly streamId = crypto.randomUUID();
  private readonly sequences = new Map<FrameSink, number>();
  readonly exited: Promise<number>;
  exitCode: number | undefined;
  closing = false;
  lastActivity = Date.now();
  private readonly pending = new Map<FrameSink, PendingSnapshot>();
  private readonly dialogs = new Map<string, Dialog>();
  private readonly statuses = new Map<string, string>();
  /** Open `setWidget` text widgets by key (the subagent roster's source). Replies on the transient `subagent-inspect` key are not state. */
  private readonly widgets = new Map<string, string[]>();
  private queue: Queue = { steering: [], followUp: [] };
  private replay: BufferedRecord[] = [];
  private replayBytes = 0;
  private running = false;
  private compacting = false;
  private readonly requests = new Map<string, (record: PiRecord) => void>();
  private requestCounter = 0;
  private stderrLines: string[] = [];
  private lockFailed = false;
  private locking = false;

  constructor(
    private readonly child: PiChild,
    readonly cwd: string,
    /** What the child was started with; a request for another profile or prompt gets `profile_mismatch`. */
    readonly spec: ProfileSpec,
    private readonly lock: SessionLock | undefined,
    private readonly onExit: (session: LiveSession) => void,
    /** Called with true when a run starts, false when it settles or a client aborts it (auto-resume bookkeeping). */
    private readonly onRun: (session: LiveSession, active: boolean) => void = () => undefined,
  ) {
    const readers = Promise.all([
      this.readStdout().catch((error) => this.noteStderr(`stdout reader failed: ${String(error)}`)),
      this.readStderr().catch((error) => this.noteStderr(`stderr reader failed: ${String(error)}`)),
    ]);
    // Exit follows the child alone: a descendant holding the pipes must not keep the session and its lock.
    this.exited = child.exited.then(async (code) => {
      await Promise.race([readers, Bun.sleep(READER_DRAIN_MS)]);
      this.handleExit(code);
      return code;
    });
  }

  /** Record the Pi session id and file once `get_state` named them. */
  identify(sessionId: string, sessionPath: string): void {
    this.sessionId = sessionId;
    this.sessionPath = sessionPath;
    this.lockCreatedFile();
  }

  get stderrTail(): string {
    return this.stderrLines.join("\n");
  }

  liveStatus(): LiveStatus {
    return { running: this.running || this.compacting, waiting: this.dialogs.size > 0, attached: this.clients.size };
  }

  /** True when nothing needs the child: no client, no run, no dialog, no liveness status, idle long enough. */
  isIdle(now: number, idleMs: number): boolean {
    return (
      this.exitCode === undefined &&
      !this.closing &&
      this.clients.size === 0 &&
      this.pending.size === 0 &&
      !this.running &&
      !this.compacting &&
      this.dialogs.size === 0 &&
      !this.statuses.get(LIVENESS_STATUS_KEY) &&
      now - this.lastActivity >= idleMs
    );
  }

  /** Send a command and call `onResponse` synchronously with its response record (or a synthetic failure). */
  request(command: PiRecord, onResponse: (record: PiRecord) => void): void {
    const id = `${INTERNAL_ID_PREFIX}${++this.requestCounter}`;
    const started = performance.now();
    const which = () => `${String(command.type)} to pi ${this.sessionId || "(starting)"}`;
    const timer = setTimeout(() => {
      if (!this.requests.delete(id)) return;
      console.warn(`pi-host: ${which()} got no answer in ${REQUEST_TIMEOUT_MS / 1000} s`);
      onResponse({ type: "response", success: false, error: "pi did not answer" });
    }, REQUEST_TIMEOUT_MS);
    this.requests.set(id, (record) => {
      clearTimeout(timer);
      const elapsed = performance.now() - started;
      if (elapsed > SLOW_REQUEST_MS) console.warn(`pi-host: ${which()} took ${(elapsed / 1000).toFixed(1)} s`);
      onResponse(record);
    });
    if (!this.write({ ...command, id })) this.settleRequest(id, { type: "response", success: false, error: "pi is not running" });
  }

  /** `request` as a promise, for callers that do not need the synchronous snapshot point. */
  requestAsync(command: PiRecord): Promise<PiRecord> {
    const { promise, resolve } = Promise.withResolvers<PiRecord>();
    this.request(command, resolve);
    return promise;
  }

  /** Attach a client: `attached` (state, all entries, UI state), replay, then live records. */
  attach(client: FrameSink): void {
    this.snapshot(client, "attached", (pending, done) => {
      this.request({ type: "get_entries" }, (record) => {
        if (record.success !== true) return done(record);
        this.markSnapshot(client, pending, { entries: record.data });
        done();
      });
    });
  }

  /**
   * Bring a client up to date from `cursor` (M2), attaching it when it was not: `resync` with `append` (the branch
   * entries after the cursor) or `replace` (the whole branch), then replay and live records like `attach`.
   */
  resync(client: FrameSink, cursor: string | null): void {
    this.snapshot(client, "resync", (pending, done) => {
      const replaceAll = () =>
        this.request({ type: "get_entries" }, (record) => {
          if (record.success !== true) return done(record);
          const { entries, leafId } = entriesData(record.data);
          this.markSnapshot(client, pending, { mode: "replace", entries: branchOf(entries, leafId), leafId });
          done();
        });
      if (cursor === null) return replaceAll();
      // Pi answers an unknown `since` with a failure: the full branch replaces then.
      this.request({ type: "get_entries", since: cursor }, (record) => {
        if (record.success !== true) return replaceAll();
        const { entries, leafId } = entriesData(record.data);
        const appended = appendAfter(cursor, entries, leafId);
        if (!appended) return replaceAll();
        this.markSnapshot(client, pending, { mode: "append", entries: appended, leafId });
        done();
      });
    });
  }

  /**
   * Take every attached and attaching client off this child without telling them (no `exit` frame), so a restart on
   * the same file can carry them over to the next child (a turn delete).
   */
  takeClients(): FrameSink[] {
    const taken = [...new Set([...this.clients, ...this.pending.keys()])];
    this.clients.clear();
    this.pending.clear();
    this.sequences.clear();
    return taken;
  }

  /** Stop sending to a client. */
  detach(client: FrameSink): void {
    const wasAttached = this.clients.delete(client) || this.pending.delete(client);
    if (wasAttached) this.lastActivity = Date.now();
  }

  /** Detach a client whose socket closed and drop its sequence counter. */
  disconnect(client: FrameSink): void {
    this.detach(client);
    this.sequences.delete(client);
  }

  /**
   * Send one frame to a client, stamped with this child's `streamId` and the client's next `sequence`. A dropped
   * frame closes the client's socket: a gap would go unnoticed, a reconnect resyncs.
   */
  deliver(client: FrameSink, frame: string): void {
    if (this.hold(client, frame, false)) return;
    const sequence = (this.sequences.get(client) ?? -1) + 1;
    this.sequences.set(client, sequence);
    if (client.send(stamp(frame, this.streamId, sequence)) !== 0) return;
    this.disconnect(client);
    client.close?.(CLOSE_TRY_AGAIN_LATER, "frame dropped");
  }

  /** Forward a client's raw Pi command (M21: any attached client), an error message when it is refused. */
  command(client: FrameSink, command: unknown): string | undefined {
    if (!this.clients.has(client) && !this.pending.has(client)) return "not attached";
    if (typeof command !== "object" || command === null || Array.isArray(command)) return "command must be an object";
    const record = command as PiRecord;
    if (typeof record.type !== "string") return "command needs a type";
    if (SESSION_SWITCHING_COMMANDS.has(record.type)) {
      return `${record.type} would switch the session file; start or resume sessions with POST /sessions`;
    }
    if (!this.write(record)) return "pi is not running";
    this.lastActivity = Date.now();
    // An aborted run is the user's decision, never resumed after a crash
    if (record.type === "abort") this.onRun(this, false);
    if (record.type === "extension_ui_response" && typeof record.id === "string") this.closeDialog(record.id);
    return undefined;
  }

  /** Orderly stop: stdin EOF and SIGTERM, SIGKILL after a grace period. Resolves once the child is gone. */
  async close(graceMs = CLOSE_GRACE_MS): Promise<void> {
    if (this.exitCode !== undefined) return;
    this.terminate();
    const killer = setTimeout(() => this.child.kill("SIGKILL"), graceMs);
    await this.exited;
    clearTimeout(killer);
  }

  /**
   * Start an orderly stop without waiting (pi-host shutdown: children persist and exit on their own). Only the first
   * call signals: Pi treats a second SIGTERM during shutdown as "skip cleanup".
   */
  terminate(): void {
    if (this.closing) return;
    this.closing = true;
    try {
      this.child.stdin.end();
    } catch {
      // stdin already closed
    }
    this.child.kill("SIGTERM");
  }

  /**
   * Shared attach/resync flow. `take` requests the snapshot and calls `markSnapshot` at its point, then `done`;
   * `get_state` follows and the frame goes out with it. An attached client keeps its live frames until the
   * snapshot point, an unattached one is attached by this.
   */
  private snapshot(
    client: FrameSink,
    type: "attached" | "resync",
    take: (pending: PendingSnapshot, done: (failure?: PiRecord) => void) => void,
  ): void {
    const pending: PendingSnapshot = { queued: [], ...(this.clients.has(client) ? {} : { early: [] }) };
    this.pending.set(client, pending);
    const fail = (record: PiRecord) => {
      if (this.pending.get(client) !== pending) return;
      this.pending.delete(client);
      this.clients.delete(client);
      this.deliver(client, envelope("error", this.sessionId, { message: `${type === "attached" ? "attach" : "resync"} failed: ${String(record.error ?? "unknown")}` }));
    };
    take(pending, (failure) => {
      if (failure) return fail(failure);
      this.request({ type: "get_state" }, (record) => {
        if (this.pending.get(client) !== pending || !pending.frame || !pending.replay) return;
        if (record.success !== true) return fail(record);
        this.pending.delete(client);
        this.deliver(client, envelope(type, this.sessionId, { state: record.data ?? {}, ...pending.frame }));
        this.deliver(client, envelope("replay_start", this.sessionId));
        for (const line of pending.replay) this.deliver(client, piFrame(this.sessionId, line));
        this.deliver(client, envelope("replay_end", this.sessionId));
        for (const frame of pending.queued) this.deliver(client, frame);
        if (this.exitCode === undefined && this.sequences.has(client)) this.clients.add(client);
      });
    });
  }

  /** The snapshot point: copy replay and UI state, and queue later frames for this client instead of sending them. */
  private markSnapshot(client: FrameSink, pending: PendingSnapshot, frame: Record<string, unknown>): void {
    if (this.pending.get(client) !== pending) return;
    this.clients.delete(client);
    pending.frame = { ...frame, ui: this.uiState() };
    pending.replay = this.replay.map((buffered) => buffered.line);
    pending.queued = (pending.early ?? []).filter((held) => !held.covered).map((held) => held.frame);
    pending.early = undefined;
  }

  /** Hold a frame for a client whose snapshot is in progress, false when it is sent now (attached, or done). */
  private hold(client: FrameSink, frame: string, covered: boolean): boolean {
    const pending = this.pending.get(client);
    if (pending?.replay) pending.queued.push(frame);
    else if (pending?.early) pending.early.push({ frame, covered });
    else return false;
    return true;
  }

  private write(record: PiRecord): boolean {
    if (this.exitCode !== undefined || this.closing) return false;
    try {
      this.child.stdin.write(`${JSON.stringify(record)}\n`);
      this.child.stdin.flush();
      return true;
    } catch {
      return false;
    }
  }

  /** Hand a response (real or synthetic) to the pi-host request waiting for it. */
  private settleRequest(id: string, record: PiRecord): void {
    const callback = this.requests.get(id);
    this.requests.delete(id);
    callback?.(record);
  }

  private uiState(): UiState {
    return {
      dialogs: [...this.dialogs.values()].map((dialog) => dialog.record),
      statuses: Object.fromEntries(this.statuses),
      widgets: Object.fromEntries(this.widgets),
      queue: this.queue,
    };
  }

  /** Send a frame to attached clients and hold it for snapshotting ones; `covered` when a snapshot reproduces it. */
  private broadcast(frame: string, covered: boolean): void {
    for (const client of this.clients) this.deliver(client, frame);
    for (const client of this.pending.keys()) if (!this.clients.has(client)) this.hold(client, frame, covered);
  }

  private noteStderr(line: string): void {
    this.stderrLines = [...this.stderrLines, line].slice(-STDERR_TAIL_LINES);
  }

  private async readStdout(): Promise<void> {
    const decoder = new TextDecoder();
    let rest = "";
    for await (const chunk of this.child.stdout) {
      const text = rest + decoder.decode(chunk, { stream: true });
      let start = 0;
      let newline: number;
      while ((newline = text.indexOf("\n", start)) >= 0) {
        const line = text.slice(start, text[newline - 1] === "\r" ? newline - 1 : newline);
        start = newline + 1;
        if (line) this.handleLine(line);
      }
      rest = text.slice(start);
    }
  }

  private async readStderr(): Promise<void> {
    const decoder = new TextDecoder();
    for await (const chunk of this.child.stderr) {
      const lines = decoder.decode(chunk, { stream: true }).split("\n").filter(Boolean);
      this.stderrLines = [...this.stderrLines, ...lines].slice(-STDERR_TAIL_LINES);
    }
  }

  private handleLine(line: string): void {
    let record: PiRecord;
    try {
      record = JSON.parse(line);
    } catch {
      this.noteStderr(`non-JSON stdout: ${line.slice(0, 200)}`);
      return;
    }
    if (typeof record !== "object" || record === null) return;
    // Tool-result images reach clients (and the replay buffer) as media hashes, not base64. The one record that
    // contains an image is re-serialized, every other line stays byte-identical (M16).
    if (line.includes('"image"')) {
      const hashed = withMediaHashes(record);
      if (hashed !== record) {
        record = hashed;
        line = JSON.stringify(record);
      }
    }
    this.lastActivity = Date.now();
    const type = String(record.type);
    if (type === "response" && typeof record.id === "string" && record.id.startsWith(INTERNAL_ID_PREFIX)) {
      this.settleRequest(record.id, record);
      return;
    }
    const covered = this.track(type, record, line);
    this.broadcast(piFrame(this.sessionId, line), covered);
    if (!STREAMING_RECORDS.has(type)) this.lockCreatedFile();
  }

  /** Update run, UI and replay state from one stdout record; true when a snapshot reproduces the record. */
  private track(type: string, record: PiRecord, line: string): boolean {
    switch (type) {
      case "response":
        // A bash command's result is an entry now, its streamed output is history.
        if (record.command === "bash") this.dropReplay((buffered) => buffered.type === "bash_execution_update");
        return false;
      case "entry_appended":
        // The entry is persisted, `get_entries` returns it.
        return true;
      case "extension_ui_request":
        return this.trackUi(record);
      case "queue_update":
        this.queue = { steering: asArray(record.steering), followUp: asArray(record.followUp) };
        return true;
      case "agent_start":
        this.running = true;
        this.onRun(this, true);
        break;
      case "agent_settled":
        this.running = false;
        this.onRun(this, false);
        this.dropReplay(() => true);
        return true;
      case "compaction_start":
        this.compacting = true;
        break;
      case "compaction_end":
        this.compacting = false;
        break;
      case "message_end":
        // The message is persisted now, `get_entries` returns it.
        this.dropReplay(() => true);
        return true;
      case "tool_execution_update": {
        // `partialResult` is the latest whole partial, older updates of the same call add nothing.
        const toolCallId = String(record.toolCallId);
        this.dropReplay((buffered) => buffered.toolCallId === toolCallId);
        this.bufferReplay({ line, type, toolCallId });
        return true;
      }
    }
    this.bufferReplay({ line, type });
    return true;
  }

  private bufferReplay(buffered: BufferedRecord): void {
    this.replay.push(buffered);
    this.replayBytes += buffered.line.length;
    let dropped = 0;
    while (this.replayBytes > REPLAY_MAX_BYTES && dropped < this.replay.length - 1) {
      this.replayBytes -= this.replay[dropped]!.line.length;
      dropped++;
    }
    if (dropped) this.replay = this.replay.slice(dropped);
  }

  private dropReplay(matches: (buffered: BufferedRecord) => boolean): void {
    this.replay = this.replay.filter((buffered) => !matches(buffered));
    this.replayBytes = this.replay.reduce((total, buffered) => total + buffered.line.length, 0);
  }

  /** Track dialogs and statuses, true for the UI requests the snapshot's UI state carries. */
  private trackUi(record: PiRecord): boolean {
    const method = String(record.method);
    if (DIALOG_METHODS.has(method) && typeof record.id === "string") {
      const id = record.id;
      const timeout = typeof record.timeout === "number" && record.timeout > 0 ? record.timeout : undefined;
      const timer = timeout === undefined ? undefined : setTimeout(() => this.closeDialog(id), timeout);
      this.dialogs.set(id, { record, timer });
      return true;
    }
    if (method === "setStatus" && typeof record.statusKey === "string") {
      if (typeof record.statusText === "string" && record.statusText) this.statuses.set(record.statusKey, record.statusText);
      else this.statuses.delete(record.statusKey);
      return true;
    }
    if (method === "setWidget" && typeof record.widgetKey === "string") {
      if (record.widgetKey === TRANSIENT_WIDGET_KEY) return false;
      const lines = Array.isArray(record.widgetLines) ? record.widgetLines.filter((line): line is string => typeof line === "string") : [];
      if (lines.length > 0) this.widgets.set(record.widgetKey, lines);
      else this.widgets.delete(record.widgetKey);
      return true;
    }
    return false;
  }

  /** Forget an answered or timed-out dialog and tell every client (`ui_closed`). */
  private closeDialog(id: string): void {
    const dialog = this.dialogs.get(id);
    if (!dialog) return;
    clearTimeout(dialog.timer);
    this.dialogs.delete(id);
    this.broadcast(envelope("ui_closed", this.sessionId, { id }), true);
  }

  /** M1 for new sessions: Pi writes the file lazily, the lock is taken once it exists (a missing file retries). */
  private lockCreatedFile(): void {
    const lock = this.lock;
    if (!lock || lock.held || this.locking || this.lockFailed || !this.sessionPath || this.exitCode !== undefined) return;
    this.locking = true;
    lock.take(this.sessionPath).then(
      (outcome) => {
        this.locking = false;
        if (outcome !== "busy" || this.exitCode !== undefined) return;
        this.lockFailed = true;
        this.broadcast(envelope("lock_held", this.sessionId), false);
        this.terminate();
      },
      (error) => {
        this.locking = false;
        this.noteStderr(`session lock failed: ${String(error)}`);
      },
    );
  }

  private handleExit(code: number): void {
    this.exitCode = code;
    this.lock?.release();
    for (const dialog of this.dialogs.values()) clearTimeout(dialog.timer);
    this.dialogs.clear();
    for (const id of [...this.requests.keys()]) {
      this.settleRequest(id, { type: "response", success: false, error: "pi exited" });
    }
    this.broadcast(envelope("exit", this.sessionId, { code, stderrTail: this.stderrTail }), false);
    this.clients.clear();
    this.pending.clear();
    this.sequences.clear();
    this.onExit(this);
  }
}
