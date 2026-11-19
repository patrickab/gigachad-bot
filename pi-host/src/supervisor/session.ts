/**
 * One live Pi session: a `pi --mode rpc` child, its attached clients and the state replayed on attach.
 *
 * - Consumes: the child's JSONL stdout and stderr (Pi `rpc.md`), commands from attached clients, the session lock
 * - Provides: `LiveSession` with attach (snapshot + replay, M2), client commands (M21), extension UI state (M3),
 *   liveness (M17), live status and orderly close
 *
 * Snapshot consistency: the replay buffer is copied in the same synchronous step that handles the `get_entries`
 * response, so the entries plus the replayed records are exactly what Pi had emitted up to that point. Records after
 * it are queued for the attaching client until its `attached` frame is out.
 */
import { existsSync } from "node:fs";
import { tryLockFile, type FileLock } from "./lock.ts";
import type { PiChild, Profile } from "./spawn.ts";

/** Anything a frame can be sent to, a `ServerWebSocket` in production. */
export interface FrameSink {
  send(data: string): unknown;
}

export type PiRecord = { type?: unknown; id?: unknown; [field: string]: unknown };
export type LiveStatus = { running: boolean; waiting: boolean; attached: number };

/** Ids of commands pi-host sends itself; their responses never reach clients. */
const INTERNAL_ID_PREFIX = "pi-host:";
/** An extension keeps its idle session alive by setting a non-empty status under this key (M17). */
export const LIVENESS_STATUS_KEY = "session-liveness";
const DIALOG_METHODS = new Set(["select", "confirm", "input", "editor"]);
/** Commands that would move the child to another session file, breaking M1 locking and id addressing. */
const SESSION_SWITCHING_COMMANDS = new Set(["new_session", "switch_session", "fork", "clone"]);
/** High-rate records that cannot create the session file, skipped by the late-lock check. */
const STREAMING_RECORDS = new Set(["message_update", "tool_execution_update", "bash_execution_update"]);
const REQUEST_TIMEOUT_MS = 30_000;
const CLOSE_GRACE_MS = 10_000;
const STDERR_TAIL_LINES = 20;

type BufferedRecord = { line: string; toolCallId?: string };
type Dialog = { record: PiRecord; timer?: Timer };
type UiState = { dialogs: PiRecord[]; statuses: Record<string, string>; queue: Queue };
type Queue = { steering: unknown[]; followUp: unknown[] };
type PendingAttach = {
  state?: unknown;
  entries?: unknown;
  ui?: UiState;
  replay?: string[];
  /** Frames emitted after the snapshot, sent once `attached` and the replay are out. */
  queued: string[];
};

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
  private readonly pending = new Map<FrameSink, PendingAttach>();
  private readonly dialogs = new Map<string, Dialog>();
  private readonly statuses = new Map<string, string>();
  private queue: Queue = { steering: [], followUp: [] };
  private replay: BufferedRecord[] = [];
  private running = false;
  private compacting = false;
  private readonly requests = new Map<string, (record: PiRecord) => void>();
  private requestCounter = 0;
  private stderrLines: string[] = [];
  private lockFailed = false;

  constructor(
    private readonly child: PiChild,
    readonly cwd: string,
    readonly profile: Profile,
    private lock: FileLock | undefined,
    private readonly onExit: (session: LiveSession) => void,
  ) {
    this.exited = Promise.all([child.exited, this.readStdout(), this.readStderr()]).then(([code]) => {
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
    const timer = setTimeout(() => {
      if (this.requests.delete(id)) onResponse({ type: "response", success: false, error: "pi did not answer" });
    }, REQUEST_TIMEOUT_MS);
    this.requests.set(id, (record) => {
      clearTimeout(timer);
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

  /** Start attaching a client: snapshot (`get_state`, `get_entries`, UI state), replay, then live records. */
  attach(client: FrameSink): void {
    this.clients.delete(client);
    const pending: PendingAttach = { queued: [] };
    this.pending.set(client, pending);
    const fail = (record: PiRecord) => {
      if (this.pending.get(client) !== pending) return;
      this.pending.delete(client);
      this.deliver(client, envelope("error", this.sessionId, { message: `attach failed: ${String(record.error ?? "unknown")}` }));
    };
    const finish = () => {
      if (this.pending.get(client) !== pending || pending.state === undefined || pending.replay === undefined) return;
      this.pending.delete(client);
      this.deliver(client, envelope("attached", this.sessionId, { state: pending.state, entries: pending.entries, ui: pending.ui }));
      this.deliver(client, envelope("replay_start", this.sessionId));
      for (const line of pending.replay) this.deliver(client, piFrame(this.sessionId, line));
      this.deliver(client, envelope("replay_end", this.sessionId));
      for (const frame of pending.queued) this.deliver(client, frame);
      this.clients.add(client);
    };
    this.request({ type: "get_state" }, (record) => {
      if (record.success !== true) return fail(record);
      pending.state = record.data ?? {};
      finish();
    });
    this.request({ type: "get_entries" }, (record) => {
      if (record.success !== true) return fail(record);
      pending.entries = record.data;
      pending.ui = this.uiState();
      pending.replay = this.replay.map((buffered) => buffered.line);
      finish();
    });
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

  /** Send one frame to a client, stamped with this child's `streamId` and the client's next `sequence`. */
  deliver(client: FrameSink, frame: string): void {
    const sequence = (this.sequences.get(client) ?? -1) + 1;
    this.sequences.set(client, sequence);
    client.send(stamp(frame, this.streamId, sequence));
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

  /** Start an orderly stop without waiting (pi-host shutdown: children persist and exit on their own). */
  terminate(): void {
    this.closing = true;
    try {
      this.child.stdin.end();
    } catch {
      // stdin already closed
    }
    this.child.kill("SIGTERM");
  }

  private write(record: PiRecord): boolean {
    if (this.exitCode !== undefined) return false;
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
      queue: this.queue,
    };
  }

  private broadcast(frame: string): void {
    for (const client of this.clients) this.deliver(client, frame);
    for (const pending of this.pending.values()) if (pending.replay) pending.queued.push(frame);
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
      this.stderrLines = [...this.stderrLines, `non-JSON stdout: ${line.slice(0, 200)}`].slice(-STDERR_TAIL_LINES);
      return;
    }
    if (typeof record !== "object" || record === null) return;
    this.lastActivity = Date.now();
    const type = String(record.type);
    if (type === "response" && typeof record.id === "string" && record.id.startsWith(INTERNAL_ID_PREFIX)) {
      this.settleRequest(record.id, record);
      return;
    }
    this.track(type, record, line);
    this.broadcast(piFrame(this.sessionId, line));
    if (!STREAMING_RECORDS.has(type)) this.lockCreatedFile();
  }

  /** Update run, UI and replay state from one stdout record. */
  private track(type: string, record: PiRecord, line: string): void {
    switch (type) {
      case "response":
        return;
      case "extension_ui_request":
        this.trackUi(record);
        return;
      case "queue_update":
        this.queue = { steering: asArray(record.steering), followUp: asArray(record.followUp) };
        return;
      case "agent_start":
        this.running = true;
        break;
      case "agent_settled":
        this.running = false;
        this.replay = [];
        return;
      case "compaction_start":
        this.compacting = true;
        break;
      case "compaction_end":
        this.compacting = false;
        break;
      case "message_end":
        // The message is persisted now, `get_entries` returns it.
        this.replay = [];
        return;
      case "tool_execution_update": {
        // `partialResult` is the latest whole partial, older updates of the same call add nothing.
        const toolCallId = String(record.toolCallId);
        this.replay = this.replay.filter((buffered) => buffered.toolCallId !== toolCallId);
        this.replay.push({ line, toolCallId });
        return;
      }
    }
    this.replay.push({ line });
  }

  private trackUi(record: PiRecord): void {
    const method = String(record.method);
    if (DIALOG_METHODS.has(method) && typeof record.id === "string") {
      const id = record.id;
      const timeout = typeof record.timeout === "number" && record.timeout > 0 ? record.timeout : undefined;
      const timer = timeout === undefined ? undefined : setTimeout(() => this.closeDialog(id), timeout);
      this.dialogs.set(id, { record, timer });
    } else if (method === "setStatus" && typeof record.statusKey === "string") {
      if (typeof record.statusText === "string" && record.statusText) this.statuses.set(record.statusKey, record.statusText);
      else this.statuses.delete(record.statusKey);
    }
  }

  /** Forget an answered or timed-out dialog and tell every client (`ui_closed`). */
  private closeDialog(id: string): void {
    const dialog = this.dialogs.get(id);
    if (!dialog) return;
    clearTimeout(dialog.timer);
    this.dialogs.delete(id);
    this.broadcast(envelope("ui_closed", this.sessionId, { id }));
  }

  /** M1 for new sessions: Pi writes the file lazily, the lock is taken once it exists. */
  private lockCreatedFile(): void {
    if (this.lock || this.lockFailed || !this.sessionPath || this.exitCode !== undefined) return;
    if (!existsSync(this.sessionPath)) return;
    this.lock = tryLockFile(this.sessionPath);
    if (this.lock) return;
    this.lockFailed = true;
    this.broadcast(envelope("lock_held", this.sessionId));
    this.terminate();
  }

  private handleExit(code: number): void {
    this.exitCode = code;
    this.lock?.release();
    for (const dialog of this.dialogs.values()) clearTimeout(dialog.timer);
    this.dialogs.clear();
    for (const id of [...this.requests.keys()]) {
      this.settleRequest(id, { type: "response", success: false, error: "pi exited" });
    }
    this.broadcast(envelope("exit", this.sessionId, { code, stderrTail: this.stderrTail }));
    this.clients.clear();
    this.pending.clear();
    this.sequences.clear();
    this.onExit(this);
  }
}
