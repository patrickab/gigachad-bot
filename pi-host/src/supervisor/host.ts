/**
 * The supervisor behind `createSupervisor()`: live session registry, `POST /sessions`, `/ws` multiplexing, eviction,
 * and the `sessions_changed` broadcast (M25).
 *
 * - Consumes: `SupervisorConfig`, the `Host` (spawn, session lock, session metadata, fork, session watcher), Catalog's
 *   `sessionInfo`, `copyPrivateState` and the profile marker (G5), `profiles.ts` for each profile's Pi arguments
 * - Provides: `SupervisorHost`, the `Supervisor` implementation (HTTP rows owned by Supervisor, WS `/ws`, M18)
 */
import { isAbsolute, resolve } from "node:path";
import type { ServerWebSocket, WebSocketHandler } from "bun";
import { copyPrivateState, markProfile, profileOf, sessionInfo, type SessionInfo } from "../catalog/index.ts";
import { isPlainObject as isRecord } from "../catalog/state.ts";
import type { Host, Profile, SessionMeta } from "../host/index.ts";
import type { SupervisorConfig } from "./config.ts";
import type { Health, SocketData, Supervisor } from "./index.ts";
import { SessionLock } from "./lock.ts";
import { parseProfileSpec, profileArgs, sameProfile, type ProfileSpec } from "./profiles.ts";
import { envelope, LiveSession, stamp, type LiveStatus } from "./session.ts";

type ForkOf = { sessionPath: string; entryId: string };
type Resume = null | "continue" | { sessionPath: string } | { sessionId: string } | { forkOf: ForkOf };
type StartRequest = {
  cwd: string;
  spec: ProfileSpec;
  resume: Resume;
  model?: { provider: string; id: string };
  thinking?: string;
};
type Failure = { status: number; body: Record<string, unknown> };
type StartOutcome = { session: LiveSession } | Failure;

const SESSION_CLOSE = /^\/sessions\/([^/]+)\/close$/;
const MAX_EVICTION_INTERVAL_MS = 60_000;
const MIN_EVICTION_INTERVAL_MS = 50;
/** Far above any snapshot (omp sessions reach 27 MB): a healthy client never hits it, a stalled one is closed. */
const BACKPRESSURE_LIMIT_BYTES = 512 * 1024 * 1024;

function fail(status: number, error: string, extra: Record<string, unknown> = {}): Failure {
  return { status, body: { error, ...extra } };
}

/** Validated `POST /sessions` body, or the failure to answer with. */
async function parseStartRequest(body: unknown, host: Host): Promise<StartRequest | Failure> {
  if (!isRecord(body)) return fail(400, "invalid_body");
  const { cwd, profile, systemPrompt, tools, resume, model, thinking } = body;
  if (typeof cwd !== "string" || !isAbsolute(cwd) || !(await host.isDirectory(cwd))) {
    return fail(400, "invalid_cwd");
  }
  const spec = parseProfileSpec({ profile, systemPrompt, tools });
  if ("error" in spec) return fail(400, spec.error);
  const validResume =
    resume === null ||
    resume === undefined ||
    resume === "continue" ||
    (isRecord(resume) && typeof resume.sessionPath === "string" && isAbsolute(resume.sessionPath)) ||
    (isRecord(resume) && typeof resume.sessionId === "string" && resume.sessionId !== "") ||
    (isRecord(resume) &&
      isRecord(resume.forkOf) &&
      typeof resume.forkOf.sessionPath === "string" &&
      isAbsolute(resume.forkOf.sessionPath) &&
      typeof resume.forkOf.entryId === "string" &&
      resume.forkOf.entryId !== "");
  if (!validResume) return fail(400, "invalid_resume");
  if (model !== undefined && !(isRecord(model) && typeof model.provider === "string" && typeof model.id === "string")) {
    return fail(400, "invalid_model");
  }
  if (thinking !== undefined && typeof thinking !== "string") return fail(400, "invalid_thinking");
  return {
    cwd: resolve(cwd),
    spec,
    resume: (resume ?? null) as Resume,
    model: model as StartRequest["model"],
    thinking: thinking as string | undefined,
  };
}

export class SupervisorHost implements Supervisor {
  private readonly byId = new Map<string, LiveSession>();
  private readonly byPath = new Map<string, LiveSession>();
  private readonly starting = new Map<string, Promise<StartOutcome>>();
  private readonly evictionTimer: Timer;
  private readonly sockets = new Set<ServerWebSocket<SocketData>>();
  private readonly stopWatching: () => void;
  private evicting = false;

  readonly websocket: WebSocketHandler<SocketData> = {
    open: (ws) => void this.sockets.add(ws),
    message: (ws, message) => this.onSocketMessage(ws, message),
    close: (ws) => {
      this.sockets.delete(ws);
      for (const session of this.byId.values()) session.disconnect(ws);
    },
    backpressureLimit: BACKPRESSURE_LIMIT_BYTES,
    // An overrun closes the socket instead of dropping frames silently; the client reconnects and resyncs.
    closeOnBackpressureLimit: true,
  };

  constructor(
    private readonly config: SupervisorConfig,
    private readonly piVersion: string,
    private readonly isPinned: (sessionPath: string) => Promise<boolean>,
    readonly host: Host,
  ) {
    const interval = Math.min(MAX_EVICTION_INTERVAL_MS, Math.max(MIN_EVICTION_INTERVAL_MS, config.idleMs / 4));
    this.evictionTimer = setInterval(() => void this.evictIdle(), interval);
    this.evictionTimer.unref();
    // A host frame for every connected client, attached or not: session lists change outside any session.
    this.stopWatching = host.watchSessions((sessionPaths) => {
      const frame = JSON.stringify({ type: "sessions_changed", sessionPaths });
      for (const ws of this.sockets) ws.send(frame);
    });
  }

  socketData(): SocketData {
    return { clientId: crypto.randomUUID() };
  }

  /** `GET /health` (redeploy guard, M4): every live session; a pin that cannot be read counts as pinned. */
  async health(): Promise<Health> {
    const sessions = await Promise.all(
      [...this.byId.values()].map(async (session) => {
        const { running, attached } = session.liveStatus();
        const pinned = await this.isPinned(session.sessionPath).catch(() => true);
        return { sessionId: session.sessionId, sessionPath: session.sessionPath, running, attached, pinned };
      }),
    );
    return { ok: true, piVersion: this.piVersion, sessions };
  }

  /** Live status by the path a session is keyed under: as Pi named a new file, with symlinks resolved for a resume. */
  liveStatus(sessionPath: string): LiveStatus | undefined {
    return this.byPath.get(resolve(sessionPath))?.liveStatus();
  }

  /** Archive/restore (C7): close a detached idle child through the orderly close path, never a running, attached or asking one. */
  async closeIfIdle(sessionPath: string): Promise<boolean> {
    const session = this.byPath.get(resolve(sessionPath));
    if (!session) return true;
    if (!session.closing && !session.isIdle(Date.now(), 0)) return false;
    await session.close();
    return true;
  }

  async handle(request: Request, url: URL): Promise<Response | undefined> {
    if (url.pathname === "/sessions" && request.method === "POST") {
      const parsed = await parseStartRequest(await request.json().catch(() => undefined), this.host);
      if ("status" in parsed) return Response.json(parsed.body, { status: parsed.status });
      const outcome = await this.start(parsed);
      if ("status" in outcome) return Response.json(outcome.body, { status: outcome.status });
      return Response.json({ session: await this.info(outcome.session) });
    }
    const close = SESSION_CLOSE.exec(url.pathname);
    if (close && request.method === "POST") {
      await this.byId.get(decodeURIComponent(close[1]!))?.close();
      return new Response(null, { status: 204 });
    }
    return undefined;
  }

  async shutdown(): Promise<void> {
    clearInterval(this.evictionTimer);
    this.stopWatching();
    for (const session of this.byId.values()) session.terminate();
  }

  /** Stop live sessions nobody needs (no client, not running, no dialog or liveness status, not pinned). */
  async evictIdle(now = Date.now()): Promise<void> {
    if (this.evicting) return;
    this.evicting = true;
    try {
      for (const session of [...this.byId.values()]) {
        if (!session.isIdle(now, this.config.idleMs)) continue;
        // Unreadable private state fails safe: the session stays.
        const pinned = await this.isPinned(session.sessionPath).catch(() => true);
        if (!pinned && session.isIdle(now, this.config.idleMs)) void session.close();
      }
    } finally {
      this.evicting = false;
    }
  }

  /** Contract `SessionInfo`; a new session without a file yet gets an empty summary. */
  private async info(session: LiveSession): Promise<SessionInfo> {
    const existing = await sessionInfo(this.host, session.sessionPath, this);
    if (existing) return existing;
    const now = new Date().toISOString();
    return {
      sessionId: session.sessionId,
      sessionPath: session.sessionPath,
      cwd: session.cwd,
      machine: this.host.machine,
      createdAt: now,
      modifiedAt: now,
      firstMessage: "(no messages)",
      messageCount: 0,
      live: session.liveStatus(),
      pinned: false,
      archived: false,
      profile: session.spec.profile,
    };
  }

  /**
   * Resolve the session file to resume (writing it first for a fork), then return its live child or spawn one. A
   * session started with another profile, or a live child with another system prompt or tools, is `profile_mismatch`.
   */
  private async start(request: StartRequest): Promise<StartOutcome> {
    const { resume, spec } = request;
    let meta: SessionMeta | undefined;
    let cwd = request.cwd;
    if (resume === "continue") {
      meta = await this.newestOf(cwd, spec.profile);
    } else if (resume && "forkOf" in resume) {
      const { forkOf } = resume;
      if ((await profileOf(this.host, forkOf.sessionPath)) !== spec.profile) return fail(409, "profile_mismatch");
      const forked = await this.host.forkSession(forkOf);
      if (!forked.ok) return fail(404, forked.error);
      await copyPrivateState(this.host, forkOf.sessionPath, forked.path);
      meta = await this.host.sessionMeta(forked.path);
      if (!meta) return fail(404, "not_found");
      cwd = meta.cwd;
    } else if (resume) {
      // A new session has no file before its first message, only its live child knows the id.
      const liveById = "sessionId" in resume ? this.byId.get(resume.sessionId) : undefined;
      if (liveById && !liveById.closing) return this.reuse(liveById, spec);
      const sessionPath = "sessionId" in resume ? await this.host.findSession(resume.sessionId) : resume.sessionPath;
      meta = sessionPath === undefined ? undefined : await this.host.sessionMeta(sessionPath);
      if (!meta) return fail(404, "not_found");
      if (meta.archived) return fail(409, "archived");
      if ((await profileOf(this.host, meta.sessionPath)) !== spec.profile) return fail(409, "profile_mismatch");
      cwd = meta.cwd;
    }
    if (!meta) return this.spawn(request, cwd, undefined);
    const sessionPath = meta.realPath;

    const live = this.byPath.get(sessionPath) ?? this.byPath.get(meta.sessionPath);
    if (live && !live.closing) return this.reuse(live, spec);
    if (live) await live.exited;
    const inFlight = this.starting.get(sessionPath);
    if (inFlight) {
      const outcome = await inFlight;
      return "session" in outcome ? this.reuse(outcome.session, spec) : outcome;
    }
    const started = this.spawn(request, cwd, sessionPath);
    this.starting.set(sessionPath, started);
    try {
      return await started;
    } finally {
      this.starting.delete(sessionPath);
    }
  }

  /** A live child for a request, unchanged, when it runs with the requested profile, prompt and tools. */
  private reuse(live: LiveSession, spec: ProfileSpec): StartOutcome {
    if (!sameProfile(live.spec, spec)) return fail(409, "profile_mismatch");
    // The client's attach follows; an eviction tick in between must not take the child.
    live.lastActivity = Date.now();
    return { session: live };
  }

  /** Newest live-root session of `cwd` started with `profile` (`continue`). */
  private async newestOf(cwd: string, profile: Profile): Promise<SessionMeta | undefined> {
    for (const meta of await this.host.listSessions(cwd, false)) {
      if ((await profileOf(this.host, meta.sessionPath)) === profile) return meta;
    }
    return undefined;
  }

  /** Lock (M1, before the child starts, which excludes a concurrent archive), spawn, identify, register. */
  private async spawn(request: StartRequest, cwd: string, sessionPath: string | undefined): Promise<StartOutcome> {
    const lock = new SessionLock(this.host);
    if (sessionPath) {
      const taken = await lock.take(sessionPath);
      // Missing: archived or deleted since it was looked up.
      if (taken === "missing") return fail(404, "not_found");
      if (taken === "busy") return fail(409, "locked");
    }
    const piArgs = ["--mode", "rpc", ...profileArgs(request.spec), ...(sessionPath ? ["--session", sessionPath] : [])];
    let session: LiveSession;
    try {
      const child = await this.host.spawn({ profile: request.spec.profile, cwd, piArgs });
      session = new LiveSession(child, cwd, request.spec, lock, (exited) => this.forget(exited));
    } catch (error) {
      lock.release();
      return fail(502, "spawn_failed", { message: String(error) });
    }

    const state = await session.requestAsync({ type: "get_state" });
    const data = isRecord(state.data) ? state.data : {};
    if (state.success !== true || typeof data.sessionId !== "string" || typeof data.sessionFile !== "string") {
      await session.close();
      return fail(502, "spawn_failed", { message: String(state.error ?? "no session file"), stderrTail: session.stderrTail });
    }
    session.identify(data.sessionId, sessionPath ?? resolve(data.sessionFile));

    const settings: [string, Record<string, unknown>][] = [];
    if (request.model) settings.push(["model", { type: "set_model", provider: request.model.provider, modelId: request.model.id }]);
    if (request.thinking) settings.push(["thinking", { type: "set_thinking_level", level: request.thinking }]);
    for (const [error, command] of settings) {
      const response = await session.requestAsync(command);
      if (response.success !== true) {
        await session.close();
        return fail(400, error, { message: String(response.error ?? "rejected") });
      }
    }

    try {
      // G5: before the session is listed, so a `document` session never shows as `coding`.
      await markProfile(this.host, session.sessionPath, request.spec.profile);
    } catch (error) {
      await session.close();
      return fail(502, "spawn_failed", { message: `profile marker: ${String(error)}` });
    }

    if (session.exitCode !== undefined) {
      return fail(502, "spawn_failed", { message: `pi exited with ${session.exitCode}`, stderrTail: session.stderrTail });
    }
    this.byId.set(session.sessionId, session);
    this.byPath.set(session.sessionPath, session);
    return { session };
  }

  private forget(session: LiveSession): void {
    if (this.byId.get(session.sessionId) === session) this.byId.delete(session.sessionId);
    if (this.byPath.get(session.sessionPath) === session) this.byPath.delete(session.sessionPath);
  }

  private onSocketMessage(ws: ServerWebSocket<SocketData>, message: string | Buffer): void {
    let frame: unknown;
    try {
      frame = JSON.parse(String(message));
    } catch {
      ws.send(stamp(envelope("error", "", { message: "frame is not JSON" }), null, 0));
      return;
    }
    const sessionId = isRecord(frame) && typeof frame.sessionId === "string" ? frame.sessionId : "";
    const session = this.byId.get(sessionId);
    const type = isRecord(frame) ? frame.type : undefined;
    if (type === "detach") {
      session?.detach(ws);
      return;
    }
    if (type !== "attach" && type !== "resync" && type !== "command") {
      const error = envelope("error", sessionId, { message: `unknown frame type ${JSON.stringify(type)}` });
      if (session) session.deliver(ws, error);
      else ws.send(stamp(error, null, 0));
      return;
    }
    if (!session) {
      ws.send(stamp(envelope("error", sessionId, { message: "session is not live, start it with POST /sessions" }), null, 0));
      return;
    }
    if (type === "attach") {
      session.attach(ws);
      return;
    }
    const fields = frame as Record<string, unknown>;
    if (type === "resync") {
      const cursor = fields.cursor;
      if (cursor === null || typeof cursor === "string") session.resync(ws, cursor);
      else session.deliver(ws, envelope("error", sessionId, { message: "resync needs a cursor (entry id or null)" }));
      return;
    }
    const refused = session.command(ws, fields.command);
    if (refused) session.deliver(ws, envelope("error", sessionId, { message: refused }));
  }
}
