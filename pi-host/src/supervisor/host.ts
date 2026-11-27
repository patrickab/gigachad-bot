/**
 * The supervisor behind `createSupervisor()`: live session registry, `POST /sessions`, `/ws` multiplexing, eviction,
 * and the `sessions_changed` broadcast (M25).
 *
 * - Consumes: `SupervisorConfig`, the `Host` (spawn, session lock, session metadata, fork, delete, session watcher),
 *   Catalog's `sessionInfo`, `copyPrivateState` and the profile marker (G5), `profiles.ts` for each profile's Pi
 *   arguments
 * - Provides: `SupervisorHost`, the `Supervisor` implementation (HTTP rows owned by Supervisor, WS `/ws`, M18). The
 *   HTTP rows: `POST /sessions`, `POST /sessions/:id/close`, `DELETE /sessions/:id` (it must stop the child first,
 *   which only the supervisor can) and `POST /sessions/:id/turns/:entryId/delete` (stop the child, rewrite the file,
 *   start it again)
 */
import { isAbsolute, resolve } from "node:path";
import type { ServerWebSocket, WebSocketHandler } from "bun";
import { forgetEntries } from "../catalog/entries.ts";
import { copyPrivateState, markCacheRetention, markProfile, profileOf, sessionInfo, type SessionInfo } from "../catalog/index.ts";
import type { DeleteTurnRequest, DeleteTurnResult, Host, SessionMeta } from "../host/index.ts";
import { forgetMedia } from "../media.ts";
import { isRecord } from "../json.ts";
import type { SupervisorConfig } from "./config.ts";
import { RESUME_PROMPT, type ResumeState, type RunRecord } from "./resume.ts";
import type { Health, SocketData, Supervisor } from "./index.ts";
import { SessionLock } from "./lock.ts";
import { childModels, parseProfileSpec, profileArgs, sameProfile, type ModelRef, type ProfileSpec } from "./profiles.ts";
import { envelope, LiveSession, stamp, type FrameSink, type LiveStatus } from "./session.ts";

type ForkOf = { sessionPath: string; entryId: string };
type Resume = null | { sessionPath: string } | { sessionId: string } | { forkOf: ForkOf };
type StartRequest = {
  cwd: string;
  spec: ProfileSpec;
  resume: Resume;
  model?: ModelRef;
  thinking?: string;
};
type Failure = { status: number; body: Record<string, unknown> };
type StartOutcome = { session: LiveSession } | Failure;
type DeleteTurnAnswer = Extract<DeleteTurnResult, { ok: true }>;

const SESSION_CLOSE = /^\/sessions\/([^/]+)\/close$/;
const SESSION = /^\/sessions\/([^/]+)$/;
const TURN_DELETE = /^\/sessions\/([^/]+)\/turns\/([^/]+)\/delete$/;
/** HTTP answer of each `Host.deleteTurn` refusal. */
const TURN_REFUSALS: Record<string, number> = {
  not_found: 404,
  unknown: 404,
  not_prompt: 400,
  off_branch: 400,
  compacted: 409,
  locked: 409,
  changed: 409,
};
/** What happened to one interrupted session at start: `resumed`, or why not (`open`, `locked`, `not_found`, ...). */
export type ResumeOutcome = { sessionId: string; outcome: string };
const MAX_EVICTION_INTERVAL_MS = 60_000;
const MIN_EVICTION_INTERVAL_MS = 50;
/** Far above any snapshot (omp sessions reach 27 MB): a healthy client never hits it, a stalled one is closed. */
const BACKPRESSURE_LIMIT_BYTES = 512 * 1024 * 1024;

function fail(status: number, error: string, extra: Record<string, unknown> = {}): Failure {
  return { status, body: { error, ...extra } };
}

/** JSON response of a failure. */
function respond({ status, body }: Failure): Response {
  return Response.json(body, { status });
}

/** The session id of a route's path segment, undefined when it is not valid percent-encoding. */
function sessionIdOf(segment: string): string | undefined {
  try {
    return decodeURIComponent(segment);
  } catch {
    return undefined;
  }
}

/** True when `live` differs from `spec` only in its document prompt or tools and nothing is in flight. */
function isRestartable(live: LiveSession, spec: ProfileSpec): boolean {
  const { running, waiting } = live.liveStatus();
  return live.spec.profile === spec.profile && !running && !waiting;
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
  /** The `starting` entries that are a turn delete's rewrite: a start waiting on one tries again once it settled. */
  private readonly rewrites = new WeakSet<Promise<StartOutcome>>();
  /** Clients of sessions whose child restarts after a turn delete, by session id: they get a `resync` from the new child. */
  private readonly carried = new Map<string, Set<FrameSink>>();
  private readonly evictionTimer: Timer;
  private readonly sockets = new Set<ServerWebSocket<SocketData>>();
  private readonly stopWatching: () => void;
  private evicting = false;
  private shuttingDown = false;

  readonly websocket: WebSocketHandler<SocketData> = {
    open: (ws) => void this.sockets.add(ws),
    message: (ws, message) => this.onSocketMessage(ws, message),
    close: (ws) => {
      this.sockets.delete(ws);
      for (const session of this.byId.values()) session.disconnect(ws);
      for (const clients of this.carried.values()) clients.delete(ws);
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
    private readonly resume?: ResumeState,
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
    if (url.pathname === "/settings" && this.resume) {
      if (request.method === "GET") return Response.json({ autoResume: await this.resume.autoResume() });
      if (request.method === "PUT") {
        const body = await request.json().catch(() => undefined);
        if (!isRecord(body) || typeof body.autoResume !== "boolean") return respond(fail(400, "invalid_body"));
        await this.resume.setAutoResume(body.autoResume);
        return Response.json({ autoResume: body.autoResume });
      }
    }
    if (url.pathname === "/sessions" && request.method === "POST") {
      const parsed = await parseStartRequest(await request.json().catch(() => undefined), this.host);
      if ("status" in parsed) return respond(parsed);
      const outcome = await this.start(parsed);
      if ("status" in outcome) return respond(outcome);
      return Response.json({ session: await this.info(outcome.session) });
    }
    const close = SESSION_CLOSE.exec(url.pathname);
    if (close && request.method === "POST") {
      await this.byId.get(decodeURIComponent(close[1]!))?.close();
      return new Response(null, { status: 204 });
    }
    const turn = TURN_DELETE.exec(url.pathname);
    if (turn && request.method === "POST") {
      const sessionId = sessionIdOf(turn[1]!);
      const entryId = sessionIdOf(turn[2]!);
      if (sessionId === undefined || entryId === undefined) return respond(fail(400, "invalid_session_id"));
      const body = await request.json().catch(() => undefined);
      if (!isRecord(body) || (body.through !== "turn" && body.through !== "end")) return respond(fail(400, "invalid_body"));
      const answer = await this.deleteTurn(sessionId, entryId, body.through);
      return "status" in answer ? respond(answer) : Response.json(answer);
    }
    const target = SESSION.exec(url.pathname);
    if (target && request.method === "DELETE") {
      const sessionId = sessionIdOf(target[1]!);
      if (sessionId === undefined) return respond(fail(400, "invalid_session_id"));
      const refused = await this.delete(sessionId);
      return refused ? respond(refused) : new Response(null, { status: 204 });
    }
    return undefined;
  }

  /**
   * `DELETE /sessions/:id`: stop the session's child whatever it is doing, then remove the session file and its
   * sidecar. 404 for an id neither a child nor a file has, 409 `live` like archive when another holder has the file
   * (a child being spawned, another pi-host).
   */
  private async delete(sessionId: string): Promise<Failure | undefined> {
    // A new session has no file before its first message, only its live child knows the id.
    const liveById = this.byId.get(sessionId);
    await liveById?.close();
    const sessionPath = await this.host.findSession(sessionId);
    const meta = sessionPath === undefined ? undefined : await this.host.sessionMeta(sessionPath);
    if (!meta) return liveById ? undefined : fail(404, "not_found");
    // Either path form may key the child: a new session's as Pi named it, a resumed one's real path
    for (const path of new Set([meta.sessionPath, meta.realPath])) await this.byPath.get(path)?.close();
    const deleted = await this.host.deleteSession(meta.sessionPath);
    if (deleted.ok) return undefined;
    return deleted.error === "locked" ? fail(409, "live") : fail(404, "not_found");
  }

  /**
   * `POST /sessions/:id/turns/:entryId/delete`: remove a turn from the session file (`Host.deleteTurn`). A live child
   * is stopped first and started again on the same path afterwards, so its context is rebuilt from the new file. The
   * rewrite and restart run as the path's start in flight, so a `POST /sessions` for it waits and gets the new child;
   * the file's lock is held for the rewrite. Attached clients are carried over: they get no `exit`, the new child
   * sends each a `resync` (`replace`, new `streamId`).
   */
  private async deleteTurn(
    sessionId: string,
    entryId: string,
    through: DeleteTurnRequest["through"],
  ): Promise<Failure | DeleteTurnAnswer> {
    const sessionPath = await this.host.findSession(sessionId);
    const meta = sessionPath === undefined ? undefined : await this.host.sessionMeta(sessionPath);
    if (!meta) return fail(404, "not_found");
    // The factory plugin's runner owns a factory task's file
    if (meta.factory) return fail(400, "factory");
    const key = meta.realPath;
    for (let inFlight = this.starting.get(key); inFlight; inFlight = this.starting.get(key)) await inFlight;
    const request = { sessionPath: meta.sessionPath, entryId, through };
    // Refusals before the child is stopped; the real run checks again under the lock
    const check = await this.host.deleteTurn({ ...request, dryRun: true });
    if (!check.ok) return fail(TURN_REFUSALS[check.error] ?? 409, check.error);
    if (this.starting.has(key)) return fail(409, "locked");

    const found = this.byPath.get(key) ?? this.byPath.get(meta.sessionPath);
    const live = found && !found.closing ? found : undefined;
    if (live) {
      const { running, waiting } = live.liveStatus();
      if (running || waiting) return fail(409, "running");
    }
    // Synchronously from the status check to the guard and the stop: no run or start gets in between
    const clients = new Set(live?.takeClients() ?? []);
    if (live) this.carried.set(sessionId, clients);
    const stopped = live?.close();
    let answer: Failure | DeleteTurnAnswer = fail(409, "locked");
    const rewrite = (async (): Promise<StartOutcome> => {
      await (stopped ?? found?.exited);
      try {
        const result = await this.host.deleteTurn(request);
        if (result.ok) {
          const notRemoved = [...(result.notRemoved ?? []), ...forgetMedia(result.removedMedia)];
          answer = { ...result, ...(notRemoved.length > 0 && { notRemoved }) };
        } else answer = fail(TURN_REFUSALS[result.error] ?? 409, result.error);
      } catch (error) {
        answer = fail(500, "rewrite_failed", { message: String(error) });
      } finally {
        // The file may be rewritten whatever the answer: the entries cache never outlives it
        for (const path of new Set([meta.sessionPath, key])) forgetEntries(path);
      }
      // A start waiting for this path without a child to restart gets what the file lock would have answered
      if (!live) return fail(409, "locked");
      return this.spawn({ cwd: live.cwd, spec: live.spec, resume: { sessionPath: key } }, live.cwd, key);
    })();
    this.starting.set(key, rewrite);
    this.rewrites.add(rewrite);
    let outcome: StartOutcome;
    try {
      outcome = await rewrite;
    } finally {
      this.starting.delete(key);
      this.carried.delete(sessionId);
    }
    if (live) this.handOver(sessionId, clients, outcome);
    return answer;
  }

  /** Carried clients of a restarted session: a `resync` from the new child, or the `exit` they were spared. */
  private handOver(sessionId: string, clients: Set<FrameSink>, outcome: StartOutcome): void {
    for (const client of clients) {
      if (!this.sockets.has(client as ServerWebSocket<SocketData>)) continue;
      if ("session" in outcome) outcome.session.resync(client, null);
      else client.send(stamp(envelope("exit", sessionId, { code: -1, stderrTail: String(outcome.body.message ?? outcome.body.error) }), null, 0));
    }
  }

  async shutdown(): Promise<void> {
    // The runs this ends are interrupted, not settled: their records stay for the next start.
    this.shuttingDown = true;
    clearInterval(this.evictionTimer);
    this.stopWatching();
    for (const session of this.byId.values()) session.terminate();
    await this.resume?.flush();
  }

  /**
   * On start: take the run lists of dead pi-hosts and, when `autoResume` is on, reopen those sessions with a short
   * resume prompt. Always claims, so a list never waits for the setting to be switched on. A session another holder
   * has (live here, locked by the Pi TUI or another pi-host), an archived one and one the user aborted (no record)
   * are left alone.
   */
  async resumeInterrupted(): Promise<ResumeOutcome[]> {
    if (!this.resume) return [];
    const records = await this.resume.claimOrphans();
    if (records.length === 0 || !(await this.resume.autoResume())) return [];
    const outcomes: ResumeOutcome[] = [];
    for (const record of records) {
      const outcome = await this.resumeOne(record).catch((error) => `failed: ${String(error)}`);
      outcomes.push({ sessionId: record.sessionId, outcome });
      if (outcome !== "resumed") console.warn(`pi-host: session ${record.sessionId} not resumed: ${outcome}`);
    }
    return outcomes;
  }

  private async resumeOne(record: RunRecord): Promise<string> {
    if (this.byId.has(record.sessionId) || this.byPath.has(resolve(record.sessionPath))) return "open";
    const started = await this.start({ cwd: record.cwd, spec: record.spec, resume: { sessionPath: record.sessionPath } });
    if ("status" in started) return String(started.body.error ?? started.status);
    const { session } = started;
    if (session.liveStatus().running) return "open";
    const response = await session.requestAsync({ type: "prompt", message: RESUME_PROMPT });
    if (response.success !== true) {
      void session.close();
      return `prompt_failed: ${String(response.error ?? "rejected")}`;
    }
    return "resumed";
  }

  /** Auto-resume bookkeeping: a run in `session` started or ended. A failed write must not touch the session. */
  private onRun(session: LiveSession, active: boolean): void {
    if (!this.resume || this.shuttingDown) return;
    const { sessionId, sessionPath, cwd, spec } = session;
    const written = active ? this.resume.track({ sessionId, sessionPath, cwd, spec }) : this.resume.untrack(sessionId);
    written.catch((error) => console.warn(`pi-host: resume state not written: ${String(error)}`));
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
      tokens: 0,
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
    if (resume && "forkOf" in resume) {
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
      if (liveById && !liveById.closing && sameProfile(liveById.spec, spec)) return this.reuse(liveById, spec);
      const sessionPath = "sessionId" in resume ? await this.host.findSession(resume.sessionId) : resume.sessionPath;
      meta = sessionPath === undefined ? undefined : await this.host.sessionMeta(sessionPath);
      if (!meta) return liveById && !liveById.closing ? fail(409, "profile_mismatch") : fail(404, "not_found");
      if (meta.archived) return fail(409, "archived");
      // The factory plugin's runner holds every factory task.
      if (meta.factory) return fail(409, "factory");
      if ((await profileOf(this.host, meta.sessionPath)) !== spec.profile) return fail(409, "profile_mismatch");
      cwd = meta.cwd;
    }
    if (!meta) return this.spawn(request, cwd, undefined);
    const sessionPath = meta.realPath;

    const live = this.byPath.get(sessionPath) ?? this.byPath.get(meta.sessionPath);
    if (live && !live.closing) {
      if (sameProfile(live.spec, spec)) return this.reuse(live, spec);
      // A document prompt or tool change (a template edit, a new AGENTS.md) restarts an idle child with the new spec
      if (!isRestartable(live, spec)) return fail(409, "profile_mismatch");
      await live.close();
    } else if (live) await live.exited;
    for (let inFlight = this.starting.get(sessionPath); inFlight; ) {
      const outcome = await inFlight;
      if ("session" in outcome) return this.reuse(outcome.session, spec);
      // A turn delete without a child to restart (or failing to) frees the file when it settles: start on it then
      if (!this.rewrites.has(inFlight)) return outcome;
      const next = this.starting.get(sessionPath);
      inFlight = next === inFlight ? undefined : next;
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
      session = new LiveSession(child, cwd, request.spec, lock, (exited) => this.forget(exited), (live, active) => this.onRun(live, active));
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

    if (request.model) {
      let response: Record<string, unknown> = {};
      for (const { provider, id } of childModels(request.spec, request.model)) {
        response = await session.requestAsync({ type: "set_model", provider, modelId: id });
        if (response.success === true) break;
      }
      if (response.success !== true) {
        await session.close();
        return fail(400, "model", { message: String(response.error ?? "rejected") });
      }
    }
    if (request.thinking) {
      const response = await session.requestAsync({ type: "set_thinking_level", level: request.thinking });
      if (response.success !== true) {
        await session.close();
        return fail(400, "thinking", { message: String(response.error ?? "rejected") });
      }
    }

    try {
      // G5: before the session is listed, so a `document` session never shows as `coding`.
      await markProfile(this.host, session.sessionPath, request.spec.profile);
      // The retention its Pi got through `childEnv`, so the session list knows how long its cache lives
      await markCacheRetention(this.host, session.sessionPath, process.env.PI_CACHE_RETENTION);
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
    this.onRun(session, false);
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
      this.carried.get(sessionId)?.delete(ws);
      return;
    }
    if (type !== "attach" && type !== "resync" && type !== "command") {
      const error = envelope("error", sessionId, { message: `unknown frame type ${JSON.stringify(type)}` });
      if (session) session.deliver(ws, error);
      else ws.send(stamp(error, null, 0));
      return;
    }
    const carried = this.carried.get(sessionId);
    if (!session && carried) {
      // The child restarts after a turn delete: an attach or resync is answered by the new child's `resync`
      if (type === "command") ws.send(stamp(envelope("error", sessionId, { message: "session is restarting" }), null, 0));
      else carried.add(ws);
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
