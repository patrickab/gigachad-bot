/**
 * The supervisor behind `createSupervisor()`: live session registry, `POST /sessions`, `/ws` multiplexing, eviction.
 *
 * - Consumes: `SupervisorConfig`, a spawn strategy per profile, Catalog's `sessionInfo`, `listSessions`, `isPinned`
 * - Provides: `SupervisorHost`, the `Supervisor` implementation (HTTP rows owned by Supervisor, WS `/ws`, M18)
 */
import { realpathSync } from "node:fs";
import { stat } from "node:fs/promises";
import { isAbsolute, resolve } from "node:path";
import type { ServerWebSocket, WebSocketHandler } from "bun";
import { sessionInfo, type SessionInfo } from "../catalog/index.ts";
import { listSessions } from "../catalog/sessions.ts";
import { isPlainObject as isRecord } from "../catalog/state.ts";
import { childEnv, type SupervisorConfig } from "./config.ts";
import type { Health, LiveSessions, SocketData, Supervisor } from "./index.ts";
import { tryLockFile, type FileLock } from "./lock.ts";
import { envelope, LiveSession, stamp, type LiveStatus } from "./session.ts";
import { PROFILES, spawnStrategyFor, type Profile } from "./spawn.ts";

type Resume = null | "continue" | { sessionPath: string };
type StartRequest = {
  cwd: string;
  profile: Profile;
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

/** Absolute path with symlinks resolved when the file exists, the key the catalog looks sessions up by. */
function canonicalPath(path: string): string {
  try {
    return realpathSync(path);
  } catch {
    return resolve(path);
  }
}

function fail(status: number, error: string, extra: Record<string, unknown> = {}): Failure {
  return { status, body: { error, ...extra } };
}

/** Validated `POST /sessions` body, or the failure to answer with. */
async function parseStartRequest(body: unknown): Promise<StartRequest | Failure> {
  if (!isRecord(body)) return fail(400, "invalid_body");
  const { cwd, profile, resume, model, thinking } = body;
  if (typeof cwd !== "string" || !isAbsolute(cwd) || !(await stat(cwd).catch(() => undefined))?.isDirectory()) {
    return fail(400, "invalid_cwd");
  }
  if (!PROFILES.includes(profile as Profile)) return fail(400, "invalid_profile");
  const validResume =
    resume === null ||
    resume === undefined ||
    resume === "continue" ||
    (isRecord(resume) && typeof resume.sessionPath === "string" && isAbsolute(resume.sessionPath));
  if (!validResume) return fail(400, "invalid_resume");
  if (model !== undefined && !(isRecord(model) && typeof model.provider === "string" && typeof model.id === "string")) {
    return fail(400, "invalid_model");
  }
  if (thinking !== undefined && typeof thinking !== "string") return fail(400, "invalid_thinking");
  return {
    cwd: resolve(cwd),
    profile: profile as Profile,
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
  private evicting = false;

  readonly websocket: WebSocketHandler<SocketData> = {
    message: (ws, message) => this.onSocketMessage(ws, message),
    close: (ws) => {
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
  ) {
    const interval = Math.min(MAX_EVICTION_INTERVAL_MS, Math.max(MIN_EVICTION_INTERVAL_MS, config.idleMs / 4));
    this.evictionTimer = setInterval(() => void this.evictIdle(), interval);
    this.evictionTimer.unref();
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

  liveStatus(sessionPath: string): LiveStatus | undefined {
    return this.livePath(sessionPath)?.liveStatus();
  }

  /** The live session of a file under either key form: a new session is keyed before its file exists (no realpath). */
  private livePath(sessionPath: string): LiveSession | undefined {
    return this.byPath.get(resolve(sessionPath)) ?? this.byPath.get(canonicalPath(sessionPath));
  }

  async handle(request: Request, url: URL): Promise<Response | undefined> {
    if (url.pathname === "/sessions" && request.method === "POST") {
      const parsed = await parseStartRequest(await request.json().catch(() => undefined));
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
    const existing = await sessionInfo(session.sessionPath, this as LiveSessions);
    if (existing) return existing;
    const now = new Date().toISOString();
    return {
      sessionId: session.sessionId,
      sessionPath: session.sessionPath,
      cwd: session.cwd,
      machine: "local",
      createdAt: now,
      modifiedAt: now,
      firstMessage: "(no messages)",
      messageCount: 0,
      live: session.liveStatus(),
      pinned: false,
      archived: false,
    };
  }

  /** Resolve the session file to resume, then return its live child or spawn one. */
  private async start(request: StartRequest): Promise<StartOutcome> {
    let requestedPath: string | undefined;
    let cwd = request.cwd;
    if (request.resume === "continue") {
      requestedPath = (await listSessions(cwd, false, this))[0]?.sessionPath;
    } else if (request.resume) {
      const info = await sessionInfo(request.resume.sessionPath, this);
      if (!info) return fail(404, "not_found");
      if (info.archived) return fail(409, "archived");
      requestedPath = info.sessionPath;
      cwd = info.cwd;
    }
    if (!requestedPath) return this.spawn(request, cwd, undefined);
    const sessionPath = canonicalPath(requestedPath);

    const live = this.byPath.get(sessionPath) ?? this.livePath(requestedPath);
    if (live && !live.closing) {
      // The client's attach follows; an eviction tick in between must not take the child.
      live.lastActivity = Date.now();
      return { session: live };
    }
    if (live) await live.exited;
    const inFlight = this.starting.get(sessionPath);
    if (inFlight) return inFlight;
    const started = this.spawn(request, cwd, sessionPath);
    this.starting.set(sessionPath, started);
    try {
      return await started;
    } finally {
      this.starting.delete(sessionPath);
    }
  }

  /** Lock (M1, synchronously before the child starts, which excludes a concurrent archive), spawn, identify, register. */
  private async spawn(request: StartRequest, cwd: string, sessionPath: string | undefined): Promise<StartOutcome> {
    let lock: FileLock | undefined;
    if (sessionPath) {
      try {
        lock = tryLockFile(sessionPath);
      } catch (error) {
        // Archived or deleted since it was looked up.
        if ((error as NodeJS.ErrnoException).code === "ENOENT") return fail(404, "not_found");
        throw error;
      }
      if (!lock) return fail(409, "locked");
    }
    const piArgs = ["--mode", "rpc", ...(sessionPath ? ["--session", sessionPath] : [])];
    let session: LiveSession;
    try {
      const child = spawnStrategyFor(request.profile, this.config.piBin).spawn({
        profile: request.profile,
        cwd,
        piArgs,
        env: childEnv(this.config.extraChildEnv),
      });
      session = new LiveSession(child, cwd, request.profile, lock, (exited) => this.forget(exited));
    } catch (error) {
      lock?.release();
      return fail(502, "spawn_failed", { message: String(error) });
    }

    const state = await session.requestAsync({ type: "get_state" });
    const data = isRecord(state.data) ? state.data : {};
    if (state.success !== true || typeof data.sessionId !== "string" || typeof data.sessionFile !== "string") {
      await session.close();
      return fail(502, "spawn_failed", { message: String(state.error ?? "no session file"), stderrTail: session.stderrTail });
    }
    session.identify(data.sessionId, sessionPath ?? canonicalPath(data.sessionFile));

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
