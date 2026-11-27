/** Supervisor and catalog against an in-memory fake `Host` (M23): no process, file or lock outside the seam. */
import { afterAll, beforeAll, expect, test } from "bun:test";
import type { Server } from "bun";
import type {
  DeleteResult,
  DeleteTurnRequest,
  DeleteTurnResult,
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
} from "../src/host/index.ts";
import { startServer } from "../src/server.ts";
import { createSupervisor, type SocketData, type Supervisor } from "../src/supervisor/index.ts";

const TOKEN = "test-token";
const TOKENS = { full: TOKEN, gateway: "test-gateway-token" };
const CWD = "/remote/project";
const SESSION_ID = "01a1f3fe-0000-7000-8000-00000000000a";
const LIVE_ROOT = "/remote/sessions";
const ARCHIVE_ROOT = "/remote/sessions-archive";

/** An in-process Pi: answers `get_state`, `get_entries` and any command, exits on SIGTERM or stdin EOF. */
function fakeChild(sessionId: string, sessionFile: string): PiChild {
  const encoder = new TextEncoder();
  let stdout!: ReadableStreamDefaultController<Uint8Array>;
  const exit = Promise.withResolvers<number>();
  const emit = (record: object) => stdout.enqueue(encoder.encode(`${JSON.stringify(record)}\n`));
  const stop = (code: number) => {
    try {
      stdout.close();
    } catch {
      // already closed
    }
    exit.resolve(code);
  };
  const answer = (command: { id?: string; type: string }) => {
    const data =
      command.type === "get_state" ? { sessionId, sessionFile } : command.type === "get_entries" ? { entries: [], leafId: null } : {};
    emit({ type: "response", id: command.id, command: command.type, success: true, data });
  };
  return {
    stdin: {
      write: (text: string) => {
        for (const line of text.split("\n")) if (line) queueMicrotask(() => answer(JSON.parse(line)));
        return text.length;
      },
      flush: () => 0,
      end: () => stop(0),
    },
    stdout: new ReadableStream<Uint8Array>({ start: (controller) => void (stdout = controller) }),
    stderr: new ReadableStream<Uint8Array>({ start: (controller) => controller.close() }),
    exited: exit.promise,
    kill: () => stop(143),
  };
}

class FakeHost implements Host {
  readonly machine = "box-1";
  readonly spawned: SpawnRequest[] = [];
  readonly locks = new Map<string, string>();
  readonly sessions = new Map<string, SessionMeta>();
  readonly states = new Map<string, PrivateState>();

  async piVersion(): Promise<string> {
    return "1.5.2";
  }

  async spawn(request: SpawnRequest): Promise<PiChild> {
    this.spawned.push(request);
    const resumed = request.piArgs.includes("--session") ? request.piArgs.at(-1) : undefined;
    const meta = resumed ? this.sessions.get(resumed) : undefined;
    return fakeChild(meta?.sessionId ?? crypto.randomUUID(), resumed ?? `${LIVE_ROOT}/new.jsonl`);
  }

  async lockSession(sessionPath: string): Promise<LockResult> {
    if (!this.sessions.has(sessionPath)) return { ok: false, error: "missing" };
    if ([...this.locks.values()].includes(sessionPath)) return { ok: false, error: "busy" };
    const lockId = crypto.randomUUID();
    this.locks.set(lockId, sessionPath);
    return { ok: true, lockId };
  }

  async unlockSession(lockId: string): Promise<void> {
    this.locks.delete(lockId);
  }

  async isDirectory(path: string): Promise<boolean> {
    return path === CWD;
  }

  async listSessions(cwd: string | null, archived: boolean): Promise<SessionMeta[]> {
    return [...this.sessions.values()].filter((meta) => (cwd === null || meta.cwd === cwd) && meta.archived === archived);
  }

  async sessionMeta(sessionPath: string): Promise<SessionMeta | undefined> {
    return this.sessions.get(sessionPath);
  }

  async findSession(sessionId: string): Promise<string | undefined> {
    return [...this.sessions.values()].find((meta) => meta.sessionId === sessionId)?.sessionPath;
  }

  async moveSession(sessionPath: string, toArchive: boolean): Promise<MoveResult> {
    const meta = this.sessions.get(sessionPath);
    if (!meta) return { ok: false, error: "not_found" };
    if ([...this.locks.values()].includes(sessionPath)) return { ok: false, error: "locked" };
    const [from, to] = toArchive ? [LIVE_ROOT, ARCHIVE_ROOT] : [ARCHIVE_ROOT, LIVE_ROOT];
    const path = sessionPath.replace(from, to);
    this.sessions.delete(sessionPath);
    this.sessions.set(path, { ...meta, sessionPath: path, realPath: path, archived: toArchive });
    return { ok: true, path };
  }

  readonly deletedTurns: DeleteTurnRequest[] = [];

  async deleteTurn(request: DeleteTurnRequest): Promise<DeleteTurnResult> {
    if (!this.sessions.has(request.sessionPath)) return { ok: false, error: "not_found" };
    if (request.dryRun) return { ok: true, removedEntries: 0, removedMedia: [] };
    if ([...this.locks.values()].includes(request.sessionPath)) return { ok: false, error: "locked" };
    this.deletedTurns.push(request);
    return { ok: true, removedEntries: 2, removedMedia: [] };
  }

  async deleteSession(sessionPath: string): Promise<DeleteResult> {
    if (!this.sessions.has(sessionPath)) return { ok: false, error: "not_found" };
    if ([...this.locks.values()].includes(sessionPath)) return { ok: false, error: "locked" };
    this.sessions.delete(sessionPath);
    this.states.delete(sessionPath);
    return { ok: true };
  }

  async forkSession(request: ForkRequest): Promise<ForkResult> {
    const source = this.sessions.get(request.sessionPath);
    if (!source) return { ok: false, error: "not_found" };
    if (request.entryId !== "e1") return { ok: false, error: "no_entry" };
    const path = `${LIVE_ROOT}/project/fork.jsonl`;
    const sessionId = "01a1f3fe-0000-7000-8000-00000000000c";
    this.sessions.set(path, { ...source, sessionId, sessionPath: path, realPath: path, parentSession: request.sessionPath });
    return { ok: true, path };
  }

  async readEntries(): Promise<SessionEntries> {
    return { entries: [], leafId: null };
  }

  async readState(sessionPath: string): Promise<PrivateState> {
    return this.states.get(sessionPath) ?? { revision: 0, pinned: false, data: {} };
  }

  async writeState(sessionPath: string, state: PrivateState): Promise<void> {
    this.states.set(sessionPath, state);
  }

  async listFiles(_cwd: string): Promise<string[]> {
    return ["README.md", "src/main.ts"];
  }

  async searchSessions(query: string, limit: number): Promise<SearchMatch[]> {
    return [...this.sessions.values()]
      .filter((meta) => !meta.archived && meta.firstMessage.includes(query))
      .slice(0, limit)
      .map((meta) => ({ sessionPath: meta.sessionPath, entryId: "e1", snippet: meta.firstMessage, matches: [], score: 1 }));
  }

  async sessionMatches(): Promise<SessionMatchPage | undefined> {
    return { matches: [], total: 0 };
  }

  readonly watchers = new Set<(sessionPaths: string[]) => void>();

  watchSessions(onChange: (sessionPaths: string[]) => void): () => void {
    this.watchers.add(onChange);
    return () => this.watchers.delete(onChange);
  }
}

const host = new FakeHost();
const sessionPath = `${LIVE_ROOT}/project/2026-11-19_${SESSION_ID}.jsonl`;
let supervisor: Supervisor;
let server: Server<SocketData>;

async function api(path: string, init: RequestInit = {}): Promise<any> {
  const response = await fetch(`http://127.0.0.1:${server.port}${path}`, {
    ...init,
    headers: { authorization: `Bearer ${TOKEN}`, "content-type": "application/json" },
  });
  return { status: response.status, body: response.status === 204 ? null : await response.json() };
}

beforeAll(async () => {
  host.sessions.set(sessionPath, {
    sessionId: SESSION_ID,
    sessionPath,
    realPath: sessionPath,
    cwd: CWD,
    createdAt: "2026-11-19T10:00:00.000Z",
    modifiedAt: "2026-11-19T10:05:00.000Z",
    firstMessage: "hello remote",
    messageCount: 2,
    tokens: 0,
    archived: false,
  });
  supervisor = await createSupervisor({ host, idleMs: 3_600_000 });
  server = startServer({ hostname: "127.0.0.1", port: 0, tokens: TOKENS, supervisor });
});

afterAll(async () => {
  await supervisor.shutdown();
  server.stop(true);
});

test("supervisor and catalog run every machine action through the Host", async () => {
  const listed = await api(`/sessions?cwd=${encodeURIComponent(CWD)}`);
  expect(listed.body.sessions).toEqual([
    expect.objectContaining({ sessionId: SESSION_ID, sessionPath, machine: "box-1", live: null, pinned: false, archived: false }),
  ]);

  const started = await api("/sessions", {
    method: "POST",
    body: JSON.stringify({ cwd: CWD, profile: "coding", resume: { sessionPath } }),
  });
  expect(started.status).toBe(200);
  expect(started.body.session).toMatchObject({ sessionId: SESSION_ID, machine: "box-1", live: { attached: 0 } });
  expect(host.spawned).toEqual([{ profile: "coding", cwd: CWD, piArgs: ["--mode", "rpc", "--session", sessionPath] }]);
  expect([...host.locks.values()]).toEqual([sessionPath]);

  // The child is detached and idle: archive closes it (releasing the Host lock) and moves the file
  const archived = await api(`/sessions/${SESSION_ID}/archive`, { method: "POST" });
  expect(host.locks.size).toBe(0);
  expect(archived.body.session).toMatchObject({ archived: true, machine: "box-1", sessionPath: sessionPath.replace(LIVE_ROOT, ARCHIVE_ROOT) });
  const pinned = await api(`/sessions/${SESSION_ID}/pin`, { method: "PUT", body: JSON.stringify({ pinned: true }) });
  expect(pinned.body.session.pinned).toBe(true);
  expect(host.states.get(archived.body.session.sessionPath)?.pinned).toBe(true);

  const files = await api(`/files?cwd=${encodeURIComponent(CWD)}`);
  expect(files.body.files.map((file: { path: string }) => file.path)).toContain("src/main.ts");
  expect((await api("/files?cwd=/not/on/the/host")).status).toBe(400);
  expect((await api("/health")).body).toMatchObject({ ok: true, piVersion: "1.5.2", sessions: [] });
});

test("a new session is identified from the child and carries the host's machine id", async () => {
  const started = await api("/sessions", { method: "POST", body: JSON.stringify({ cwd: CWD, profile: "coding", resume: null }) });
  expect(started.status).toBe(200);
  expect(started.body.session).toMatchObject({ sessionPath: `${LIVE_ROOT}/new.jsonl`, machine: "box-1", cwd: CWD });
  expect(host.spawned.at(-1)).toEqual({ profile: "coding", cwd: CWD, piArgs: ["--mode", "rpc"] });
});

/** The socket's next `type` event. */
function nextEvent(socket: WebSocket, type: "open" | "message"): Promise<Event> {
  const { promise, resolve } = Promise.withResolvers<Event>();
  socket.addEventListener(type, resolve, { once: true });
  return promise;
}

test("listing without a cwd, search and the sessions_changed broadcast go through the Host", async () => {
  const otherPath = `${LIVE_ROOT}/other/2026-11-19_other.jsonl`;
  host.sessions.set(otherPath, {
    sessionId: "01a1f3fe-0000-7000-8000-00000000000b",
    sessionPath: otherPath,
    realPath: otherPath,
    cwd: "/remote/other",
    createdAt: "2026-11-19T11:00:00.000Z",
    modifiedAt: "2026-11-19T11:05:00.000Z",
    firstMessage: "hello other",
    messageCount: 1,
    tokens: 0,
    archived: false,
  });
  const listed = await api("/sessions");
  expect(listed.body.sessions.map((session: { cwd: string }) => session.cwd)).toContain("/remote/other");
  expect((await api("/sessions?cwd=relative")).status).toBe(400);

  const searched = await api("/search?q=other");
  expect(searched.body).toEqual({
    hits: [{ session: expect.objectContaining({ sessionPath: otherPath, machine: "box-1" }), snippet: "hello other", entryId: "e1", matches: [] }],
  });
  expect((await api("/search?q=")).body).toEqual({ hits: [] });

  const options = { headers: { authorization: `Bearer ${TOKEN}` } } as unknown as string[];
  const sockets = [0, 1].map(() => new WebSocket(`ws://127.0.0.1:${server.port}/ws`, options));
  const received = sockets.map((socket) => nextEvent(socket, "message"));
  await Promise.all(sockets.map((socket) => nextEvent(socket, "open")));
  for (const notify of host.watchers) notify([otherPath]);
  const frames = (await Promise.all(received)).map((event) => JSON.parse(String((event as MessageEvent).data)));
  expect(frames).toEqual([0, 1].map(() => ({ type: "sessions_changed", sessionPaths: [otherPath] })));
  for (const socket of sockets) socket.close();
});

test("a fork is written by the Host, gets the source's private data and runs under a lock on the new file", async () => {
  const sourcePath = `${LIVE_ROOT}/project/2026-11-19_source.jsonl`;
  host.sessions.set(sourcePath, {
    sessionId: "01a1f3fe-0000-7000-8000-00000000000d",
    sessionPath: sourcePath,
    realPath: sourcePath,
    cwd: CWD,
    createdAt: "2026-11-19T12:00:00.000Z",
    modifiedAt: "2026-11-19T12:05:00.000Z",
    firstMessage: "fork me",
    messageCount: 2,
    tokens: 0,
    archived: false,
  });
  host.states.set(sourcePath, { revision: 3, pinned: true, data: { draft: "x" } });
  const fork = (entryId: string) =>
    api("/sessions", { method: "POST", body: JSON.stringify({ cwd: CWD, profile: "coding", resume: { forkOf: { sessionPath: sourcePath, entryId } } }) });

  expect(await fork("nope")).toEqual({ status: 404, body: { error: "no_entry" } });
  const started = await fork("e1");
  const forkPath = `${LIVE_ROOT}/project/fork.jsonl`;
  expect(started.status).toBe(200);
  expect(started.body.session).toMatchObject({ sessionPath: forkPath, parentSession: sourcePath, machine: "box-1" });
  expect(host.spawned.at(-1)).toEqual({ profile: "coding", cwd: CWD, piArgs: ["--mode", "rpc", "--session", forkPath] });
  expect([...host.locks.values()]).toContain(forkPath);
  expect(host.states.get(forkPath)).toEqual({ revision: 0, pinned: false, data: { draft: "x" } });
  expect(host.states.get(sourcePath)).toEqual({ revision: 3, pinned: true, data: { draft: "x" } });
});

test("a document fork carries the source's profile marker and refuses a coding start", async () => {
  const sourcePath = `${LIVE_ROOT}/project/2026-11-19_document.jsonl`;
  host.sessions.set(sourcePath, {
    sessionId: "01a1f3fe-0000-7000-8000-00000000000e",
    sessionPath: sourcePath,
    realPath: sourcePath,
    cwd: CWD,
    createdAt: "2026-11-19T13:00:00.000Z",
    modifiedAt: "2026-11-19T13:05:00.000Z",
    firstMessage: "about the paper",
    messageCount: 2,
    tokens: 0,
    archived: false,
  });
  host.states.set(sourcePath, { revision: 1, pinned: false, data: {}, profile: "document" });
  // The previous test's fork child holds the fork path the fake Host always writes.
  await api("/sessions/01a1f3fe-0000-7000-8000-00000000000c/close", { method: "POST" });
  const fork = (profile: object) =>
    api("/sessions", {
      method: "POST",
      body: JSON.stringify({ cwd: CWD, ...profile, resume: { forkOf: { sessionPath: sourcePath, entryId: "e1" } } }),
    });

  expect(await fork({ profile: "coding" })).toEqual({ status: 409, body: { error: "profile_mismatch" } });
  const started = await fork({ profile: "document", systemPrompt: "Answer only with DOC" });
  const forkPath = `${LIVE_ROOT}/project/fork.jsonl`;
  expect(started.status).toBe(200);
  expect(started.body.session).toMatchObject({ sessionPath: forkPath, profile: "document" });
  expect(host.states.get(forkPath)).toEqual({ revision: 0, pinned: false, data: {}, profile: "document" });
  expect(host.spawned.at(-1)).toEqual({
    profile: "document",
    cwd: CWD,
    piArgs: ["--mode", "rpc", "--append-system-prompt", "Answer only with DOC", "--session", forkPath],
  });
  const listed = await api(`/sessions?cwd=${encodeURIComponent(CWD)}`);
  const documents = listed.body.sessions.filter((session: { profile: string }) => session.profile === "document");
  expect(documents.map((session: { sessionPath: string }) => session.sessionPath).sort()).toEqual([sourcePath, forkPath].sort());
});

test("a delete stops the session's child, then removes file and private state through the Host", async () => {
  const fork = { sessionId: "01a1f3fe-0000-7000-8000-00000000000c", sessionPath: `${LIVE_ROOT}/project/fork.jsonl` };
  expect(host.locks.size).toBeGreaterThan(0);
  expect(supervisor.liveStatus(fork.sessionPath)).toBeDefined();

  expect(await api(`/sessions/${fork.sessionId}`, { method: "DELETE" })).toEqual({ status: 204, body: null });
  expect(supervisor.liveStatus(fork.sessionPath)).toBeUndefined();
  expect(host.sessions.has(fork.sessionPath)).toBe(false);
  expect(host.states.has(fork.sessionPath)).toBe(false);
  expect([...host.locks.values()]).not.toContain(fork.sessionPath);
  expect(await api(`/sessions/${fork.sessionId}`, { method: "DELETE" })).toEqual({ status: 404, body: { error: "not_found" } });
});
