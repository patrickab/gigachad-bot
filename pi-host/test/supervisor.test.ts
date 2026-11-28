/** Supervisor tests over HTTP and `/ws` against the fake `pi` fixture (no model calls). */
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, renameSync, rmSync, statSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Server } from "bun";
import { loadToken } from "../src/auth.ts";
import { startServer } from "../src/server.ts";
import type { PiChild } from "../src/host/index.ts";
import { childEnv, gatewayEnv, sandboxEnv } from "../src/host/local/env.ts";
import { GATEWAY_EXTENSION } from "../src/host/local/index.ts";
import { tryLockFile } from "../src/host/local/lock.ts";
import { cwdDirName } from "../src/host/local/paths.ts";
import { createSupervisor, type SocketData, type Supervisor } from "../src/supervisor/index.ts";
import type { SupervisorHost } from "../src/supervisor/host.ts";
import { LiveSession } from "../src/supervisor/session.ts";

type Frame = { type: string; sessionId: string; [field: string]: any };

const TOKEN = "test-token";
const TOKENS = { full: TOKEN, gateway: "test-gateway-token" };
const root = mkdtempSync(join(tmpdir(), "pi-host-supervisor-"));
const agentDir = join(root, "agent");
const savedEnv = { agentDir: process.env.PI_CODING_AGENT_DIR, sessionDir: process.env.PI_CODING_AGENT_SESSION_DIR };

/** Executable that runs `script` with this Bun. */
function wrapper(name: string, script: string): string {
  const path = join(root, name);
  writeFileSync(path, `#!/bin/sh\n${script}\n`);
  chmodSync(path, 0o755);
  return path;
}

const fakePi = wrapper("pi", `exec "${process.execPath}" "${join(import.meta.dir, "fixtures", "fake-pi.ts")}" "$@"`);

function workdir(name: string): string {
  const path = join(root, "work", name);
  mkdirSync(path, { recursive: true });
  return path;
}

async function until<T>(probe: () => T | undefined, timeoutMs = 20_000): Promise<T> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const value = probe();
    if (value !== undefined) return value;
    await Bun.sleep(10);
  }
  throw new Error("timed out");
}

class Client {
  readonly frames: Frame[] = [];
  /** Host frames (`sessions_changed`, M25): no session, no stream, kept apart from the session frames. */
  readonly hostFrames: { type: string; sessionPaths: string[] }[] = [];
  /** Frames whose `sequence` is not the previous one of their `streamId` plus 1 (M16). */
  readonly sequenceGaps: Frame[] = [];
  private readonly lastSequence = new Map<string, number>();
  private cursor = 0;

  private constructor(readonly socket: WebSocket) {
    socket.addEventListener("message", (event) => {
      const frame: Frame = JSON.parse(String(event.data));
      if (frame.type === "sessions_changed") {
        this.hostFrames.push(frame as unknown as Client["hostFrames"][number]);
        return;
      }
      if (frame.streamId !== null) {
        if (frame.sequence !== (this.lastSequence.get(frame.streamId) ?? -1) + 1) this.sequenceGaps.push(frame);
        this.lastSequence.set(frame.streamId, frame.sequence);
      }
      this.frames.push(frame);
    });
  }

  static async open(server: Server<SocketData>): Promise<Client> {
    // Bun's WebSocket takes `{headers}`; the DOM typing it is declared with only knows protocols.
    const options = { headers: { authorization: `Bearer ${TOKEN}` } } as unknown as string[];
    const socket = new WebSocket(`ws://127.0.0.1:${server.port}/ws`, options);
    const client = new Client(socket);
    await until(() => (socket.readyState === WebSocket.OPEN ? true : undefined));
    return client;
  }

  send(frame: object): void {
    this.socket.send(JSON.stringify(frame));
  }

  /** The next frame after the cursor matching `match`, moving the cursor past it. */
  async next(match: (frame: Frame) => boolean): Promise<Frame> {
    return until(() => {
      const index = this.frames.findIndex((frame, position) => position >= this.cursor && match(frame));
      if (index < 0) return undefined;
      this.cursor = index + 1;
      return this.frames[index];
    });
  }

  /** Frames from `attached` through `replay_end` of an attach. */
  async attach(sessionId: string): Promise<{ attached: Frame; replay: Frame[] }> {
    this.send({ type: "attach", sessionId });
    const attached = await this.next((frame) => frame.type === "attached" && frame.sessionId === sessionId);
    const start = this.cursor;
    await this.next((frame) => frame.type === "replay_end");
    const frames = this.frames.slice(start, this.cursor - 1);
    expect(frames[0]?.type).toBe("replay_start");
    return { attached, replay: frames.slice(1) };
  }

  /** Frames from `resync` through `replay_end` of a resync. */
  async resync(sessionId: string, cursor: string | null): Promise<{ resync: Frame; replay: Frame[] }> {
    this.send({ type: "resync", sessionId, cursor });
    const resync = await this.next((frame) => frame.type === "resync" && frame.sessionId === sessionId);
    const start = this.cursor;
    await this.next((frame) => frame.type === "replay_end");
    const frames = this.frames.slice(start, this.cursor - 1);
    expect(frames[0]?.type).toBe("replay_start");
    return { resync, replay: frames.slice(1) };
  }

  pi(match: (record: any) => boolean): Promise<Frame> {
    return this.next((frame) => frame.type === "pi" && match(frame.record));
  }

  command(sessionId: string, command: object): void {
    this.send({ type: "command", sessionId, command });
  }
}

let supervisor: Supervisor;
let server: Server<SocketData>;
const clients: Client[] = [];

async function api(path: string, init: RequestInit = {}, serverPort = server.port): Promise<Response> {
  return fetch(`http://127.0.0.1:${serverPort}${path}`, {
    ...init,
    headers: { authorization: `Bearer ${TOKEN}`, "content-type": "application/json", ...init.headers },
  });
}

async function startSession(body: object, serverPort = server.port): Promise<Response> {
  return api("/sessions", { method: "POST", body: JSON.stringify({ profile: "coding", resume: null, ...body }) }, serverPort);
}

async function client(target = server): Promise<Client> {
  const opened = await Client.open(target);
  clients.push(opened);
  return opened;
}

beforeAll(async () => {
  process.env.PI_CODING_AGENT_DIR = agentDir;
  delete process.env.PI_CODING_AGENT_SESSION_DIR;
  // The watcher (M25) starts with the supervisor; without a root it would only retry after 5 s.
  mkdirSync(join(agentDir, "sessions"), { recursive: true });
  mkdirSync(join(agentDir, "factory", "sessions"), { recursive: true });
  supervisor = await createSupervisor({ piBin: fakePi, idleMs: 3_600_000, isPinned: async () => false });
  server = startServer({ hostname: "127.0.0.1", port: 0, tokens: TOKENS, supervisor });
});

afterAll(async () => {
  for (const opened of clients) opened.socket.close();
  await supervisor.shutdown();
  server.stop(true);
  if (savedEnv.agentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
  else process.env.PI_CODING_AGENT_DIR = savedEnv.agentDir;
  if (savedEnv.sessionDir !== undefined) process.env.PI_CODING_AGENT_SESSION_DIR = savedEnv.sessionDir;
  rmSync(root, { recursive: true, force: true });
});

describe("auth", () => {
  test("rejects a missing bearer and any Origin, HTTP and WS alike", async () => {
    const base = `http://127.0.0.1:${server.port}`;
    expect((await fetch(`${base}/health`)).status).toBe(401);
    expect((await fetch(`${base}/health`, { headers: { authorization: "Bearer wrong" } })).status).toBe(401);
    const withOrigin = { authorization: `Bearer ${TOKEN}`, origin: "http://localhost:2999" };
    expect((await fetch(`${base}/health`, { headers: withOrigin })).status).toBe(403);
    expect((await fetch(`${base}/ws`, { headers: { ...withOrigin, upgrade: "websocket" } })).status).toBe(403);
    expect((await fetch(`${base}/ws`, { headers: { upgrade: "websocket" } })).status).toBe(401);
    expect(await (await api("/health")).json()).toEqual({ ok: true, piVersion: "1.5.2", sessions: [] });
  });
});

test("starts on a Pi binary of any version", async () => {
  const otherPi = wrapper("pi-other", "echo 9.9.9");
  await expect(createSupervisor({ piBin: otherPi })).resolves.toBeDefined();
});

describe("live sessions", () => {
  const cwd = workdir("repo");
  let sessionId = "";
  let sessionPath = "";
  let first: Client;
  let second: Client;

  test("POST /sessions spawns a child with the requested model and thinking", async () => {
    const response = await startSession({ cwd, model: { provider: "faux", id: "faux-2" }, thinking: "high" });
    expect(response.status).toBe(200);
    const { session } = await response.json();
    expect(session).toMatchObject({ cwd, machine: "local", live: { running: false, waiting: false, attached: 0 } });
    sessionId = session.sessionId;
    sessionPath = session.sessionPath;
    expect(sessionPath.endsWith(`_${sessionId}.jsonl`)).toBe(true);
  });

  test("rejects an invalid body and a model Pi refuses", async () => {
    expect((await startSession({ cwd: "relative" })).status).toBe(400);
    expect((await startSession({ cwd, profile: "remote" })).status).toBe(400);
    const refused = await startSession({ cwd, model: { provider: "bad", id: "x" } });
    expect(refused.status).toBe(400);
    expect(await refused.json()).toMatchObject({ error: "model" });
  });

  test("a model of a provider the child lacks falls back to the gateway provider", async () => {
    const response = await startSession({ cwd: workdir("keyless"), model: { provider: "keyless", id: "acme/x" } });
    expect(response.status).toBe(200);
    const { session } = await response.json();
    const driver = await client();
    expect((await driver.attach(session.sessionId)).attached.state.model).toEqual({ provider: "gateway", id: "keyless/acme/x" });
  });

  test("two clients attach, both receive events, either may command", async () => {
    first = await client();
    second = await client();
    for (const attaching of [first, second]) {
      const { attached, replay } = await attaching.attach(sessionId);
      expect(attached.state).toMatchObject({ sessionId, isStreaming: false, thinkingLevel: "high", model: { id: "faux-2" } });
      expect(attached.entries).toEqual({ entries: [], leafId: null });
      expect(attached.ui).toEqual({ dialogs: [], statuses: {}, widgets: {}, queue: { steering: [], followUp: [] } });
      expect(replay).toEqual([]);
    }
    first.command(sessionId, { id: "from-first", type: "prompt", message: "hello" });
    for (const receiving of [first, second]) {
      await receiving.pi((record) => record.type === "response" && record.id === "from-first");
      await receiving.pi((record) => record.type === "agent_settled");
    }
    second.command(sessionId, { id: "from-second", type: "get_state" });
    const state = await first.pi((record) => record.id === "from-second");
    expect(state.record.data.messageCount).toBe(2);
    expect(supervisor.liveStatus(sessionPath)).toEqual({ running: false, waiting: false, attached: 2 });
  });

  test("attach mid-turn replays the records since the streaming message started", async () => {
    first.command(sessionId, { type: "prompt", message: "hold" });
    await first.pi((record) => record.type === "message_update");
    expect(supervisor.liveStatus(sessionPath)?.running).toBe(true);

    const late = await client();
    const { attached, replay } = await late.attach(sessionId);
    expect(attached.state.isStreaming).toBe(true);
    expect(attached.entries.entries.at(-1).message).toMatchObject({ role: "user", content: "hold" });
    expect(replay.map((frame) => frame.record.type)).toEqual(["message_start", "message_update"]);
    expect(replay[1]!.record.assistantMessageEvent.delta).toBe("partial ");

    late.command(sessionId, { type: "abort" });
    for (const receiving of [first, second, late]) {
      expect((await receiving.pi((record) => record.type === "agent_settled")).record.aborted).toBe(true);
    }
    late.send({ type: "detach", sessionId });
  });

  test("open dialogs are session state and closing one reaches every client", async () => {
    first.command(sessionId, { type: "prompt", message: "dialog" });
    const request = await second.pi((record) => record.type === "extension_ui_request" && record.method === "select");
    expect(supervisor.liveStatus(sessionPath)?.waiting).toBe(true);

    const late = await client();
    const { attached } = await late.attach(sessionId);
    expect(attached.ui.dialogs.map((dialog: { id: string }) => dialog.id)).toEqual([request.record.id]);

    late.command(sessionId, { type: "extension_ui_response", id: request.record.id, value: "b" });
    for (const receiving of [first, second, late]) {
      expect((await receiving.next((frame) => frame.type === "ui_closed")).id).toBe(request.record.id);
    }
    await first.pi((record) => record.method === "notify" && record.message === "chose b");
    expect(supervisor.liveStatus(sessionPath)?.waiting).toBe(false);
    late.socket.close();
  });

  test("refuses commands that switch the session file and commands from unattached clients", async () => {
    first.command(sessionId, { type: "new_session" });
    expect((await first.next((frame) => frame.type === "error")).message).toContain("POST /sessions");
    const stranger = await client();
    stranger.command(sessionId, { type: "get_state" });
    expect((await stranger.next((frame) => frame.type === "error")).message).toBe("not attached");
    stranger.send({ type: "attach", sessionId: "missing" });
    expect((await stranger.next((frame) => frame.type === "error")).sessionId).toBe("missing");
  });

  test("a held session file gives 409, resume returns the live child", async () => {
    expect((await api(`/sessions/${sessionId}/close`, { method: "POST" })).status).toBe(204);
    expect((await first.next((frame) => frame.type === "exit")).sessionId).toBe(sessionId);
    expect(supervisor.liveStatus(sessionPath)).toBeUndefined();

    const holder = Bun.spawn(["flock", "--nonblock", "--no-fork", sessionPath, "sleep", "30"]);
    await until(() => {
      const probe = tryLockFile(sessionPath);
      probe?.release();
      return probe ? undefined : true;
    });
    const locked = await startSession({ cwd, resume: { sessionPath } });
    expect(locked.status).toBe(409);
    expect(await locked.json()).toEqual({ error: "locked" });
    holder.kill();
    await holder.exited;

    const resumed = await (await startSession({ cwd, resume: { sessionPath } })).json();
    expect(resumed.session).toMatchObject({ sessionId, sessionPath, messageCount: 4 });
    const again = await (await startSession({ cwd, resume: { sessionPath } })).json();
    expect(again.session.live).toEqual({ running: false, waiting: false, attached: 0 });
    const contender = Bun.spawnSync(["flock", "--nonblock", "--conflict-exit-code", "75", sessionPath, "true"]);
    expect(contender.exitCode).toBe(75);
  });

  test("every frame carries streamId and sequence, a new child gets a new streamId", async () => {
    const previousStream = first.frames.find((frame) => frame.type === "attached")!.streamId;
    const { attached } = await first.attach(sessionId);
    expect(attached.streamId).not.toBe(previousStream);
    expect(attached.sequence).toBe(0);
    for (const opened of clients) {
      expect(opened.frames.every((frame) => "streamId" in frame && typeof frame.sequence === "number")).toBe(true);
      expect(opened.sequenceGaps).toEqual([]);
    }
  });
});

test("a burst of session writes outside any child reaches every client as one sessions_changed frame", async () => {
  const first = await client();
  const second = await client();
  const dir = join(agentDir, "sessions", "--tui-cwd--");
  mkdirSync(dir, { recursive: true });
  const paths = ["a", "b"].map((name) => join(dir, `2026-11-19T10-00-00-000Z_${name}.jsonl`));
  for (const path of paths) writeFileSync(path, `${JSON.stringify({ type: "session", version: 3, id: path, cwd: "/tmp" })}\n`);
  for (const path of paths) writeFileSync(path, "{}\n", { flag: "a" });
  writeFileSync(join(dir, "notes.txt"), "not a session");

  const ours = (target: Client) => target.hostFrames.filter((frame) => frame.sessionPaths.some((path) => paths.includes(path)));
  await until(() => (ours(first).length > 0 && ours(second).length > 0 ? true : undefined));
  // Real fs.watch events against the real clock: wait past a further 500 ms window to prove no second frame came.
  await Bun.sleep(800);
  for (const target of [first, second]) {
    expect(ours(target)).toEqual([{ type: "sessions_changed", sessionPaths: expect.arrayContaining(paths) }]);
    expect(ours(target)[0]!.sessionPaths.every((path) => path.endsWith(".jsonl"))).toBe(true);
    expect(target.frames).toEqual([]);
  }
});

test("idle eviction skips pinned, streaming and liveness-reporting sessions", async () => {
  const pinned = new Set<string>();
  const idleMs = 60_000;
  const evicting = await createSupervisor({ piBin: fakePi, idleMs, isPinned: async (path) => pinned.has(path) });
  const evictingServer = startServer({ hostname: "127.0.0.1", port: 0, tokens: TOKENS, supervisor: evicting });
  try {
    const start = async (name: string) =>
      (await (await startSession({ cwd: workdir(name) }, evictingServer.port)).json()).session as {
        sessionId: string;
        sessionPath: string;
      };
    const idle = await start("idle");
    const pinnedSession = await start("pinned");
    pinned.add(pinnedSession.sessionPath);
    const streaming = await start("streaming");
    const busy = await start("busy");

    const driver = await client(evictingServer);
    await driver.attach(streaming.sessionId);
    driver.command(streaming.sessionId, { type: "prompt", message: "hold" });
    await driver.pi((record) => record.type === "message_update");
    await driver.attach(busy.sessionId);
    driver.command(busy.sessionId, { type: "prompt", message: "status session-liveness subagents" });
    await driver.pi((record) => record.method === "setStatus");
    driver.socket.close();
    await until(() => (evicting.liveStatus(busy.sessionPath)?.attached === 0 ? true : undefined));
    // The test owns the clock: one eviction pass as if every session had been idle past the limit.
    await (evicting as SupervisorHost).evictIdle(Date.now() + 2 * idleMs);
    await until(() => (evicting.liveStatus(idle.sessionPath) === undefined ? true : undefined));
    expect(evicting.liveStatus(pinnedSession.sessionPath)).toBeDefined();
    expect(evicting.liveStatus(streaming.sessionPath)?.running).toBe(true);
    expect(evicting.liveStatus(busy.sessionPath)).toBeDefined();
  } finally {
    await evicting.shutdown();
    evictingServer.stop(true);
  }
});

test("children get an allowlisted environment without backend secrets", () => {
  const env = childEnv(["EXTRA_ALLOWED"], {
    PATH: "/bin",
    HOME: "/home/someone",
    LC_ALL: "C",
    PI_CODING_AGENT_DIR: "/scratch",
    EXTRA_ALLOWED: "yes",
    ANTHROPIC_API_KEY: "secret",
    POSTGRES_PASSWORD_DEV: "secret",
    GIGACHAD_PI_HOST_URL: "http://127.0.0.1:8300",
  });
  expect(env).toEqual({
    PATH: "/bin",
    HOME: "/home/someone",
    LC_ALL: "C",
    PI_CODING_AGENT_DIR: "/scratch",
    EXTRA_ALLOWED: "yes",
    PI_SKIP_VERSION_CHECK: "1",
  });
});

/** A session file with a header and `entries`, under the scratch agent dir's sessions root. */
function writeSessionFile(cwd: string, entries: object[]): { sessionId: string; sessionPath: string } {
  const sessionId = crypto.randomUUID();
  const dir = join(agentDir, "sessions", `--${cwd.replace(/^\//, "").replace(/[/\\:]/g, "-")}--`);
  mkdirSync(dir, { recursive: true });
  const sessionPath = join(dir, `2026-11-19T10-00-00-000Z_${sessionId}.jsonl`);
  const header = { type: "session", version: 3, id: sessionId, timestamp: new Date().toISOString(), cwd };
  writeFileSync(sessionPath, [header, ...entries].map((line) => `${JSON.stringify(line)}\n`).join(""));
  return { sessionId, sessionPath };
}

function messageEntry(id: string, parentId: string | null, role: "user" | "assistant", text: string): object {
  const content = role === "user" ? text : [{ type: "text", text }];
  return { type: "message", id, parentId, timestamp: new Date().toISOString(), message: { role, content, timestamp: 0 } };
}

function entryIds(entries: { id: string }[]): string[] {
  return entries.map((entry) => entry.id);
}

test("an attached frame above Bun's default 16 MB backpressure limit keeps its replay markers", async () => {
  const cwd = workdir("large");
  const { sessionPath } = writeSessionFile(cwd, [
    messageEntry("u1", null, "user", "x".repeat(24 * 1024 * 1024)),
    messageEntry("a1", "u1", "assistant", "ok"),
  ]);
  const { session } = await (await startSession({ cwd, resume: { sessionPath } })).json();
  const reader = await client();
  const { attached, replay } = await reader.attach(session.sessionId);
  expect(entryIds(attached.entries.entries)).toEqual(["u1", "a1"]);
  expect(replay).toEqual([]);
  expect(reader.sequenceGaps).toEqual([]);
  await api(`/sessions/${session.sessionId}/close`, { method: "POST" });
}, 30_000);

test("archive during a spawn is refused and the resumed child keeps its file", async () => {
  const slowPi = wrapper("pi-slow", `[ "$1" = --version ] || sleep 1.5\nexec "${fakePi}" "$@"`);
  const slow = await createSupervisor({ piBin: slowPi, idleMs: 3_600_000, isPinned: async () => false });
  const slowServer = startServer({ hostname: "127.0.0.1", port: 0, tokens: TOKENS, supervisor: slow });
  try {
    const cwd = workdir("archive-race");
    const { sessionId, sessionPath } = writeSessionFile(cwd, [
      messageEntry("u1", null, "user", "hi"),
      messageEntry("a1", "u1", "assistant", "hello"),
    ]);
    const spawning = startSession({ cwd, resume: { sessionPath } }, slowServer.port);
    await Bun.sleep(400);
    const archive = await api(`/sessions/${sessionId}/archive`, { method: "POST" }, slowServer.port);
    expect(archive.status).toBe(409);
    expect(await archive.json()).toEqual({ error: "live" });
    const spawned = await spawning;
    expect(spawned.status).toBe(200);
    expect((await spawned.json()).session).toMatchObject({ sessionId, archived: false });
    expect(existsSync(sessionPath)).toBe(true);
  } finally {
    await slow.shutdown();
    slowServer.stop(true);
  }
});

test("archive closes a detached idle child and refuses an attached one", async () => {
  const cwd = workdir("archive-idle");
  const { sessionId, sessionPath } = writeSessionFile(cwd, [
    messageEntry("u1", null, "user", "hi"),
    messageEntry("a1", "u1", "assistant", "hello"),
  ]);
  expect((await startSession({ cwd, resume: { sessionPath } })).status).toBe(200);
  const viewer = await client();
  await viewer.attach(sessionId);
  const refused = await api(`/sessions/${sessionId}/archive`, { method: "POST" });
  expect(refused.status).toBe(409);
  expect(await refused.json()).toEqual({ error: "live" });
  expect(supervisor.liveStatus(sessionPath)).toBeDefined();

  viewer.send({ type: "detach", sessionId });
  await until(() => (supervisor.liveStatus(sessionPath)?.attached === 0 ? true : undefined));
  const archived = await api(`/sessions/${sessionId}/archive`, { method: "POST" });
  expect(archived.status).toBe(200);
  expect((await archived.json()).session).toMatchObject({ sessionId, archived: true, live: null });
  expect(supervisor.liveStatus(sessionPath)).toBeUndefined();
  expect(existsSync(sessionPath)).toBe(false);
});

describe("fork", () => {
  const cwd = workdir("fork");
  const labelEntry = (id: string, parentId: string, targetId: string, label: string) => ({
    type: "label",
    id,
    parentId,
    timestamp: "2026-11-19T10:00:01.000Z",
    targetId,
    label,
  });
  // u1 -> a1 -> l1 (labels u1) -> u3 -> a3 -> l2 (labels a3) is abandoned; u1 -> a1 -> l1 -> u2 -> a2 is the leaf.
  const { sessionId: sourceId, sessionPath: source } = writeSessionFile(cwd, [
    messageEntry("u1", null, "user", "first"),
    messageEntry("a1", "u1", "assistant", "one"),
    labelEntry("l1", "a1", "u1", "start"),
    messageEntry("u3", "l1", "user", "abandoned"),
    messageEntry("a3", "u3", "assistant", "three"),
    labelEntry("l2", "a3", "a3", "dead end"),
    messageEntry("u2", "l1", "user", "second"),
    messageEntry("a2", "u2", "assistant", "two"),
  ]);
  const sourceText = readFileSync(source, "utf8");
  writeFileSync(`${source}.gigachad.json`, JSON.stringify({ revision: 4, pinned: true, data: { draft: "keep" } }));

  async function fork(entryId: string): Promise<{ session: any; lines: any[] }> {
    const response = await startSession({ cwd, resume: { forkOf: { sessionPath: source, entryId } } });
    expect(response.status).toBe(200);
    const { session } = await response.json();
    const lines = readFileSync(session.sessionPath, "utf8").trim().split("\n").map((line) => JSON.parse(line));
    return { session, lines };
  }

  test("a fork mid-branch keeps the root..entry path and its labels, copies private state, holds the new file", async () => {
    const { session, lines } = await fork("u2");
    const [header, ...entries] = lines;
    expect(header).toMatchObject({ type: "session", version: 3, id: session.sessionId, cwd, parentSession: source });
    expect(session).toMatchObject({ cwd, parentSession: source, messageCount: 3, live: { running: false } });
    expect(session.sessionPath).toBe(join(source, "..", `${header.timestamp.replace(/[:.]/g, "-")}_${header.id}.jsonl`));
    expect(entries.map((entry) => [entry.id, entry.parentId])).toEqual([
      ["u1", null],
      ["a1", "u1"],
      ["u2", "a1"],
      [entries[3].id, "u2"],
    ]);
    expect(entries[3]).toMatchObject({ type: "label", targetId: "u1", label: "start", timestamp: "2026-11-19T10:00:01.000Z" });

    expect(readFileSync(source, "utf8")).toBe(sourceText);
    expect(JSON.parse(readFileSync(`${session.sessionPath}.gigachad.json`, "utf8"))).toEqual({
      revision: 0,
      pinned: false,
      data: { draft: "keep" },
    });
    const contender = Bun.spawnSync(["flock", "--nonblock", "--conflict-exit-code", "75", session.sessionPath, "true"]);
    expect(contender.exitCode).toBe(75);

    const reader = await client();
    await reader.attach(session.sessionId);
    const { resync } = await reader.resync(session.sessionId, null);
    expect(resync.entries.filter((entry: any) => entry.type === "message").map((entry: any) => entry.id)).toEqual(["u1", "a1", "u2"]);
  });

  test("forking at the leaf clones the active branch", async () => {
    const { session, lines } = await fork("a2");
    expect(lines[0]).toMatchObject({ id: session.sessionId, parentSession: source });
    expect(session.sessionId).not.toBe(sourceId);
    expect(lines.slice(1).map((entry) => entry.id).slice(0, 4)).toEqual(["u1", "a1", "u2", "a2"]);
    expect(lines.slice(1).filter((entry) => entry.type === "label")).toEqual([
      expect.objectContaining({ parentId: "a2", targetId: "u1", label: "start" }),
    ]);
    expect(readFileSync(source, "utf8")).toBe(sourceText);
  });

  test("an unknown source or entry gives 404, an invalid fork 400", async () => {
    const missing = await startSession({ cwd, resume: { forkOf: { sessionPath: join(cwd, "none.jsonl"), entryId: "u1" } } });
    expect([missing.status, await missing.json()]).toEqual([404, { error: "not_found" }]);
    const noEntry = await startSession({ cwd, resume: { forkOf: { sessionPath: source, entryId: "zz" } } });
    expect([noEntry.status, await noEntry.json()]).toEqual([404, { error: "no_entry" }]);
    for (const forkOf of [{ sessionPath: "relative.jsonl", entryId: "u1" }, { sessionPath: source, entryId: "" }, { sessionPath: source }]) {
      expect((await startSession({ cwd, resume: { forkOf } })).status).toBe(400);
    }
  });
});

test("a child's exit is handled while a descendant still holds its stdout and stderr", async () => {
  const { session } = await (await startSession({ cwd: workdir("orphan") })).json();
  const driver = await client();
  await driver.attach(session.sessionId);
  driver.command(session.sessionId, { id: "orphan", type: "prompt", message: "orphan" });
  const orphanPid: number = (await driver.pi((record) => record.id === "orphan")).record.data.pid;
  try {
    const started = Date.now();
    expect((await api(`/sessions/${session.sessionId}/close`, { method: "POST" })).status).toBe(204);
    expect(Date.now() - started).toBeLessThan(5_000);
    expect(supervisor.liveStatus(session.sessionPath)).toBeUndefined();
    expect((await driver.next((frame) => frame.type === "exit")).sessionId).toBe(session.sessionId);
  } finally {
    process.kill(orphanPid);
  }
}, 15_000);

test("a session started under a symlinked agent dir resumes as the same live child", async () => {
  const link = join(root, "agent-link");
  mkdirSync(agentDir, { recursive: true });
  symlinkSync(agentDir, link);
  process.env.PI_CODING_AGENT_DIR = link;
  try {
    const cwd = workdir("symlinked");
    const { session } = await (await startSession({ cwd })).json();
    const driver = await client();
    await driver.attach(session.sessionId);
    driver.command(session.sessionId, { type: "prompt", message: "hello" });
    await driver.pi((record) => record.type === "agent_settled");
    const answer = await startSession({ cwd, resume: { sessionPath: session.sessionPath } });
    expect(answer.status).toBe(200);
    expect((await answer.json()).session.sessionId).toBe(session.sessionId);
  } finally {
    process.env.PI_CODING_AGENT_DIR = agentDir;
  }
});

describe("resync", () => {
  const cwd = workdir("resync");
  let sessionId = "";
  let sessionPath = "";
  let driver: Client;

  async function prompt(message: string): Promise<void> {
    driver.command(sessionId, { type: "prompt", message });
    await driver.pi((record) => record.type === "agent_settled");
  }

  test("cursor null or unknown replaces with the branch, a known cursor appends what follows it", async () => {
    ({ sessionId, sessionPath } = (await (await startSession({ cwd })).json()).session);
    driver = await client();
    await driver.attach(sessionId);
    await prompt("one");

    const full = await driver.resync(sessionId, null);
    expect(full.resync.mode).toBe("replace");
    expect(full.resync.entries.map((entry: any) => entry.message.role)).toEqual(["user", "assistant"]);
    expect(full.resync.leafId).toBe(full.resync.entries[1].id);
    expect(full.resync.state).toMatchObject({ sessionId, isStreaming: false });
    expect(full.resync.ui).toEqual({ dialogs: [], statuses: {}, widgets: {}, queue: { steering: [], followUp: [] } });
    expect(full.replay).toEqual([]);

    const cursor = full.resync.leafId;
    await prompt("two");
    const { resync } = await driver.resync(sessionId, cursor);
    expect(resync.mode).toBe("append");
    expect(resync.entries.map((entry: any) => entry.parentId)).toEqual([cursor, resync.entries[0].id]);
    expect(resync.entries[1].id).toBe(resync.leafId);
    expect((await driver.resync(sessionId, resync.leafId)).resync).toMatchObject({ mode: "append", entries: [] });

    const unknown = (await driver.resync(sessionId, "no-such-entry")).resync;
    expect(unknown.mode).toBe("replace");
    expect(entryIds(unknown.entries)).toEqual([...entryIds(full.resync.entries), ...entryIds(resync.entries)]);
  });

  test("a resync mid-turn appends the user entry and replays the streaming records", async () => {
    const cursor = (await driver.resync(sessionId, null)).resync.leafId;
    driver.command(sessionId, { type: "prompt", message: "hold" });
    await driver.pi((record) => record.type === "message_update");
    const { resync, replay } = await driver.resync(sessionId, cursor);
    expect(resync.mode).toBe("append");
    expect(resync.entries.map((entry: any) => entry.message.content)).toEqual(["hold"]);
    expect(resync.state.isStreaming).toBe(true);
    expect(replay.map((frame) => frame.record.type)).toEqual(["message_start", "message_update"]);
    driver.command(sessionId, { type: "abort" });
    await driver.pi((record) => record.type === "agent_settled");
  });

  test("a compaction after the cursor replaces", async () => {
    const before = (await driver.resync(sessionId, null)).resync;
    driver.command(sessionId, { type: "prompt", message: "compact" });
    await driver.pi((record) => record.type === "compaction_end");
    const { resync } = await driver.resync(sessionId, before.leafId);
    expect(resync.mode).toBe("replace");
    expect(entryIds(resync.entries.slice(0, -1))).toEqual(entryIds(before.entries));
    expect(resync.entries.at(-1).type).toBe("compaction");
  });

  test("a leaf moved off the cursor's branch replaces with the new branch", async () => {
    const before = (await driver.resync(sessionId, null)).resync;
    const forkPoint = before.entries[1].id;
    driver.command(sessionId, { id: "goto", type: "prompt", message: `goto ${forkPoint}` });
    await driver.pi((record) => record.id === "goto");
    await prompt("branch");
    const { resync } = await driver.resync(sessionId, before.leafId);
    expect(resync.mode).toBe("replace");
    expect(resync.entries).toHaveLength(4);
    expect(entryIds(resync.entries.slice(0, 2))).toEqual(entryIds(before.entries.slice(0, 2)));
    expect(entryIds(resync.entries)).not.toContain(before.leafId);
    expect(resync.entries[2].parentId).toBe(forkPoint);
  });

  test("a resync on a connection that never attached subscribes it", async () => {
    const fresh = await client();
    expect((await fresh.resync(sessionId, null)).resync.mode).toBe("replace");
    await prompt("three");
    await fresh.pi((record) => record.type === "agent_settled");
    expect(supervisor.liveStatus(sessionPath)?.attached).toBe(2);
    expect(fresh.sequenceGaps).toEqual([]);
  });

  test("a bash command's streamed output leaves the replay buffer with its response", async () => {
    driver.command(sessionId, { id: "bash", type: "bash", command: "ls", updates: 5 });
    await driver.pi((record) => record.id === "bash");
    const late = await client();
    expect((await late.attach(sessionId)).replay).toEqual([]);
  });

  test("GET /health lists live sessions with running, attached and pinned", async () => {
    const health = await (await api("/health")).json();
    expect(health).toMatchObject({ ok: true, piVersion: "1.5.2" });
    expect(health.sessions).toContainEqual({ sessionId, sessionPath, running: false, attached: 3, pinned: false });
  });
});

describe("LiveSession against a scripted child", () => {
  /** A child whose stdout the test writes and whose stderr never closes (a descendant holding it). */
  function scriptedChild() {
    const encoder = new TextEncoder();
    let stdout!: ReadableStreamDefaultController<Uint8Array>;
    const commands: Record<string, any>[] = [];
    const kills: string[] = [];
    const exit = Promise.withResolvers<number>();
    const child = {
      stdin: {
        write: (text: string) => {
          for (const line of text.split("\n")) if (line) commands.push(JSON.parse(line));
          return text.length;
        },
        flush: () => 0,
        end: () => undefined,
      },
      stdout: new ReadableStream<Uint8Array>({ start: (controller) => void (stdout = controller) }),
      stderr: new ReadableStream<Uint8Array>(),
      exited: exit.promise,
      kill: (signal: string) => void kills.push(signal),
    };
    const emit = (record: object) => stdout.enqueue(encoder.encode(`${JSON.stringify(record)}\n`));
    return { child: child as unknown as PiChild, commands, kills, exit: exit.resolve, emit };
  }

  test("attach asks for the state only after the entries snapshot", async () => {
    const scripted = scriptedChild();
    const session = new LiveSession(scripted.child, "/", { profile: "coding" }, undefined, () => undefined);
    session.identify("scripted", join(root, "missing.jsonl"));
    const frames: Frame[] = [];
    session.attach({ send: (data) => frames.push(JSON.parse(data)) });
    expect(scripted.commands.map((command) => command.type)).toEqual(["get_entries"]);
    scripted.emit({ type: "response", id: scripted.commands[0]!.id, command: "get_entries", success: true, data: { entries: [], leafId: null } });
    const stateRequest = await until(() => scripted.commands[1]);
    expect(stateRequest.type).toBe("get_state");
    scripted.emit({ type: "agent_settled" });
    scripted.emit({ type: "response", id: stateRequest.id, command: "get_state", success: true, data: { isStreaming: false } });
    await until(() => (frames.length === 4 ? true : undefined));
    expect(frames.map((frame) => (frame.type === "pi" ? frame.record.type : frame.type))).toEqual([
      "attached",
      "replay_start",
      "replay_end",
      "agent_settled",
    ]);
    scripted.exit(0);
    await session.exited;
  });

  test("an attaching client keeps frames from before its snapshot point the snapshot does not carry", async () => {
    const scripted = scriptedChild();
    const session = new LiveSession(scripted.child, "/", { profile: "coding" }, undefined, () => undefined);
    session.identify("scripted", join(root, "missing.jsonl"));
    const frames: Frame[] = [];
    const sink = { send: (data: string) => frames.push(JSON.parse(data)) };
    session.attach(sink);
    expect(session.command(sink, { type: "get_commands", id: "mine" })).toBeUndefined();
    const entriesRequest = scripted.commands[0]!;
    // Before the snapshot point: a replayed record, the answer to the client's command, a transient and a status.
    scripted.emit({ type: "agent_start" });
    scripted.emit({ type: "response", id: "mine", command: "get_commands", success: true, data: { commands: [] } });
    scripted.emit({ type: "extension_ui_request", id: "n1", method: "notify", message: "hi" });
    scripted.emit({ type: "extension_ui_request", id: "s1", method: "setStatus", statusKey: "build", statusText: "on" });
    scripted.emit({ type: "response", id: entriesRequest.id, command: "get_entries", success: true, data: { entries: [], leafId: null } });
    const stateRequest = await until(() => scripted.commands.find((command) => command.type === "get_state"));
    scripted.emit({ type: "message_update" });
    scripted.emit({ type: "response", id: stateRequest.id, command: "get_state", success: true, data: {} });
    await until(() => (frames.length === 7 ? true : undefined));
    expect(frames.map((frame) => (frame.type === "pi" ? `pi:${frame.record.method ?? frame.record.id ?? frame.record.type}` : frame.type))).toEqual([
      "attached",
      "replay_start",
      "pi:agent_start",
      "replay_end",
      "pi:mine",
      "pi:notify",
      "pi:message_update",
    ]);
    expect(frames[0]!.ui.statuses).toEqual({ build: "on" });
    expect(frames.map((frame) => frame.sequence)).toEqual([0, 1, 2, 3, 4, 5, 6]);
    scripted.exit(0);
    await session.exited;
  });

  test("setWidget text widgets are session state, the transient inspect reply is not", async () => {
    const scripted = scriptedChild();
    const session = new LiveSession(scripted.child, "/", { profile: "coding" }, undefined, () => undefined);
    session.identify("scripted", join(root, "missing.jsonl"));
    const widget = (widgetKey: string, widgetLines?: string[]) =>
      scripted.emit({ type: "extension_ui_request", id: crypto.randomUUID(), method: "setWidget", widgetKey, widgetLines });
    widget("subagent-async", ["PI_SUBAGENT_ASYNC_JSON:{}"]);
    widget("subagent-inspect", ["PI_SUBAGENT_INSPECT_JSON:{}"]);
    widget("subagent-inspect");
    widget("gone", ["x"]);
    widget("gone");
    const frames: Frame[] = [];
    session.attach({ send: (data: string) => frames.push(JSON.parse(data)) });
    scripted.emit({ type: "response", id: scripted.commands[0]!.id, command: "get_entries", success: true, data: { entries: [], leafId: null } });
    const stateRequest = await until(() => scripted.commands.find((command) => command.type === "get_state"));
    scripted.emit({ type: "response", id: stateRequest.id, command: "get_state", success: true, data: {} });
    const attached = await until(() => frames.find((frame) => frame.type === "attached"));
    expect(attached.ui.widgets).toEqual({ "subagent-async": ["PI_SUBAGENT_ASYNC_JSON:{}"] });
    scripted.exit(0);
    await session.exited;
  });

  test("a closing child refuses commands, is signalled once and exits although stderr stays open", async () => {
    const scripted = scriptedChild();
    const session = new LiveSession(scripted.child, "/", { profile: "coding" }, undefined, () => undefined);
    session.terminate();
    session.terminate();
    expect(scripted.kills).toEqual(["SIGTERM"]);
    expect(await session.requestAsync({ type: "get_state" })).toMatchObject({ success: false, error: "pi is not running" });
    expect(scripted.commands).toEqual([]);
    scripted.exit(143);
    expect(await session.exited).toBe(143);
  });
});

test("first starts create one 0600 token exclusively and all agree on it", async () => {
  const path = join(root, "state", "pi-host.token");
  const tokens = await Promise.all([loadToken(path), loadToken(path), loadToken(path)]);
  expect(new Set(tokens).size).toBe(1);
  expect(readFileSync(path, "utf8").trim()).toBe(tokens[0]!);
  expect(statSync(path).mode & 0o777).toBe(0o600);
});

describe("profiles (G1, G5, G6)", () => {
  const cwd = workdir("documents");
  const DOC = { profile: "document", systemPrompt: "Answer only with the word DOC" };
  const DOCUMENT_ARGV = ["--mode", "rpc", "--append-system-prompt", "Answer only with the word DOC"];
  let document: { sessionId: string; sessionPath: string };

  async function settle(sessionId: string, message: string): Promise<void> {
    const driver = await client();
    await driver.attach(sessionId);
    driver.command(sessionId, { type: "prompt", message });
    await driver.pi((record) => record.type === "agent_settled");
  }

  async function refusal(body: object): Promise<{ status: number; body: unknown }> {
    const response = await startSession({ cwd, ...body });
    return { status: response.status, body: await response.json() };
  }

  test("a document session runs the user's Pi with the request's prompt appended, marked in its sidecar", async () => {
    const response = await startSession({ cwd, ...DOC });
    expect(response.status).toBe(200);
    const { session } = await response.json();
    expect(session).toMatchObject({ cwd, profile: "document" });
    document = session;
    const driver = await client();
    expect((await driver.attach(document.sessionId)).attached.state.argv.slice(0, DOCUMENT_ARGV.length)).toEqual(DOCUMENT_ARGV);
    await settle(document.sessionId, "hello");
    expect(JSON.parse(readFileSync(`${document.sessionPath}.gigachad.json`, "utf8"))).toMatchObject({ profile: "document" });
  });

  test("tools restrict a document session to them and nothing discovered; systemPrompt and tools are validated per profile", async () => {
    const { session } = await (await startSession({ cwd, ...DOC, tools: ["read", "grep"] })).json();
    const driver = await client();
    expect((await driver.attach(session.sessionId)).attached.state.argv.slice(2)).toEqual([
      "--tools",
      "read,grep",
      "--append-system-prompt",
      "Answer only with the word DOC",
      "--no-skills",
      "--no-extensions",
      "--no-prompt-templates",
      "--no-approve",
      "-e",
      GATEWAY_EXTENSION,
    ]);
    await api(`/sessions/${session.sessionId}/close`, { method: "POST" });

    const invalidPrompt = { status: 400, body: { error: "invalid_system_prompt" } };
    const invalidTools = { status: 400, body: { error: "invalid_tools" } };
    expect(await refusal({ profile: "document" })).toEqual(invalidPrompt);
    expect(await refusal({ profile: "document", systemPrompt: "  " })).toEqual(invalidPrompt);
    expect(await refusal({ profile: "coding", systemPrompt: "x" })).toEqual(invalidPrompt);
    expect(await refusal({ profile: "coding", tools: ["read"] })).toEqual(invalidTools);
    for (const tools of [[], ["+bash"], ["read,bash"], "read"]) expect(await refusal({ ...DOC, tools })).toEqual(invalidTools);
  });

  test("resume by sessionId reuses the live child for the same spec, restarts an idle one for a new prompt or tools", async () => {
    const { sessionId, sessionPath } = document;
    const again = await startSession({ cwd, ...DOC, resume: { sessionId } });
    expect(again.status).toBe(200);
    expect((await again.json()).session).toMatchObject({ sessionId, profile: "document", live: { running: false } });
    const changed = await startSession({ cwd, ...DOC, systemPrompt: "Answer only with the word NEW", resume: { sessionId } });
    expect(changed.status).toBe(200);
    expect((await changed.json()).session).toMatchObject({ sessionId, sessionPath, profile: "document" });
    const driver = await client();
    expect((await driver.attach(sessionId)).attached.state.argv).toContain("Answer only with the word NEW");
    expect(await refusal({ profile: "coding", resume: { sessionId } })).toEqual({ status: 409, body: { error: "profile_mismatch" } });
    expect(await refusal({ ...DOC, resume: { sessionId: "no-such-session" } })).toEqual({ status: 404, body: { error: "not_found" } });
    expect((await refusal({ ...DOC, resume: { sessionId: "" } })).status).toBe(400);
  });

  test("a closed document session resumes by id with its prompt sent again, never as coding", async () => {
    const { sessionId, sessionPath } = document;
    await api(`/sessions/${sessionId}/close`, { method: "POST" });
    expect(await refusal({ resume: { sessionId } })).toEqual({ status: 409, body: { error: "profile_mismatch" } });
    expect(await refusal({ resume: { sessionPath } })).toEqual({ status: 409, body: { error: "profile_mismatch" } });
    const resumed = await startSession({ cwd, ...DOC, resume: { sessionId } });
    expect(resumed.status).toBe(200);
    expect((await resumed.json()).session).toMatchObject({ sessionId, sessionPath, profile: "document" });
    const driver = await client();
    expect((await driver.attach(sessionId)).attached.state.argv.slice(0, DOCUMENT_ARGV.length + 2)).toEqual([
      ...DOCUMENT_ARGV,
      "--session",
      sessionPath,
    ]);
  });

  test("marks the cache retention a session's Pi started with, none for Pi's default", async () => {
    const plain = (await (await startSession({ cwd })).json()).session;
    await api(`/sessions/${plain.sessionId}/close`, { method: "POST" });
    expect(existsSync(`${plain.sessionPath}.gigachad.json`)).toBe(false);

    process.env.PI_CACHE_RETENTION = "long";
    try {
      const long = (await (await startSession({ cwd })).json()).session;
      expect(JSON.parse(readFileSync(`${long.sessionPath}.gigachad.json`, "utf8"))).toMatchObject({ cacheRetention: "long" });
      await api(`/sessions/${long.sessionId}/close`, { method: "POST" });
    } finally {
      delete process.env.PI_CACHE_RETENTION;
    }
  });

  test("a coding session keeps its profile and writes no marker", async () => {
    const coding = (await (await startSession({ cwd })).json()).session;
    expect(coding.profile).toBe("coding");
    await settle(coding.sessionId, "code");
    await api(`/sessions/${coding.sessionId}/close`, { method: "POST" });
    expect(existsSync(`${coding.sessionPath}.gigachad.json`)).toBe(false);
  });
});

describe("factory task sessions", () => {
  const repo = workdir("factory-repo");
  const factoryDir = join(agentDir, "factory", "sessions", cwdDirName(repo));
  const FACTORY_ID = "01a1f3fe-0000-7000-8000-0000000000fa";

  test("session and task record writes under the factory's data reach clients as sessions_changed, other files do not", async () => {
    const watching = await client();
    const reported = (path: string) => () => (watching.hostFrames.some((frame) => frame.sessionPaths.includes(path)) ? true : undefined);
    mkdirSync(factoryDir, { recursive: true });
    const sessionPath = join(factoryDir, `2026-11-19T10-00-00-000Z_${FACTORY_ID}.jsonl`);
    writeFileSync(sessionPath, `${JSON.stringify({ type: "session", version: 3, id: FACTORY_ID, cwd: repo })}\n`);
    await until(reported(sessionPath));

    // A task record is written to a temp file and renamed, and removed by a CLI.
    const tasks = join(agentDir, "factory", "tasks");
    mkdirSync(tasks, { recursive: true });
    const record = join(tasks, "t1.json");
    writeFileSync(`${record}.1.tmp`, "{}");
    renameSync(`${record}.1.tmp`, record);
    await until(reported(record));
    const before = watching.hostFrames.length;
    rmSync(record);
    await until(() => (watching.hostFrames.slice(before).some((frame) => frame.sessionPaths.includes(record)) ? true : undefined));

    writeFileSync(join(tasks, "t1.inbox"), "{}\n");
    writeFileSync(join(agentDir, "factory", "notes.json"), "{}");
    await Bun.sleep(800);
    const paths = watching.hostFrames.flatMap((frame) => frame.sessionPaths);
    expect(paths.every((path) => path.endsWith(".jsonl") || path === record)).toBe(true);
  });

  test("a factory task is never resumed: the factory plugin's runner holds it", async () => {
    mkdirSync(factoryDir, { recursive: true });
    const sessionPath = join(factoryDir, `2026-11-19T10-00-00-000Z_${FACTORY_ID}.jsonl`);
    const state = { type: "custom", id: "s1", parentId: null, timestamp: "2026-11-19T10:00:00.000Z", customType: "factory-state", data: { state: "running", round: 1, reason: "" } };
    const header = { type: "session", version: 3, id: FACTORY_ID, timestamp: "2026-11-19T10:00:00.000Z", cwd: repo };
    writeFileSync(sessionPath, `${JSON.stringify(header)}\n${JSON.stringify(state)}\n`);
    for (const resume of [{ sessionId: FACTORY_ID }, { sessionPath }]) {
      const response = await startSession({ cwd: repo, resume });
      expect(response.status).toBe(409);
      expect(await response.json()).toEqual({ error: "factory" });
    }
  });
});

test("the gateway token opens /v1/* only, the full token every route (C1)", async () => {
  const base = `http://127.0.0.1:${server.port}`;
  const gateway = { authorization: `Bearer ${TOKENS.gateway}` };
  expect((await fetch(`${base}/v1/models`, { headers: gateway })).status).toBe(200);
  expect((await api("/v1/models")).status).toBe(200);
  const body = JSON.stringify({ cwd: workdir("gateway-token"), profile: "coding", resume: null });
  expect((await fetch(`${base}/sessions`, { method: "POST", headers: gateway, body })).status).toBe(401);
  expect((await fetch(`${base}/sessions?cwd=/`, { headers: gateway })).status).toBe(401);
  expect((await fetch(`${base}/health`, { headers: gateway })).status).toBe(401);
  expect((await fetch(`${base}/usage`, { headers: gateway })).status).toBe(401);
  expect((await fetch(`${base}/sessions/any/close`, { method: "POST", headers: gateway })).status).toBe(401);
  expect((await fetch(`${base}/sessions/any`, { method: "DELETE", headers: gateway })).status).toBe(401);
  expect((await fetch(`${base}/ws`, { headers: { ...gateway, upgrade: "websocket" } })).status).toBe(401);
});

describe("spawn strategies (C5, C6)", () => {
  const extensionsDir = join(root, "pi-extensions");
  const recordDir = join(root, "sandbox-record");
  const gatewayTokenPath = join(root, "state", "pi-gateway.token");
  const fakeSandbox = wrapper(
    "agent-sandbox",
    [
      `mkdir -p "${recordDir}"`,
      `printf '%s\\n' "$@" > "${recordDir}/argv"`,
      `env > "${recordDir}/env"`,
      `while [ "$1" != "--" ]; do shift; done`,
      "shift",
      `exec "${fakePi}" "$@"`,
    ].join("\n"),
  );
  type Started = { session: { sessionId: string; sessionPath: string; cwd: string; profile: string }; argv: string[] };
  let strategies: Supervisor;
  let strategiesServer: Server<SocketData>;

  async function argvOf(body: object): Promise<Started> {
    const response = await startSession(body, strategiesServer.port);
    expect(response.status).toBe(200);
    const { session } = await response.json();
    const driver = await client(strategiesServer);
    return { session, argv: (await driver.attach(session.sessionId)).attached.state.argv };
  }

  beforeAll(async () => {
    for (const dir of ["tools", "empty", "node_modules"]) mkdirSync(join(extensionsDir, dir), { recursive: true });
    for (const file of ["files.ts", "alpha.ts", "alpha.test.ts", "types.d.ts", "README.md", "tools/index.ts"]) {
      writeFileSync(join(extensionsDir, file), "");
    }
    strategies = await createSupervisor({
      piBin: fakePi,
      idleMs: 3_600_000,
      isPinned: async () => false,
      extensionsDir,
      agentSandboxBin: fakeSandbox,
      sandboxProfile: "test-profile",
      gatewayPort: 8399,
      gatewayTokenPath,
    });
    strategiesServer = startServer({ hostname: "127.0.0.1", port: 0, tokens: TOKENS, supervisor: strategies });
  });

  afterAll(async () => {
    await strategies.shutdown();
    strategiesServer.stop(true);
  });

  test("coding and document children load the gateway provider, then each top-level extension sorted, restricted document children the gateway provider only", async () => {
    const gateway = ["-e", GATEWAY_EXTENSION];
    const extensions = [...gateway, "-e", join(extensionsDir, "alpha.ts"), "-e", join(extensionsDir, "files.ts"), "-e", join(extensionsDir, "tools")];
    expect((await argvOf({ cwd: workdir("extensions") })).argv).toEqual(["--mode", "rpc", ...extensions]);
    const document = { cwd: workdir("extensions"), profile: "document", systemPrompt: "Doc" };
    expect((await argvOf(document)).argv).toEqual(["--mode", "rpc", "--append-system-prompt", "Doc", ...extensions]);
    expect((await argvOf({ ...document, tools: ["read"] })).argv.slice(-2)).toEqual(gateway);
  });

  test("gatewayEnv points a local child at this pi-host's /v1 with the gateway token", () => {
    expect(gatewayEnv(8399, "gw")).toEqual({ PI_GATEWAY_URL: "http://127.0.0.1:8399/v1", PI_GATEWAY_TOKEN: "gw" });
  });

  test("a missing extensions dir adds only the gateway provider", async () => {
    const bare = await createSupervisor({ piBin: fakePi, idleMs: 3_600_000, extensionsDir: join(root, "no-extensions") });
    const bareServer = startServer({ hostname: "127.0.0.1", port: 0, tokens: TOKENS, supervisor: bare });
    try {
      const response = await startSession({ cwd: workdir("no-extensions") }, bareServer.port);
      const { session } = await response.json();
      const driver = await client(bareServer);
      expect((await driver.attach(session.sessionId)).attached.state.argv).toEqual(["--mode", "rpc", "-e", GATEWAY_EXTENSION]);
    } finally {
      await bare.shutdown();
      bareServer.stop(true);
    }
  });

  test("a sandboxed child runs through agent-sandbox pi-rpc with the gateway token only and a gateway model", async () => {
    process.env.ANTHROPIC_API_KEY = "secret";
    const cwd = workdir("sandboxed");
    const sessionDir = join(agentDir, "sessions", cwdDirName(cwd));
    let started: Started;
    try {
      started = await argvOf({ cwd, profile: "sandboxed", model: { provider: "anthropic", id: "claude-x" } });
    } finally {
      delete process.env.ANTHROPIC_API_KEY;
    }
    const piArgs = ["--mode", "rpc", "--session-dir", sessionDir];
    expect(started.argv).toEqual(piArgs);
    expect(readFileSync(join(recordDir, "argv"), "utf8").trimEnd().split("\n")).toEqual([
      "pi-rpc",
      "--profile",
      "test-profile",
      "--cwd",
      cwd,
      "--session-dir",
      sessionDir,
      "--gateway-port",
      "8399",
      "--",
      ...piArgs,
    ]);
    const env = Object.fromEntries(
      readFileSync(join(recordDir, "env"), "utf8")
        .trimEnd()
        .split("\n")
        .map((line) => [line.slice(0, line.indexOf("=")), line.slice(line.indexOf("=") + 1)]),
    );
    expect(env.PI_GATEWAY_TOKEN).toBe(readFileSync(gatewayTokenPath, "utf8").trim());
    const shellOwn = ["PWD", "OLDPWD", "SHLVL", "_"];
    const allowed = ["HOME", "USER", "LOGNAME", "PATH", "LANG", "TZ", "TMPDIR", "XDG_RUNTIME_DIR", "PI_GATEWAY_TOKEN"];
    expect(Object.keys(env).filter((name) => !shellOwn.includes(name) && !allowed.includes(name))).toEqual([]);

    const { session } = started;
    expect(session).toMatchObject({ cwd, profile: "sandboxed" });
    expect(session.sessionPath.startsWith(`${sessionDir}/`)).toBe(true);
    const driver = await client(strategiesServer);
    driver.send({ type: "attach", sessionId: session.sessionId });
    const attached = await driver.next((frame) => frame.type === "attached");
    expect(attached.state.model).toEqual({ provider: "gateway", id: "anthropic/claude-x" });
  });

  test("a resumed sandboxed session mounts its file's directory and refuses another profile", async () => {
    const cwd = workdir("sandboxed");
    const { sessionPath } = writeSessionFile(cwd, [messageEntry("u1", null, "user", "hi")]);
    const marker = { revision: 1, pinned: false, data: {}, profile: "sandboxed" };
    writeFileSync(`${sessionPath}.gigachad.json`, JSON.stringify(marker));
    const resumed = await startSession({ cwd, profile: "sandboxed", resume: { sessionPath } }, strategiesServer.port);
    expect(resumed.status).toBe(200);
    const recorded = readFileSync(join(recordDir, "argv"), "utf8").trimEnd().split("\n");
    expect(recorded.slice(5, 7)).toEqual(["--session-dir", join(agentDir, "sessions", cwdDirName(cwd))]);
    expect(recorded.slice(10)).toEqual(["--mode", "rpc", "--session", sessionPath, "--session-dir", join(agentDir, "sessions", cwdDirName(cwd))]);
    const { session } = await resumed.json();
    await api(`/sessions/${session.sessionId}/close`, { method: "POST" }, strategiesServer.port);
    const coding = await startSession({ cwd, resume: { sessionPath } }, strategiesServer.port);
    expect(coding.status).toBe(409);
  });

  test("sandboxEnv keeps only what agent-sandbox needs plus the gateway token", () => {
    const env = sandboxEnv("gw", {
      PATH: "/bin",
      HOME: "/home/someone",
      XDG_RUNTIME_DIR: "/run/user/1000",
      SSH_AUTH_SOCK: "/tmp/agent",
      PI_CODING_AGENT_DIR: "/scratch",
      ANTHROPIC_API_KEY: "secret",
    });
    expect(env).toEqual({ PATH: "/bin", HOME: "/home/someone", XDG_RUNTIME_DIR: "/run/user/1000", PI_GATEWAY_TOKEN: "gw" });
  });
});

describe("DELETE /sessions/:id", () => {
  function remove(sessionId: string): Promise<Response> {
    return api(`/sessions/${sessionId}`, { method: "DELETE" });
  }

  async function liveIds(): Promise<string[]> {
    const health = await (await api("/health")).json();
    return health.sessions.map((session: { sessionId: string }) => session.sessionId);
  }

  test("removes an idle session's file and sidecar, 404 once it is gone", async () => {
    const cwd = workdir("delete-idle");
    const { sessionId, sessionPath } = writeSessionFile(cwd, [messageEntry("u1", null, "user", "hi")]);
    const sidecar = `${sessionPath}.gigachad.json`;
    expect((await api(`/sessions/${sessionId}/pin`, { method: "PUT", body: JSON.stringify({ pinned: true }) })).status).toBe(200);
    expect(existsSync(sidecar)).toBe(true);

    const removed = await remove(sessionId);
    expect(removed.status).toBe(204);
    expect(await removed.text()).toBe("");
    expect(existsSync(sessionPath)).toBe(false);
    expect(existsSync(sidecar)).toBe(false);
    expect((await (await api(`/sessions?cwd=${encodeURIComponent(cwd)}`)).json()).sessions).toEqual([]);

    const again = await remove(sessionId);
    expect(again.status).toBe(404);
    expect(await again.json()).toEqual({ error: "not_found" });
  });

  test("stops a running, attached child first", async () => {
    const cwd = workdir("delete-live");
    const { sessionId, sessionPath } = writeSessionFile(cwd, [
      messageEntry("u1", null, "user", "hi"),
      messageEntry("a1", "u1", "assistant", "hello"),
    ]);
    expect((await startSession({ cwd, resume: { sessionPath } })).status).toBe(200);
    const viewer = await client();
    await viewer.attach(sessionId);
    viewer.command(sessionId, { type: "prompt", message: "hold" });
    await viewer.pi((record) => record.type === "message_update");
    expect(supervisor.liveStatus(sessionPath)).toMatchObject({ running: true, attached: 1 });

    expect((await remove(sessionId)).status).toBe(204);
    expect((await viewer.next((frame) => frame.type === "exit")).sessionId).toBe(sessionId);
    expect(supervisor.liveStatus(sessionPath)).toBeUndefined();
    expect(await liveIds()).not.toContain(sessionId);
    expect(existsSync(sessionPath)).toBe(false);
    expect(existsSync(`${sessionPath}.gigachad.json`)).toBe(false);
  });

  test("stops the child of a new session that has no file yet", async () => {
    const { session } = await (await startSession({ cwd: workdir("delete-new") })).json();
    expect(existsSync(session.sessionPath)).toBe(false);
    expect(await liveIds()).toContain(session.sessionId);
    expect((await remove(session.sessionId)).status).toBe(204);
    expect(await liveIds()).not.toContain(session.sessionId);
    expect((await remove(session.sessionId)).status).toBe(404);
  });

  test("an unknown id is 404, a file another holder has locked 409 live like archive", async () => {
    expect((await remove("no-such-session")).status).toBe(404);
    expect((await remove("..%2F..%2Fetc")).status).toBe(404);
    expect((await remove("%E0%A4%A")).status).toBe(400);
    expect((await api("/sessions/%E0%A4%A/close", { method: "POST" })).status).toBe(400);

    const { sessionId, sessionPath } = writeSessionFile(workdir("delete-held"), [messageEntry("u1", null, "user", "hi")]);
    const held = tryLockFile(sessionPath)!;
    const refused = await remove(sessionId);
    expect(refused.status).toBe(409);
    expect(await refused.json()).toEqual({ error: "live" });
    expect(existsSync(sessionPath)).toBe(true);
    held.release();
    expect((await remove(sessionId)).status).toBe(204);
    expect(existsSync(sessionPath)).toBe(false);
  });
});
