/** Supervisor tests over HTTP and `/ws` against the fake `pi` fixture (no model calls). */
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { chmodSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Server } from "bun";
import { startServer } from "../src/server.ts";
import { childEnv } from "../src/supervisor/config.ts";
import { tryLockFile } from "../src/supervisor/lock.ts";
import { createSupervisor, type SocketData, type Supervisor } from "../src/supervisor/index.ts";
import type { SupervisorHost } from "../src/supervisor/host.ts";

type Frame = { type: string; sessionId: string; [field: string]: any };

const TOKEN = "test-token";
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

async function until<T>(probe: () => T | undefined, timeoutMs = 5000): Promise<T> {
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
  /** Frames whose `sequence` is not the previous one of their `streamId` plus 1 (M16). */
  readonly sequenceGaps: Frame[] = [];
  private readonly lastSequence = new Map<string, number>();
  private cursor = 0;

  private constructor(readonly socket: WebSocket) {
    socket.addEventListener("message", (event) => {
      const frame: Frame = JSON.parse(String(event.data));
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
  supervisor = await createSupervisor({ piBin: fakePi, idleMs: 3_600_000, isPinned: async () => false });
  server = startServer({ hostname: "127.0.0.1", port: 0, token: TOKEN, supervisor });
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
    expect(await (await api("/health")).json()).toEqual({ ok: true, piVersion: "1.5.2" });
  });
});

test("refuses a Pi binary with another version", async () => {
  const otherPi = wrapper("pi-old", "echo 1.4.0");
  await expect(createSupervisor({ piBin: otherPi })).rejects.toThrow("pi-host needs 1.5.2");
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
    expect((await startSession({ cwd, profile: "sandboxed" })).status).toBe(400);
    const refused = await startSession({ cwd, model: { provider: "bad", id: "x" } });
    expect(refused.status).toBe(400);
    expect(await refused.json()).toMatchObject({ error: "model" });
  });

  test("two clients attach, both receive events, either may command", async () => {
    first = await client();
    second = await client();
    for (const attaching of [first, second]) {
      const { attached, replay } = await attaching.attach(sessionId);
      expect(attached.state).toMatchObject({ sessionId, isStreaming: false, thinkingLevel: "high", model: { id: "faux-2" } });
      expect(attached.entries).toEqual({ entries: [], leafId: null });
      expect(attached.ui).toEqual({ dialogs: [], statuses: {}, queue: { steering: [], followUp: [] } });
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
    expect((await startSession({ cwd, resume: "continue" })).status).toBe(409);
    holder.kill();
    await holder.exited;

    const resumed = await (await startSession({ cwd, resume: "continue" })).json();
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

test("idle eviction skips pinned, streaming and liveness-reporting sessions", async () => {
  const pinned = new Set<string>();
  const idleMs = 60_000;
  const evicting = await createSupervisor({ piBin: fakePi, idleMs, isPinned: async (path) => pinned.has(path) });
  const evictingServer = startServer({ hostname: "127.0.0.1", port: 0, token: TOKEN, supervisor: evicting });
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
