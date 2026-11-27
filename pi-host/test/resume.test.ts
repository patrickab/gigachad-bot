/** Auto-resume of interrupted runs: the active-run file, claiming a dead owner's list, the opt-in resume (fake `pi`, no model). */
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Server, ServerWebSocket } from "bun";
import { startServer } from "../src/server.ts";
import { isProcessAlive, processStartIdentity } from "../src/supervisor/process-identity.ts";
import { createSupervisor, type SocketData, type Supervisor } from "../src/supervisor/index.ts";
import { RESUME_PROMPT, ResumeState, type RunRecord } from "../src/supervisor/resume.ts";

const TOKEN = "test-token";
const root = mkdtempSync(join(tmpdir(), "pi-host-resume-"));
const agentDir = join(root, "agent");
const savedAgentDir = process.env.PI_CODING_AGENT_DIR;
const wrapperPath = join(root, "pi");
writeFileSync(wrapperPath, `#!/bin/sh\nexec "${process.execPath}" "${join(import.meta.dir, "fixtures", "fake-pi.ts")}" "$@"\n`);
chmodSync(wrapperPath, 0o755);

const record: RunRecord = { sessionId: "s1", sessionPath: "/x/s1.jsonl", cwd: "/x", spec: { profile: "coding" } };

async function until<T>(probe: () => T | undefined, timeoutMs = 5000): Promise<T> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const value = probe();
    if (value !== undefined) return value;
    await Bun.sleep(10);
  }
  throw new Error("timed out");
}

function freshDir(name: string): string {
  const dir = join(root, name);
  mkdirSync(dir, { recursive: true });
  return dir;
}

function listFiles(dir: string): string[] {
  return readdirSync(dir).filter((name) => name.startsWith("active-runs-") && name.endsWith(".json"));
}

describe("process identity", () => {
  test("names this process, and no process for a pid that cannot exist", async () => {
    expect(isProcessAlive(process.pid)).toBe(true);
    expect(isProcessAlive(0)).toBe(false);
    const own = await processStartIdentity(process.pid);
    expect(own).toMatch(/^linux:.+:\d+$/);
    expect(await processStartIdentity(process.pid)).toBe(own);
    expect(await processStartIdentity(2 ** 30)).toBeNull();
  });
});

describe("ResumeState", () => {
  test("writes its list atomically on track and deletes it on untrack", async () => {
    const dir = freshDir("state-basic");
    const state = new ResumeState(dir, 4242, async () => "linux:boot:7");
    await state.track(record);
    expect(listFiles(dir)).toEqual(["active-runs-4242.json"]);
    expect(JSON.parse(readFileSync(join(dir, "active-runs-4242.json"), "utf8"))).toEqual({
      ownerPid: 4242,
      ownerStart: "linux:boot:7",
      sessions: [record],
    });
    expect(readdirSync(dir).filter((name) => name.endsWith(".tmp"))).toEqual([]);
    await state.untrack("s1");
    expect(readdirSync(dir)).toEqual([]);
  });

  test("claims the list of a dead owner once, and leaves the list of a live one", async () => {
    const dir = freshDir("state-claim");
    const dead = new ResumeState(dir, 1111, async () => "start-a");
    await dead.track({ ...record, sessionId: "dead" });
    // Tracking claims first: both owners look alive to it, so the dead owner's list stays for the claimant below
    const live = new ResumeState(dir, 2222, async (pid) => (pid === 1111 ? "start-a" : "start-b"), () => true);
    await live.track({ ...record, sessionId: "live" });

    const alive = (pid: number) => pid === 2222;
    const claimant = new ResumeState(dir, 3333, async (pid) => (pid === 2222 ? "start-b" : null), alive);
    expect((await claimant.claimOrphans()).map((entry) => entry.sessionId)).toEqual(["dead"]);
    expect(listFiles(dir)).toEqual(["active-runs-2222.json"]);
    expect(await claimant.claimOrphans()).toHaveLength(1);
    const second = new ResumeState(dir, 4444, async (pid) => (pid === 2222 ? "start-b" : null), alive);
    expect(await second.claimOrphans()).toEqual([]);
  });

  test("a reused pid does not protect a list: the start time must match", async () => {
    const dir = freshDir("state-reuse");
    await new ResumeState(dir, 1111, async () => "before-reboot").track(record);
    const claimant = new ResumeState(dir, 3333, async () => "after-reboot", () => true);
    expect(await claimant.claimOrphans()).toEqual([record]);
  });

  test("a list from before a reboot under this very pid is claimed, not mistaken for our own", async () => {
    const dir = freshDir("state-same-pid");
    await new ResumeState(dir, 1111, async () => "before-reboot").track(record);
    const restarted = new ResumeState(dir, 1111, async () => "after-reboot", () => true);
    expect(await restarted.claimOrphans()).toEqual([record]);
  });

  test("the setting is off by default and persists", async () => {
    const dir = freshDir("state-setting");
    const state = new ResumeState(dir);
    expect(await state.autoResume()).toBe(false);
    await state.setAutoResume(true);
    expect(await new ResumeState(dir).autoResume()).toBe(true);
    await state.setAutoResume(false);
    expect(await state.autoResume()).toBe(false);
  });
});

describe("resume after an interruption", () => {
  let server: Server<SocketData>;
  const cwd = join(root, "work");
  const supervisors: Supervisor[] = [];

  beforeAll(() => {
    process.env.PI_CODING_AGENT_DIR = agentDir;
    mkdirSync(join(agentDir, "sessions"), { recursive: true });
    mkdirSync(join(agentDir, "factory", "sessions"), { recursive: true });
    mkdirSync(cwd, { recursive: true });
  });

  afterAll(async () => {
    server?.stop(true);
    for (const supervisor of supervisors) await supervisor.shutdown();
    if (savedAgentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
    else process.env.PI_CODING_AGENT_DIR = savedAgentDir;
    rmSync(root, { recursive: true, force: true });
  });

  async function supervise(dir: string, pid: number, dead = false): Promise<{ supervisor: Supervisor; state: ResumeState }> {
    const state = new ResumeState(dir, pid, async () => `start-${pid}`, () => !dead);
    const supervisor = await createSupervisor({ piBin: wrapperPath, idleMs: 3_600_000, isPinned: async () => false, resume: state });
    supervisors.push(supervisor);
    return { supervisor, state };
  }

  function post(supervisor: Supervisor, path: string, body: object): Promise<Response> {
    return Promise.resolve(supervisor.handle(new Request(`http://pi-host${path}`, { method: "POST", body: JSON.stringify(body) }), new URL(`http://pi-host${path}`))).then(
      (response) => response!,
    );
  }

  /** A session with one finished turn (so its file exists) and a run in flight. Returns its path and id. */
  async function interruptedRun(supervisor: Supervisor, dir: string): Promise<{ sessionId: string; sessionPath: string }> {
    const started = await post(supervisor, "/sessions", { cwd, profile: "coding", resume: null });
    const { session } = await started.json();
    const frames: string[] = [];
    const sink = { send: (frame: string) => frames.push(frame), data: { clientId: "resume" } } as unknown as ServerWebSocket<SocketData>;
    const send = (frame: object) => supervisor.websocket.message!(sink, JSON.stringify(frame));
    const prompt = (message: string) => send({ type: "command", sessionId: session.sessionId, command: { type: "prompt", message } });
    send({ type: "attach", sessionId: session.sessionId });
    prompt("hello");
    await until(() => (frames.some((frame) => frame.includes('"agent_settled"')) ? true : undefined));
    await until(() => (existsSync(session.sessionPath) ? true : undefined));
    await until(() => (supervisor.liveStatus(session.sessionPath)?.running === false ? true : undefined));
    prompt("hold");
    await until(() => (listFiles(dir).length === 1 ? true : undefined));
    return session;
  }

  /** The owner ends mid-run: shutdown keeps its record, then its children are gone. */
  async function crash(supervisor: Supervisor, sessionPath: string): Promise<void> {
    await supervisor.shutdown();
    await until(() => (supervisor.liveStatus(sessionPath) === undefined ? true : undefined));
  }

  test("records a run on start and drops it when it settles", async () => {
    const dir = freshDir("run-record");
    const { supervisor } = await supervise(dir, process.pid);
    const session = await interruptedRun(supervisor, dir);
    const list = JSON.parse(readFileSync(join(dir, `active-runs-${process.pid}.json`), "utf8"));
    expect(list.sessions).toEqual([
      { sessionId: session.sessionId, sessionPath: session.sessionPath, cwd, spec: { profile: "coding" } },
    ]);
  });

  test("resumes an interrupted session with the resume prompt when the setting is on", async () => {
    const dir = freshDir("run-resume");
    const first = await supervise(dir, 5001);
    const session = await interruptedRun(first.supervisor, dir);
    await crash(first.supervisor, session.sessionPath);
    expect(listFiles(dir)).toEqual(["active-runs-5001.json"]);

    const claimant = new ResumeState(dir, 5002, async (pid) => `start-${pid}`, () => false);
    const supervisor = await createSupervisor({ piBin: wrapperPath, idleMs: 3_600_000, isPinned: async () => false, resume: claimant });
    supervisors.push(supervisor);
    await claimant.setAutoResume(true);
    const outcomes = await supervisor.resumeInterrupted();
    expect(outcomes).toEqual([{ sessionId: session.sessionId, outcome: "resumed" }]);
    await until(() => (readFileSync(session.sessionPath, "utf8").includes(RESUME_PROMPT) ? true : undefined));
  });

  test("claims and discards the list while the setting is off", async () => {
    const dir = freshDir("run-off");
    const first = await supervise(dir, 6001);
    const session = await interruptedRun(first.supervisor, dir);
    await crash(first.supervisor, session.sessionPath);

    const { supervisor } = await supervise(dir, 6002, true);
    expect(await supervisor.resumeInterrupted()).toEqual([]);
    expect(listFiles(dir)).toEqual([]);
    expect(readFileSync(session.sessionPath, "utf8")).not.toContain(RESUME_PROMPT);
  });

  test("never resumes a session another holder has locked", async () => {
    const dir = freshDir("run-locked");
    const first = await supervise(dir, 7001);
    const session = await interruptedRun(first.supervisor, dir);
    await crash(first.supervisor, session.sessionPath);

    const holder = Bun.spawn(["flock", "--nonblock", "--no-fork", session.sessionPath, "sh", "-c", "echo locked; exec sleep 30"], { stdout: "pipe" });
    try {
      // The holder prints only once it holds the lock.
      const reader = holder.stdout.getReader();
      const deadline = Date.now() + 10_000;
      let seen = "";
      while (!seen.includes("locked")) {
        const left = deadline - Date.now();
        if (left <= 0) throw new Error("lock holder did not report the lock");
        const chunk = await Promise.race([reader.read(), Bun.sleep(left).then(() => undefined)]);
        if (!chunk) continue;
        if (chunk.done) throw new Error("lock holder exited early");
        seen += new TextDecoder().decode(chunk.value);
      }
      const { supervisor, state } = await supervise(dir, 7002, true);
      await state.setAutoResume(true);
      const outcomes = await supervisor.resumeInterrupted();
      expect(outcomes).toEqual([{ sessionId: session.sessionId, outcome: "locked" }]);
      expect(readFileSync(session.sessionPath, "utf8")).not.toContain(RESUME_PROMPT);
    } finally {
      holder.kill();
      await holder.exited;
    }
  });

  test("never resumes a run the user aborted", async () => {
    const dir = freshDir("run-abort");
    const first = await supervise(dir, 8001);
    server = startServer({ hostname: "127.0.0.1", port: 0, tokens: { full: TOKEN, gateway: "g" }, supervisor: first.supervisor });
    const session = await interruptedRun(first.supervisor, dir);

    const socket = new WebSocket(`ws://127.0.0.1:${server.port}/ws`, { headers: { authorization: `Bearer ${TOKEN}` } } as unknown as string[]);
    const frames: string[] = [];
    socket.addEventListener("message", (event) => frames.push(String(event.data)));
    await until(() => (socket.readyState === WebSocket.OPEN ? true : undefined));
    socket.send(JSON.stringify({ type: "attach", sessionId: session.sessionId }));
    await until(() => (frames.some((frame) => frame.includes('"type":"attached"')) ? true : undefined));
    socket.send(JSON.stringify({ type: "command", sessionId: session.sessionId, command: { type: "abort" } }));
    await until(() => (listFiles(dir).length === 0 ? true : undefined));
    socket.close();
    await crash(first.supervisor, session.sessionPath);

    const { supervisor, state } = await supervise(dir, 8002, true);
    await state.setAutoResume(true);
    expect(await supervisor.resumeInterrupted()).toEqual([]);
  });

  test("serves the setting over /settings", async () => {
    const dir = freshDir("run-settings");
    const { supervisor } = await supervise(dir, 9001);
    const call = async (method: string, body?: object) => {
      const response = await supervisor.handle(new Request("http://pi-host/settings", { method, body: body && JSON.stringify(body) }), new URL("http://pi-host/settings"));
      return { status: response!.status, body: await response!.json() };
    };
    expect(await call("GET")).toEqual({ status: 200, body: { autoResume: false } });
    expect(await call("PUT", { autoResume: true })).toEqual({ status: 200, body: { autoResume: true } });
    expect((await call("PUT", { autoResume: "yes" })).status).toBe(400);
    expect(await call("GET")).toEqual({ status: 200, body: { autoResume: true } });
  });
});
