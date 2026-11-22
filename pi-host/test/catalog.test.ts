/** Catalog routes against fixture Pi session files in a scratch agent dir. */
import { afterAll, beforeAll, beforeEach, describe, expect, test } from "bun:test";
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, readFile, rm, utimes, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { LiveStatus, SessionInfo } from "../src/catalog/index.ts";
import type { RankedFile } from "../src/catalog/files.ts";
import { tryLockFile } from "../src/host/local/lock.ts";
import { cwdDirName } from "../src/host/local/paths.ts";
import type { LiveSessions } from "../src/supervisor/index.ts";
import { copyPrivateState, handleCatalog, isPinned } from "./fixtures/local-catalog.ts";

/** Union of the catalog response bodies; each assertion reads the fields of its route. */
type Body = {
  sessions: SessionInfo[];
  session: SessionInfo;
  files: RankedFile[];
  revision: number;
  data: Record<string, unknown>;
  error: string;
};

/** Private state route body: `{revision, data}`, or `{revision}` after a write. */
type StateBody = { revision: number; data?: Record<string, unknown> };

let scratch: string;
let projectDir: string;
let sessionDir: string;
const liveByPath = new Map<string, LiveStatus>();
/** Session paths whose idle child the catalog closed. */
const closed: string[] = [];
const live: LiveSessions = {
  liveStatus: (path) => liveByPath.get(path),
  closeIfIdle: async (path) => {
    const status = liveByPath.get(path);
    if (!status) return true;
    if (status.running || status.waiting || status.attached > 0) return false;
    liveByPath.delete(path);
    closed.push(path);
    return true;
  },
};
const savedEnv = { agent: process.env.PI_CODING_AGENT_DIR, sessions: process.env.PI_CODING_AGENT_SESSION_DIR };

const OLD_ID = "01a1f3fe-0000-7000-8000-000000000001";
const NEW_ID = "01a1f3fe-0000-7000-8000-000000000002";
const OTHER_CWD_ID = "01a1f3fe-0000-7000-8000-000000000003";

/** JSONL session file content: header plus entries. */
function sessionLines(id: string, cwd: string, timestamp: string, entries: object[], parentSession?: string): string {
  const header = { type: "session", version: 3, id, timestamp, cwd, ...(parentSession ? { parentSession } : {}) };
  return [header, ...entries].map((entry) => JSON.stringify(entry)).join("\n") + "\n";
}

/** A `message` entry, `usage` on an assistant answer. */
function message(id: string, role: string, content: unknown, usage?: { totalTokens: number }): object {
  return { type: "message", id, parentId: null, timestamp: "2026-11-19T10:00:00.000Z", message: { role, content, timestamp: 0, ...(usage ? { usage } : {}) } };
}

/** Write a fixture session file and set its mtime. */
async function writeSession(id: string, content: string, mtime: Date): Promise<string> {
  const path = join(sessionDir, `2026-11-19T10-00-00-000Z_${id}.jsonl`);
  await writeFile(path, content);
  await utimes(path, mtime, mtime);
  return path;
}

/** Call the catalog router and return status and JSON body. */
async function call<T = Body>(method: string, path: string, body?: unknown): Promise<{ status: number; json: T }> {
  const url = new URL(`http://pi-host${path}`);
  const request = new Request(url, {
    method,
    ...(body === undefined ? {} : { body: JSON.stringify(body), headers: { "content-type": "application/json" } }),
  });
  const response = await handleCatalog(request, url, live);
  if (!response) throw new Error(`catalog did not handle ${method} ${path}`);
  return { status: response.status, json: (await response.json()) as T };
}

/** Run a command in `cwd`, failing the test on a non-zero exit. */
async function run(cwd: string, ...command: string[]): Promise<void> {
  const child = Bun.spawn(command, { cwd, stdout: "ignore", stderr: "pipe" });
  if ((await child.exited) !== 0) throw new Error(await new Response(child.stderr).text());
}

beforeAll(async () => {
  scratch = await mkdtemp(join(tmpdir(), "pi-host-catalog-"));
  process.env.PI_CODING_AGENT_DIR = join(scratch, "agent");
  delete process.env.PI_CODING_AGENT_SESSION_DIR;
  projectDir = join(scratch, "project");
  await mkdir(projectDir);
  sessionDir = join(scratch, "agent", "sessions", cwdDirName(projectDir));
});

beforeEach(async () => {
  liveByPath.clear();
  closed.length = 0;
  await rm(join(scratch, "agent"), { recursive: true, force: true });
  await mkdir(sessionDir, { recursive: true });
});

afterAll(async () => {
  await rm(scratch, { recursive: true, force: true });
  if (savedEnv.agent === undefined) delete process.env.PI_CODING_AGENT_DIR;
  else process.env.PI_CODING_AGENT_DIR = savedEnv.agent;
  if (savedEnv.sessions !== undefined) process.env.PI_CODING_AGENT_SESSION_DIR = savedEnv.sessions;
});

/** Two sessions of the project (old, new) plus one of another cwd in the same dir. */
async function writeFixtures(): Promise<{ oldPath: string; newPath: string }> {
  const oldPath = await writeSession(
    OLD_ID,
    sessionLines(OLD_ID, projectDir, "2026-11-18T09:00:00.000Z", [
      { type: "model_change", id: "m1", parentId: null, timestamp: "2026-11-18T09:00:00.000Z", provider: "anthropic", modelId: "x" },
      message("a1", "system", ""),
      message("a2", "user", "first question"),
      message("a3", "assistant", [{ type: "text", text: "draft" }], { totalTokens: 900 }),
      message("a4", "assistant", [{ type: "text", text: "answer" }], { totalTokens: 1200 }),
    ]),
    new Date("2026-11-18T09:05:00.000Z"),
  );
  const newPath = await writeSession(
    NEW_ID,
    sessionLines(
      NEW_ID,
      projectDir,
      "2026-11-19T08:00:00.000Z",
      [
        message("b1", "user", [{ type: "image", data: "" }]),
        message("b2", "user", [{ type: "text", text: "fix the" }, { type: "text", text: "tests" }]),
        message("b3", "toolResult", [{ type: "text", text: '"role":"user"' }]),
      ],
      oldPath,
    ),
    new Date("2026-11-19T08:30:00.000Z"),
  );
  await writeSession(
    OTHER_CWD_ID,
    sessionLines(OTHER_CWD_ID, "/somewhere/else", "2026-11-19T09:00:00.000Z", []),
    new Date("2026-11-19T09:00:00.000Z"),
  );
  await writeFile(join(sessionDir, "notes.jsonl"), "not json\n");
  return { oldPath, newPath };
}

describe("session listing", () => {
  test("lists the cwd's sessions newest first with contract fields", async () => {
    const { oldPath, newPath } = await writeFixtures();
    liveByPath.set(newPath, { running: true, waiting: false, attached: 1 });
    const { status, json } = await call("GET", `/sessions?cwd=${encodeURIComponent(projectDir)}`);
    expect(status).toBe(200);
    expect(json.sessions.map((session) => session.sessionId)).toEqual([NEW_ID, OLD_ID]);
    expect(json.sessions[0]).toEqual({
      sessionId: NEW_ID,
      sessionPath: newPath,
      cwd: projectDir,
      machine: "local",
      createdAt: "2026-11-19T08:00:00.000Z",
      modifiedAt: "2026-11-19T08:30:00.000Z",
      firstMessage: "fix the tests",
      messageCount: 3,
      tokens: 0,
      parentSession: oldPath,
      live: { running: true, waiting: false, attached: 1 },
      pinned: false,
      archived: false,
      profile: "coding",
    });
    expect(json.sessions[1]).toMatchObject({ firstMessage: "first question", messageCount: 4, tokens: 1200, live: null });
    expect(json.sessions[1].parentSession).toBeUndefined();
  });

  test("usage sums the assistant answers and the reviews a factory task recorded, absent before any answer", async () => {
    const answered = (id: string, input: number, output: number, cacheRead: number, total: number) => ({
      type: "message",
      id,
      parentId: null,
      timestamp: "2026-11-19T10:00:00.000Z",
      message: {
        role: "assistant",
        content: [{ type: "text", text: "ok" }],
        usage: { input, output, cacheRead, cacheWrite: 1, totalTokens: input + output + cacheRead + 1, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total } },
      },
    });
    const review = { type: "custom", id: "r1", parentId: null, timestamp: "2026-11-19T10:00:00.000Z", customType: "factory-review", data: { usage: { input: 5, output: 7, cacheRead: 20, cacheWrite: 0, cost: 0.5 } } };
    await writeSession(OLD_ID, sessionLines(OLD_ID, projectDir, "2026-11-18T09:00:00.000Z", [
      message("u1", "user", "go"),
      answered("a1", 2, 30, 100, 0.25),
      review,
      answered("a2", 3, 40, 200, 0.125),
    ]), new Date("2026-11-18T09:05:00.000Z"));
    await writeSession(NEW_ID, sessionLines(NEW_ID, projectDir, "2026-11-19T08:00:00.000Z", [message("u1", "user", "go")]), new Date("2026-11-19T08:00:00.000Z"));

    const { json } = await call("GET", `/sessions?cwd=${encodeURIComponent(projectDir)}`);

    const byId = new Map(json.sessions.map((session) => [session.sessionId, session]));
    expect(byId.get(OLD_ID)!.usage).toEqual({ input: 2 + 100 + 1 + 5 + 20 + 3 + 200 + 1, output: 30 + 7 + 40, cost: 0.875 });
    expect(byId.get(NEW_ID)!.usage).toBeUndefined();
  });

  test("honours a flat PI_CODING_AGENT_SESSION_DIR", async () => {
    const flatDir = join(scratch, "flat");
    await mkdir(flatDir, { recursive: true });
    await writeFile(join(flatDir, `t_${OLD_ID}.jsonl`), sessionLines(OLD_ID, projectDir, "2026-11-18T09:00:00.000Z", []));
    await writeFile(join(flatDir, `t_${OTHER_CWD_ID}.jsonl`), sessionLines(OTHER_CWD_ID, "/elsewhere", "2026-11-18T09:00:00.000Z", []));
    process.env.PI_CODING_AGENT_SESSION_DIR = flatDir;
    try {
      const { json } = await call("GET", `/sessions?cwd=${encodeURIComponent(projectDir)}`);
      expect(json.sessions.map((session) => [session.sessionId, session.firstMessage])).toEqual([[OLD_ID, "(no messages)"]]);
    } finally {
      delete process.env.PI_CODING_AGENT_SESSION_DIR;
      await rm(flatDir, { recursive: true, force: true });
    }
  });

  test("lists every cwd's sessions without a cwd, newest first", async () => {
    const { oldPath, newPath } = await writeFixtures();
    const elsewhereDir = join(scratch, "agent", "sessions", cwdDirName("/elsewhere"));
    await mkdir(elsewhereDir, { recursive: true });
    const elsewhereId = "01a1f3fe-0000-7000-8000-000000000004";
    const elsewherePath = join(elsewhereDir, `2026-11-19T10-00-00-000Z_${elsewhereId}.jsonl`);
    await writeFile(elsewherePath, sessionLines(elsewhereId, "/elsewhere", "2026-11-17T09:00:00.000Z", []));
    await utimes(elsewherePath, new Date("2026-11-17T09:00:00.000Z"), new Date("2026-11-17T09:00:00.000Z"));

    const { status, json } = await call("GET", "/sessions");
    expect(status).toBe(200);
    expect(json.sessions.map((session) => [session.sessionId, session.cwd])).toEqual([
      [OTHER_CWD_ID, "/somewhere/else"],
      [NEW_ID, projectDir],
      [OLD_ID, projectDir],
      [elsewhereId, "/elsewhere"],
    ]);
    expect(json.sessions.map((session) => session.sessionPath)).toContain(newPath);
    expect(json.sessions.map((session) => session.sessionPath)).toContain(oldPath);
    expect((await call("GET", "/sessions?archived=1")).json.sessions).toEqual([]);
  });

  test("rejects a relative or missing cwd", async () => {
    expect((await call("GET", "/sessions?cwd=relative/dir")).status).toBe(400);
    expect((await call("GET", `/sessions?cwd=${encodeURIComponent(join(scratch, "missing"))}`)).status).toBe(400);
  });

  test("unknown session id is 404", async () => {
    expect((await call("GET", "/sessions/does-not-exist/state")).status).toBe(404);
  });
});

describe("session entries", () => {
  test("answers every entry after the header in append order, the last one the leaf", async () => {
    await writeFixtures();
    const { status, json } = await call<{ entries: { id: string }[]; leafId: string }>("GET", `/sessions/${OLD_ID}/entries`);
    expect(status).toBe(200);
    expect(json.entries.map((entry) => entry.id)).toEqual(["m1", "a1", "a2", "a3", "a4"]);
    expect(json.leafId).toBe("a4");
  });

  test("with turns, answers a preview: the active branch from the turns-th last prompt on, without tool result images", async () => {
    await writeSession(OLD_ID, sessionLines(OLD_ID, projectDir, "2026-11-18T09:00:00.000Z", [
      { ...message("u1", "user", "first"), parentId: null },
      { ...message("r1", "assistant", []), parentId: "u1" },
      { ...message("u2", "user", "second"), parentId: "r1" },
      { ...message("x", "assistant", []), parentId: "u2" },
      { ...message("t2", "toolResult", [{ type: "text", text: "shot" }, { type: "image", data: "png" }]), parentId: "u2" },
      { ...message("r2", "assistant", []), parentId: "t2" },
    ]), new Date("2026-11-18T09:05:00.000Z"));
    const { json } = await call<{ entries: { id: string; message: { content: unknown } }[]; leafId: string }>("GET", `/sessions/${OLD_ID}/entries?turns=1`);
    expect(json.entries.map((entry) => entry.id)).toEqual(["u2", "t2", "r2"]);
    expect(json.entries[1].message.content).toEqual([{ type: "text", text: "shot" }]);
    expect(json.leafId).toBe("r2");
    expect((await call("GET", `/sessions/${OLD_ID}/entries?turns=0`)).status).toBe(400);
  });

  test("takes only GET and 404s an unknown session", async () => {
    await writeFixtures();
    expect((await call("POST", `/sessions/${OLD_ID}/entries`)).status).toBe(405);
    expect((await call("GET", "/sessions/missing/entries")).status).toBe(404);
  });
});

describe("filtered session entries", () => {
  type EntriesBody = { entries: { id: string }[]; leafId: string | null; error?: string };

  /** A `custom` entry as `pi.appendEntry` writes it. */
  function custom(id: string, customType: string, data: object = {}): object {
    return { type: "custom", id, parentId: null, timestamp: "2026-11-19T10:00:00.000Z", customType, data };
  }

  async function writeTimeline(): Promise<void> {
    await writeSession(OLD_ID, sessionLines(OLD_ID, projectDir, "2026-11-18T09:00:00.000Z", [
      custom("c1", "factory-task"),
      message("u1", "user", "go"),
      custom("c2", "factory-round", { round: 1 }),
      custom("c3", "other-thing"),
      { type: "custom_message", id: "f1", parentId: null, timestamp: "2026-11-19T10:00:00.000Z", customType: "factory-feedback", content: "fix" },
      custom("c4", "factory-state", { state: "running", round: 1, reason: "" }),
      message("a1", "assistant", []),
    ]), new Date("2026-11-18T09:07:00.000Z"));
  }

  async function entries(query: string): Promise<{ status: number; json: EntriesBody }> {
    return call<EntriesBody>("GET", `/sessions/${OLD_ID}/entries?${query}`);
  }

  test("customType keeps the custom entries whose type starts with the prefix, the leaf stays the file's", async () => {
    await writeTimeline();
    const { status, json } = await entries("customType=factory-");
    expect(status).toBe(200);
    expect(json.entries.map((entry) => entry.id)).toEqual(["c1", "c2", "c4"]);
    expect(json.leafId).toBe("a1");
    expect((await entries("customType=factory-state")).json.entries.map((entry) => entry.id)).toEqual(["c4"]);
    expect((await entries("customType=nothing-")).json).toEqual({ entries: [], leafId: "a1" });
  });

  test("since keeps what was appended after the entry, of any type without customType", async () => {
    await writeTimeline();
    expect((await entries("customType=factory-&since=c1")).json.entries.map((entry) => entry.id)).toEqual(["c2", "c4"]);
    expect((await entries("customType=factory-&since=u1")).json.entries.map((entry) => entry.id)).toEqual(["c2", "c4"]);
    expect((await entries("since=c3")).json.entries.map((entry) => entry.id)).toEqual(["f1", "c4", "a1"]);
    expect((await entries("customType=factory-&since=a1")).json).toEqual({ entries: [], leafId: "a1" });
  });

  test("assistant adds each assistant answer cut to its text, model and usage, without tool calls or thinking", async () => {
    await writeSession(OLD_ID, sessionLines(OLD_ID, projectDir, "2026-11-18T09:00:00.000Z", [
      custom("c1", "factory-round", { round: 1 }),
      message("u1", "user", "go"),
      {
        type: "message",
        id: "a1",
        parentId: null,
        timestamp: "2026-11-19T10:00:00.000Z",
        message: {
          role: "assistant",
          content: [{ type: "thinking", thinking: "hmm" }, { type: "text", text: "Done." }, { type: "toolCall", id: "t", name: "bash", arguments: {} }],
          model: "claude-sonnet-5-5",
          stopReason: "stop",
          usage: { input: 2, output: 30, cacheRead: 100, cacheWrite: 8, totalTokens: 140, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0.25 } },
        },
      },
      message("r1", "toolResult", "ok"),
    ]), new Date("2026-11-18T09:07:00.000Z"));

    const { json } = await entries("customType=factory-&assistant=1");

    expect(json.entries.map((entry) => entry.id)).toEqual(["c1", "a1"]);
    expect(json.entries[1] as unknown).toEqual({
      type: "message",
      id: "a1",
      timestamp: "2026-11-19T10:00:00.000Z",
      message: { role: "assistant", text: "Done.", model: "claude-sonnet-5-5", stopReason: "stop", usage: { input: 110, output: 30, cost: 0.25 } },
    });
    expect((await entries("customType=factory-")).json.entries.map((entry) => entry.id)).toEqual(["c1"]);
    expect((await entries("turns=1&assistant=1")).json.error).toBe("invalid_query");
  });

  test("an unknown since is 400 invalid_since, turns takes neither parameter", async () => {
    await writeTimeline();
    const unknown = await entries("customType=factory-&since=nope");
    expect(unknown.status).toBe(400);
    expect(unknown.json.error).toBe("invalid_since");
    expect((await entries("since=")).json.error).toBe("invalid_since");
    for (const query of ["turns=1&customType=factory-", "turns=1&since=c1"]) {
      const combined = await entries(query);
      expect(combined.status).toBe(400);
      expect(combined.json.error).toBe("invalid_query");
    }
  });

  test("an unchanged file answers each query its own cut", async () => {
    await writeTimeline();
    const whole = (await call<EntriesBody>("GET", `/sessions/${OLD_ID}/entries`)).json.entries.length;
    expect(whole).toBe(7);
    expect((await entries("customType=factory-")).json.entries.length).toBe(3);
    expect((await entries("customType=factory-&since=c2")).json.entries.length).toBe(1);
    expect((await entries("turns=1")).json.entries.map((entry) => entry.id)).toEqual(["a1"]);
    expect((await call<EntriesBody>("GET", `/sessions/${OLD_ID}/entries`)).json.entries.length).toBe(whole);
  });
});

describe("factory state", () => {
  /** A `factory-state` entry as the factory extension appends it. */
  function state(id: string, timestamp: string, data: object): object {
    return { type: "custom", id, parentId: null, timestamp, customType: "factory-state", data };
  }

  /** Write the one fixture session with `lines` after the header and answer its listed row. */
  async function listed(lines: string[], minute: number): Promise<SessionInfo> {
    const header = sessionLines(OLD_ID, projectDir, "2026-11-18T09:00:00.000Z", []);
    await writeSession(OLD_ID, header + lines.map((line) => `${line}\n`).join(""), new Date(Date.UTC(2026, 10, 18, 10, minute)));
    const { json } = await call("GET", `/sessions?cwd=${encodeURIComponent(projectDir)}`);
    expect(json.sessions.map((session) => session.sessionId)).toEqual([OLD_ID]);
    return json.sessions[0]!;
  }

  const user = JSON.stringify(message("u1", "user", "build it"));

  test("a session without a factory-state entry has no factory field", async () => {
    const quoted = message("a1", "assistant", [{ type: "text", text: '{"type":"custom","customType":"factory-state"}' }]);
    const feedback = { type: "custom_message", id: "f1", parentId: null, timestamp: "2026-11-19T10:00:00.000Z", customType: "factory-state", content: "x" };
    const other = { type: "custom", id: "c1", parentId: null, timestamp: "2026-11-19T10:00:00.000Z", customType: "factory-task", data: { state: "done" } };
    const session = await listed([user, JSON.stringify(quoted), JSON.stringify(feedback), JSON.stringify(other)], 1);
    expect("factory" in session).toBe(false);
    expect(session.messageCount).toBe(2);
  });

  test("one entry gives state, round, reason and the entry's timestamp", async () => {
    const entry = state("s1", "2026-11-18T09:30:00.000Z", { state: "blocked", round: 3, reason: "gates failed" });
    const session = await listed([user, JSON.stringify(entry)], 2);
    expect(session.factory).toEqual({ state: "blocked", round: 3, reason: "gates failed", at: "2026-11-18T09:30:00.000Z" });
    expect(session.firstMessage).toBe("build it");
  });

  test("of several entries the last in file order wins, whatever their timestamps", async () => {
    const session = await listed([
      JSON.stringify(state("s1", "2026-11-18T09:50:00.000Z", { state: "running", round: 1, reason: "" })),
      user,
      JSON.stringify(state("s2", "2026-11-18T09:40:00.000Z", { state: "waiting_approve", round: 2, reason: "review ok" })),
      JSON.stringify(state("s3", "2026-11-18T09:30:00.000Z", { state: "done", round: 2, reason: "merged" })),
    ], 3);
    expect(session.factory).toEqual({ state: "done", round: 2, reason: "merged", at: "2026-11-18T09:30:00.000Z" });
  });

  test("a malformed entry is skipped: the last valid one stays, none leaves the field out", async () => {
    const valid = JSON.stringify(state("s1", "2026-11-18T09:30:00.000Z", { state: "merging", round: 4, reason: "" }));
    const malformed = [
      JSON.stringify(state("s2", "2026-11-18T09:31:00.000Z", { state: "paused", round: 4, reason: "" })),
      JSON.stringify(state("s3", "2026-11-18T09:32:00.000Z", { state: "done", round: "4", reason: "" })),
      JSON.stringify(state("s4", "2026-11-18T09:33:00.000Z", { state: "done", round: 4 })),
      JSON.stringify(state("s5", "2026-11-18T09:34:00.000Z", { state: "done", round: 4, reason: "" })).slice(0, -9),
    ];
    const kept = await listed([user, valid, ...malformed], 4);
    expect(kept.factory).toEqual({ state: "merging", round: 4, reason: "", at: "2026-11-18T09:30:00.000Z" });
    expect("factory" in (await listed([user, ...malformed], 5))).toBe(false);
  });

  test("the state follows the file through the summary cache, into single rows and search hits", async () => {
    // Chained by `parentId`: search reads the active branch only
    const running = JSON.stringify({ ...state("s1", "2026-11-18T09:30:00.000Z", { state: "running", round: 1, reason: "" }), parentId: "u1" });
    expect((await listed([user, running], 6)).factory?.state).toBe("running");
    expect((await listed([user, running], 6)).factory?.state).toBe("running");
    const done = JSON.stringify({ ...state("s2", "2026-11-18T09:35:00.000Z", { state: "done", round: 1, reason: "merged" }), parentId: "s1" });
    expect((await listed([user, running, done], 7)).factory).toEqual({ state: "done", round: 1, reason: "merged", at: "2026-11-18T09:35:00.000Z" });

    const pinned = await call("PUT", `/sessions/${OLD_ID}/pin`, { pinned: true });
    expect(pinned.json.session.factory?.state).toBe("done");
    const search = await call<{ hits: { session: SessionInfo }[] }>("GET", "/search?q=build");
    expect(search.json.hits.map((hit) => hit.session.factory?.state)).toEqual(["done"]);
  });
});

describe("archive and pin", () => {
  test("archive and restore move the session and its sidecar", async () => {
    const { oldPath } = await writeFixtures();
    expect((await call("PUT", `/sessions/${OLD_ID}/pin`, { pinned: true })).json.session.pinned).toBe(true);

    const archived = await call("POST", `/sessions/${OLD_ID}/archive`);
    expect(archived.status).toBe(200);
    const archivedPath = join(scratch, "agent", "sessions-archive", cwdDirName(projectDir), `2026-11-19T10-00-00-000Z_${OLD_ID}.jsonl`);
    expect(archived.json.session).toMatchObject({ sessionId: OLD_ID, sessionPath: archivedPath, archived: true, pinned: true });
    expect(existsSync(oldPath)).toBe(false);
    expect(existsSync(`${oldPath}.gigachad.json`)).toBe(false);
    expect(existsSync(`${archivedPath}.gigachad.json`)).toBe(true);

    const liveList = await call("GET", `/sessions?cwd=${encodeURIComponent(projectDir)}`);
    expect(liveList.json.sessions.map((session) => session.sessionId)).toEqual([NEW_ID]);
    const archivedList = await call("GET", `/sessions?cwd=${encodeURIComponent(projectDir)}&archived=1`);
    expect(archivedList.json.sessions.map((session) => session.sessionId)).toEqual([OLD_ID]);
    expect((await call("POST", `/sessions/${OLD_ID}/archive`)).status).toBe(409);

    const restored = await call("POST", `/sessions/${OLD_ID}/restore`);
    expect(restored.status).toBe(200);
    expect(restored.json.session).toMatchObject({ sessionPath: oldPath, archived: false, pinned: true });
    expect(existsSync(archivedPath)).toBe(false);
    expect(await isPinned(oldPath)).toBe(true);
  });

  test("archive closes a detached idle child first", async () => {
    const { newPath } = await writeFixtures();
    liveByPath.set(newPath, { running: false, waiting: false, attached: 0 });
    const archived = await call("POST", `/sessions/${NEW_ID}/archive`);
    expect(archived.status).toBe(200);
    expect(archived.json.session).toMatchObject({ sessionId: NEW_ID, archived: true, live: null });
    expect(closed).toEqual([newPath]);
    expect(existsSync(newPath)).toBe(false);
  });

  test("archive is refused while the child runs, has a client or asks", async () => {
    const { newPath } = await writeFixtures();
    for (const status of [
      { running: true, waiting: false, attached: 0 },
      { running: false, waiting: false, attached: 1 },
      { running: false, waiting: true, attached: 0 },
    ]) {
      liveByPath.set(newPath, status);
      const { status: code, json } = await call<{ error: string }>("POST", `/sessions/${NEW_ID}/archive`);
      expect(code).toBe(409);
      expect(json).toEqual({ error: "live" });
    }
    expect(closed).toEqual([]);
    expect(existsSync(newPath)).toBe(true);
  });

  test("archive is refused while another holder has the session's flock (a spawning child)", async () => {
    const { newPath } = await writeFixtures();
    const lock = tryLockFile(newPath)!;
    try {
      const { status, json } = await call<{ error: string }>("POST", `/sessions/${NEW_ID}/archive`);
      expect(status).toBe(409);
      expect(json).toEqual({ error: "live" });
      expect(existsSync(newPath)).toBe(true);
    } finally {
      lock.release();
    }
    expect((await call("POST", `/sessions/${NEW_ID}/archive`)).status).toBe(200);
  });

  test("pin round trip", async () => {
    const { oldPath } = await writeFixtures();
    expect(await isPinned(oldPath)).toBe(false);
    expect((await call("PUT", `/sessions/${OLD_ID}/pin`, { pinned: true })).json.session.pinned).toBe(true);
    expect(await isPinned(oldPath)).toBe(true);
    expect((await call("PUT", `/sessions/${OLD_ID}/pin`, { pinned: false })).json.session.pinned).toBe(false);
    expect(await isPinned(oldPath)).toBe(false);
    expect((await call("PUT", `/sessions/${OLD_ID}/pin`, { pinned: "yes" })).status).toBe(400);
  });
});

test("a sidecar read error throws instead of reading as unpinned, missing or invalid JSON is the default", async () => {
  const { oldPath, newPath } = await writeFixtures();
  await mkdir(`${oldPath}.gigachad.json`);
  await expect(isPinned(oldPath)).rejects.toThrow();
  await writeFile(`${newPath}.gigachad.json`, "not json");
  expect(await isPinned(newPath)).toBe(false);
  await rm(`${oldPath}.gigachad.json`, { recursive: true });
  expect(await isPinned(oldPath)).toBe(false);
});

describe("private state", () => {
  test("revisioned writes reject a stale revision", async () => {
    const { oldPath } = await writeFixtures();
    expect((await call<StateBody>("GET", `/sessions/${OLD_ID}/state`)).json).toEqual({ revision: 0, data: {} });
    expect((await call<StateBody>("PUT", `/sessions/${OLD_ID}/state`, { revision: 0, data: { draft: "a" } })).json).toEqual({ revision: 1 });
    const stale = await call<StateBody>("PUT", `/sessions/${OLD_ID}/state`, { revision: 0, data: { draft: "b" } });
    expect(stale.status).toBe(409);
    expect(stale.json).toEqual({ revision: 1, data: { draft: "a" } });
    expect((await call<StateBody>("PUT", `/sessions/${OLD_ID}/state`, { revision: 1, data: { draft: "c" } })).json).toEqual({ revision: 2 });
    expect((await call<StateBody>("GET", `/sessions/${OLD_ID}/state`)).json).toEqual({ revision: 2, data: { draft: "c" } });
    expect(JSON.parse(await readFile(`${oldPath}.gigachad.json`, "utf8"))).toEqual({ revision: 2, pinned: false, data: { draft: "c" } });
    expect((await call("PUT", `/sessions/${OLD_ID}/state`, { revision: 2, data: [] })).status).toBe(400);
  });

  test("concurrent writes at one revision let exactly one through", async () => {
    await writeFixtures();
    const results = await Promise.all(
      ["a", "b", "c"].map((draft) => call("PUT", `/sessions/${OLD_ID}/state`, { revision: 0, data: { draft } })),
    );
    expect(results.map((result) => result.status).sort()).toEqual([200, 409, 409]);
  });

  test("copyPrivateState copies data and the profile marker to a fork, unpinned", async () => {
    const { oldPath, newPath } = await writeFixtures();
    await call("PUT", `/sessions/${OLD_ID}/pin`, { pinned: true });
    await call("PUT", `/sessions/${OLD_ID}/state`, { revision: 0, data: { card: 1 } });
    await copyPrivateState(oldPath, newPath);
    expect((await call<StateBody>("GET", `/sessions/${NEW_ID}/state`)).json).toEqual({ revision: 0, data: { card: 1 } });
    expect(await isPinned(newPath)).toBe(false);

    await writeFile(`${oldPath}.gigachad.json`, JSON.stringify({ revision: 0, pinned: false, data: {}, profile: "document" }));
    await copyPrivateState(oldPath, newPath);
    expect(JSON.parse(await readFile(`${newPath}.gigachad.json`, "utf8"))).toEqual({ revision: 0, pinned: false, data: {}, profile: "document" });
  });

  test("the profile marker shows in SessionInfo, filters the listing and survives state and pin writes", async () => {
    const { oldPath } = await writeFixtures();
    await writeFile(`${oldPath}.gigachad.json`, JSON.stringify({ revision: 0, pinned: false, data: {}, profile: "document" }));
    await call("PUT", `/sessions/${OLD_ID}/state`, { revision: 0, data: { draft: "a" } });
    await call("PUT", `/sessions/${OLD_ID}/pin`, { pinned: true });
    expect(JSON.parse(await readFile(`${oldPath}.gigachad.json`, "utf8"))).toMatchObject({ profile: "document", pinned: true });

    const cwd = encodeURIComponent(projectDir);
    const listed = async (query: string) => (await call("GET", `/sessions?cwd=${cwd}${query}`)).json.sessions;
    expect((await listed("")).map((session) => [session.sessionId, session.profile])).toEqual([
      [NEW_ID, "coding"],
      [OLD_ID, "document"],
    ]);
    expect((await listed("&profile=document")).map((session) => session.sessionId)).toEqual([OLD_ID]);
    expect((await listed("&profile=coding")).map((session) => session.sessionId)).toEqual([NEW_ID]);
    expect((await call("GET", `/sessions?cwd=${cwd}&profile=remote`)).json.error).toBe("invalid_profile");

    await writeFile(`${oldPath}.gigachad.json`, JSON.stringify({ revision: 0, pinned: false, data: {}, profile: "other" }));
    expect((await listed("&profile=coding")).map((session) => session.sessionId)).toEqual([NEW_ID, OLD_ID]);
  });
});

describe("factory draft", () => {
  test("asks the installed factory plugin, and names its source when it is missing", async () => {
    const repo = join(scratch, "draft-repo");
    await mkdir(repo, { recursive: true });
    const draftUrl = `/factory/draft?cwd=${encodeURIComponent(repo)}`;
    const missing = await call<{ error: string; message: string }>("GET", draftUrl);
    expect(missing.status).toBe(503);
    expect(missing.json.error).toBe("factory_plugin_missing");
    expect(missing.json.message).toContain("github.com/patrickab/.os/tree/master/chezmoi/dot_pi/agent/extensions/factory");

    // The plugin is a Pi user extension: a stand-in in the agent dir is what pi-host loads.
    const plugin = join(scratch, "agent", "extensions", "factory");
    await mkdir(plugin, { recursive: true });
    await writeFile(
      join(plugin, "draft.ts"),
      'export const initDraft = (root: string) => ({ text: "gates:\\n  - [make, lint]\\n", source: "Makefile" });\n',
    );
    const { status, json } = await call<{ draft: string; source: string }>("GET", `/factory/draft?cwd=${encodeURIComponent(repo)}`);
    expect(status).toBe(200);
    expect(json.source).toBe("Makefile");
    expect(json.draft).toContain("gates:\n  - [make, lint]");
    expect(existsSync(join(repo, "factory.yaml"))).toBe(false);
    expect((await call("GET", "/factory/draft?cwd=relative")).status).toBe(400);
  });
});

describe("file completion", () => {
  test("ranks git files by fuzzy match, basename first", async () => {
    const repo = join(scratch, "repo");
    const files = [
      "src/lib/pi_host.py",
      "src/lib/pi_rpc.py",
      "tests/test_pi_host.py",
      "pi-host/src/server.ts",
      "pi-host/src/catalog/index.ts",
      "src/frontend/features/pi/PiHostPanel.tsx",
      "docs/philosophy/history.md",
      "README.md",
    ];
    for (const file of files) {
      await mkdir(join(repo, file, ".."), { recursive: true });
      await writeFile(join(repo, file), "");
    }
    await writeFile(join(repo, ".gitignore"), "ignored.txt\n");
    await writeFile(join(repo, "ignored.txt"), "");
    await run(repo, "git", "init", "-q");
    await run(repo, "git", "add", "src/lib/pi_host.py", "README.md");

    const ranked = await call("GET", `/files?cwd=${encodeURIComponent(repo)}&q=pihost&limit=4`);
    expect(ranked.status).toBe(200);
    expect(ranked.json.files[0].path).toBe("src/lib/pi_host.py");
    expect(ranked.json.files).toHaveLength(4);
    expect(ranked.json.files.map((file) => file.path)).not.toContain("README.md");
    for (const [index, file] of ranked.json.files.entries()) {
      if (index > 0) expect(file.score).toBeLessThanOrEqual(ranked.json.files[index - 1].score);
    }

    const all = await call("GET", `/files?cwd=${encodeURIComponent(repo)}&limit=500`);
    const paths = all.json.files.map((file) => file.path);
    expect(paths).toContain(".gitignore");
    expect(paths).not.toContain("ignored.txt");
    expect(paths).toHaveLength(files.length + 1);
  });

  test("walks a non-git directory, skipping hidden and dependency dirs", async () => {
    const plain = join(scratch, "plain");
    await mkdir(join(plain, "notes"), { recursive: true });
    await mkdir(join(plain, "node_modules", "dep"), { recursive: true });
    await mkdir(join(plain, ".cache"), { recursive: true });
    await writeFile(join(plain, "notes", "pi_host.md"), "");
    await writeFile(join(plain, "node_modules", "dep", "pi_host.js"), "");
    await writeFile(join(plain, ".cache", "pi_host.bin"), "");
    const { json } = await call("GET", `/files?cwd=${encodeURIComponent(plain)}&q=pihost`);
    expect(json.files.map((file) => file.path)).toEqual(["notes/pi_host.md"]);
  });

  test("rejects an invalid cwd", async () => {
    expect((await call("GET", "/files?cwd=relative&q=x")).status).toBe(400);
  });

  test("without a query answers the cached list unranked, whole unless a limit is given (K6)", async () => {
    const many = join(scratch, "many");
    await mkdir(many, { recursive: true });
    for (let index = 0; index < 60; index++) await writeFile(join(many, `file-${index}.txt`), "");
    const { json } = await call("GET", `/files?cwd=${encodeURIComponent(many)}&q=`);
    expect(json.files).toHaveLength(60);
    expect(json.files.every((file) => file.score === 0)).toBe(true);
    expect((await call("GET", `/files?cwd=${encodeURIComponent(many)}&limit=5`)).json.files).toHaveLength(5);
  });
});

describe("transcript search (M24)", () => {
  type SearchBody = { hits: { session: SessionInfo; snippet: string; entryId: string }[] };

  /** Entries chained by `parentId` in order, the last one the leaf. */
  function chained(...entries: [id: string, role: string, text: string][]): object[] {
    return entries.map(([id, role, text], index) => ({
      ...message(id, role, role === "user" ? text : [{ type: "text", text }]),
      parentId: index === 0 ? null : entries[index - 1]![0],
    }));
  }

  async function search(query: string): Promise<SearchBody> {
    return (await call<SearchBody>("GET", `/search?q=${encodeURIComponent(query)}`)).json;
  }

  test("finds user and assistant text with a snippet around the hit, newest first on equal scores", async () => {
    const filler = "lorem ipsum ".repeat(20);
    const older = await writeSession(
      OLD_ID,
      sessionLines(OLD_ID, projectDir, "2026-11-18T09:00:00.000Z", chained(["o1", "user", "where is the frobnicator"])),
      new Date("2026-11-18T09:05:00.000Z"),
    );
    const newer = await writeSession(
      NEW_ID,
      sessionLines(
        NEW_ID,
        projectDir,
        "2026-11-19T08:00:00.000Z",
        chained(["n1", "user", "hello"], ["n2", "assistant", `${filler}the frobnicator lives in settings ${filler}`]),
      ),
      new Date("2026-11-19T08:30:00.000Z"),
    );

    const { hits } = await search("frobnicator");
    expect(hits.map((hit) => [hit.session.sessionPath, hit.entryId])).toEqual([
      [newer, "n2"],
      [older, "o1"],
    ]);
    expect(hits[0]!.snippet).toMatch(/^….*the frobnicator lives in settings.*…$/);
    expect(hits[0]!.snippet.length).toBeLessThan(220);
    expect(hits[1]!.snippet).toBe("where is the frobnicator");
    expect(hits[0]!.session).toMatchObject({ sessionId: NEW_ID, machine: "local", live: null });
    expect((await search("")).hits).toEqual([]);
  });

  test("skips tool output, thinking and abandoned branches; a large tool result is not parsed for text", async () => {
    const toolResult = { ...message("t2", "toolResult", [{ type: "text", text: `zanzibar ${"x".repeat(70_000)}` }]), parentId: "t1" };
    const abandoned = { ...message("t3", "user", "zanzibar on a dead branch"), parentId: "t1" };
    const leaf = {
      ...message("t4", "assistant", [
        { type: "thinking", thinking: "zanzibar" },
        { type: "text", text: "done" },
      ]),
      parentId: "t2",
    };
    await writeSession(
      NEW_ID,
      sessionLines(NEW_ID, projectDir, "2026-11-19T08:00:00.000Z", [...chained(["t1", "user", "run it"]), toolResult, abandoned, leaf]),
      new Date("2026-11-19T08:30:00.000Z"),
    );
    expect((await search("zanzibar")).hits).toEqual([]);
    expect((await search("done")).hits.map((hit) => hit.entryId)).toEqual(["t4"]);
  });

  test("an unchanged file (same size and mtime) is not parsed again", async () => {
    const mtime = new Date("2026-11-19T08:30:00.000Z");
    const content = (word: string) => sessionLines(NEW_ID, projectDir, "2026-11-19T08:00:00.000Z", chained(["c1", "user", word]));
    await writeSession(NEW_ID, content("aardvark"), mtime);
    expect((await search("aardvark")).hits).toHaveLength(1);

    // Same length, same mtime: the cached transcript answers.
    await writeSession(NEW_ID, content("pangolin"), mtime);
    expect((await search("aardvark")).hits).toHaveLength(1);
    expect((await search("pangolin")).hits).toEqual([]);

    await writeSession(NEW_ID, content("pangolin"), new Date("2026-11-19T08:31:00.000Z"));
    expect((await search("pangolin")).hits).toHaveLength(1);
    expect((await search("aardvark")).hits).toEqual([]);
  });
});
