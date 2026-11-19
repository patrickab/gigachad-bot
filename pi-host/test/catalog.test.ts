/** Catalog routes against fixture Pi session files in a scratch agent dir. */
import { afterAll, beforeAll, beforeEach, describe, expect, test } from "bun:test";
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, readFile, rm, utimes, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { copyPrivateState, handleCatalog, isPinned, type LiveStatus, type SessionInfo } from "../src/catalog/index.ts";
import type { RankedFile } from "../src/catalog/files.ts";
import { cwdDirName } from "../src/catalog/paths.ts";
import type { LiveSessions } from "../src/supervisor/index.ts";
import { tryLockFile } from "../src/supervisor/lock.ts";

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
const live: LiveSessions = { liveStatus: (path) => liveByPath.get(path) };
const savedEnv = { agent: process.env.PI_CODING_AGENT_DIR, sessions: process.env.PI_CODING_AGENT_SESSION_DIR };

const OLD_ID = "01a1f3fe-0000-7000-8000-000000000001";
const NEW_ID = "01a1f3fe-0000-7000-8000-000000000002";
const OTHER_CWD_ID = "01a1f3fe-0000-7000-8000-000000000003";

/** JSONL session file content: header plus entries. */
function sessionLines(id: string, cwd: string, timestamp: string, entries: object[], parentSession?: string): string {
  const header = { type: "session", version: 3, id, timestamp, cwd, ...(parentSession ? { parentSession } : {}) };
  return [header, ...entries].map((entry) => JSON.stringify(entry)).join("\n") + "\n";
}

/** A `message` entry. */
function message(id: string, role: string, content: unknown): object {
  return { type: "message", id, parentId: null, timestamp: "2026-11-19T10:00:00.000Z", message: { role, content, timestamp: 0 } };
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
      message("a3", "assistant", [{ type: "text", text: "answer" }]),
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
      parentSession: oldPath,
      live: { running: true, waiting: false, attached: 1 },
      pinned: false,
      archived: false,
    });
    expect(json.sessions[1]).toMatchObject({ firstMessage: "first question", messageCount: 3, live: null });
    expect(json.sessions[1].parentSession).toBeUndefined();
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

  test("rejects a relative or missing cwd", async () => {
    expect((await call("GET", "/sessions?cwd=relative/dir")).status).toBe(400);
    expect((await call("GET", `/sessions?cwd=${encodeURIComponent(join(scratch, "missing"))}`)).status).toBe(400);
  });

  test("unknown session id is 404", async () => {
    expect((await call("GET", "/sessions/does-not-exist/state")).status).toBe(404);
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

  test("archive is refused while the session is live", async () => {
    const { newPath } = await writeFixtures();
    liveByPath.set(newPath, { running: false, waiting: false, attached: 0 });
    const { status, json } = await call<{ error: string }>("POST", `/sessions/${NEW_ID}/archive`);
    expect(status).toBe(409);
    expect(json).toEqual({ error: "live" });
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

  test("copyPrivateState copies data to a fork, unpinned", async () => {
    const { oldPath, newPath } = await writeFixtures();
    await call("PUT", `/sessions/${OLD_ID}/pin`, { pinned: true });
    await call("PUT", `/sessions/${OLD_ID}/state`, { revision: 0, data: { card: 1 } });
    await copyPrivateState(oldPath, newPath);
    expect((await call<StateBody>("GET", `/sessions/${NEW_ID}/state`)).json).toEqual({ revision: 0, data: { card: 1 } });
    expect(await isPinned(newPath)).toBe(false);
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
});
