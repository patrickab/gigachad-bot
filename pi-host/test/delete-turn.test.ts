/** `POST /sessions/:id/turns/:entryId/delete` against the fake `pi` fixture: rewrite, restart, carried clients, cleanup. */
import { afterAll, beforeAll, expect, test } from "bun:test";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Server } from "bun";
import { startServer } from "../src/server.ts";
import { deleteTurn } from "../src/host/local/delete-turn.ts";
import { forgetMedia, mediaBytes, useMediaDir } from "../src/media.ts";
import { createSupervisor, type SocketData, type Supervisor } from "../src/supervisor/index.ts";

type Frame = { type: string; sessionId: string; [field: string]: any };

const TOKEN = "test-token";
const TOKENS = { full: TOKEN, gateway: "test-gateway-token" };
const root = mkdtempSync(join(tmpdir(), "pi-host-delete-turn-"));
const agentDir = join(root, "agent");
const mediaDir = join(root, "media");
const savedEnv = { agentDir: process.env.PI_CODING_AGENT_DIR, sessionDir: process.env.PI_CODING_AGENT_SESSION_DIR };

const fakePi = join(root, "pi");
writeFileSync(fakePi, `#!/bin/sh\nexec "${process.execPath}" "${join(import.meta.dir, "fixtures", "fake-pi.ts")}" "$@"\n`);
chmodSync(fakePi, 0o755);

let supervisor: Supervisor;
let server: Server<SocketData>;
const sockets: WebSocket[] = [];

async function until<T>(probe: () => T | undefined, timeoutMs = 20_000): Promise<T> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const value = probe();
    if (value !== undefined) return value;
    await Bun.sleep(10);
  }
  throw new Error("timed out");
}

function api(path: string, init: RequestInit = {}): Promise<Response> {
  return fetch(`http://127.0.0.1:${server.port}${path}`, {
    ...init,
    headers: { authorization: `Bearer ${TOKEN}`, "content-type": "application/json", ...init.headers },
  });
}

function deleteRequest(sessionId: string, entryId: string, body: unknown): Promise<Response> {
  return api(`/sessions/${sessionId}/turns/${entryId}/delete`, { method: "POST", body: JSON.stringify(body) });
}

async function start(cwd: string, sessionPath: string): Promise<string> {
  const response = await api("/sessions", { method: "POST", body: JSON.stringify({ cwd, profile: "coding", resume: { sessionPath } }) });
  expect(response.status).toBe(200);
  return (await response.json()).session.sessionId;
}

/** A `/ws` client keeping every frame. */
async function client(): Promise<{ frames: Frame[]; send(frame: object): void }> {
  const options = { headers: { authorization: `Bearer ${TOKEN}` } } as unknown as string[];
  const socket = new WebSocket(`ws://127.0.0.1:${server.port}/ws`, options);
  sockets.push(socket);
  const frames: Frame[] = [];
  socket.addEventListener("message", (event) => frames.push(JSON.parse(String(event.data))));
  await until(() => (socket.readyState === WebSocket.OPEN ? true : undefined));
  return { frames, send: (frame) => socket.send(JSON.stringify(frame)) };
}

function workdir(name: string): string {
  const path = join(root, "work", name);
  mkdirSync(path, { recursive: true });
  return path;
}

function sessionDir(base: string, cwd: string): string {
  return join(base, `--${cwd.replace(/^\//, "").replace(/[/\\:]/g, "-")}--`);
}

function writeSessionFile(cwd: string, entries: object[], base = join(agentDir, "sessions")): { sessionId: string; sessionPath: string } {
  const sessionId = crypto.randomUUID();
  const dir = sessionDir(base, cwd);
  mkdirSync(dir, { recursive: true });
  const sessionPath = join(dir, `2026-11-19T10-00-00-000Z_${sessionId}.jsonl`);
  const header = { type: "session", version: 3, id: sessionId, timestamp: new Date().toISOString(), cwd };
  writeFileSync(sessionPath, [header, ...entries].map((line) => `${JSON.stringify(line)}\n`).join(""));
  return { sessionId, sessionPath };
}

function message(id: string, parentId: string | null, role: "user" | "assistant", text: string): object {
  const content = role === "user" ? text : [{ type: "text", text }];
  return { type: "message", id, parentId, timestamp: new Date().toISOString(), message: { role, content, timestamp: 0 } };
}

/** A PNG the media store accepts (magic bytes are all it checks), unique per `tag`. */
function png(tag: string): string {
  return Buffer.concat([Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), Buffer.from(`pixels of ${tag}`)]).toString("base64");
}

function sha256(base64: string): string {
  return new Bun.CryptoHasher("sha256").update(Buffer.from(base64, "base64")).digest("hex");
}

function toolResult(id: string, parentId: string, text: string, image: string): object {
  const content = [{ type: "text", text }, { type: "image", mimeType: "image/png", data: image }];
  return { type: "message", id, parentId, timestamp: new Date().toISOString(), message: { role: "toolResult", toolCallId: "c1", toolName: "shot", content, timestamp: 0 } };
}

function ids(entries: { id: string }[]): string[] {
  return entries.map((entry) => entry.id);
}

beforeAll(async () => {
  process.env.PI_CODING_AGENT_DIR = agentDir;
  delete process.env.PI_CODING_AGENT_SESSION_DIR;
  mkdirSync(join(agentDir, "sessions"), { recursive: true });
  mkdirSync(join(agentDir, "factory", "sessions"), { recursive: true });
  useMediaDir(mediaDir);
  supervisor = await createSupervisor({ piBin: fakePi, idleMs: 3_600_000, isPinned: async () => false });
  server = startServer({ hostname: "127.0.0.1", port: 0, tokens: TOKENS, supervisor });
});

afterAll(async () => {
  for (const socket of sockets) socket.close();
  await supervisor.shutdown();
  server.stop(true);
  useMediaDir(undefined);
  if (savedEnv.agentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
  else process.env.PI_CODING_AGENT_DIR = savedEnv.agentDir;
  if (savedEnv.sessionDir !== undefined) process.env.PI_CODING_AGENT_SESSION_DIR = savedEnv.sessionDir;
  rmSync(root, { recursive: true, force: true });
});

test("deleting a turn leaves nothing of it in the file, the caches, the media store or the pastes", async () => {
  const cwd = workdir("one-turn");
  const paste = ".pi/pastes/20261119-100000-secret-clip.mp4";
  mkdirSync(join(cwd, ".pi", "pastes"), { recursive: true });
  writeFileSync(join(cwd, paste), "video bytes");
  const image = png("secret screenshot");
  const { sessionId, sessionPath } = writeSessionFile(cwd, [
    message("u1", null, "user", `zebraprompt look at @${paste}`),
    message("a1", "u1", "assistant", "zebraanswer here"),
    toolResult("t1", "a1", "zebratool output", image),
    message("a1b", "t1", "assistant", "zebrafinal"),
    message("u2", "a1b", "user", "kept question"),
    message("a2", "u2", "assistant", "kept answer"),
  ]);
  const started = await start(cwd, sessionPath);
  expect(started).toBe(sessionId);
  const reader = await client();
  reader.send({ type: "attach", sessionId });
  const attached = await until(() => reader.frames.find((frame) => frame.type === "attached"));
  // The attach stored the tool image, the reads below fill the summary, search and entries caches
  const hash = sha256(image);
  expect(await mediaBytes(hash)).toBeDefined();
  const listed = async () => (await (await api(`/sessions?cwd=${encodeURIComponent(cwd)}`)).json()).sessions;
  expect((await listed())[0].firstMessage).toContain("zebraprompt");
  expect((await (await api("/search?q=zebraanswer")).json()).hits).toHaveLength(1);
  expect(JSON.stringify(await (await api(`/sessions/${sessionId}/entries?turns=5`)).json())).toContain("zebratool");

  const headerLine = readFileSync(sessionPath, "utf8").split("\n")[0];
  const response = await deleteRequest(sessionId, "u1", { through: "turn" });
  expect(response.status).toBe(200);
  expect(await response.json()).toEqual({ ok: true, removedEntries: 4, removedMedia: [hash] });

  const text = readFileSync(sessionPath, "utf8");
  expect(text).not.toMatch(/zebra|secret-clip/);
  expect(text).not.toContain(image);
  expect(existsSync(join(cwd, paste))).toBe(false);
  expect(await mediaBytes(hash)).toBeUndefined();
  expect(text.split("\n")[0]).toBe(headerLine);
  expect((await listed())[0].firstMessage).toBe("kept question");
  expect((await (await api("/search?q=zebraanswer")).json()).hits).toEqual([]);
  expect((await (await api("/search?q=kept")).json()).hits).toHaveLength(1);
  expect(JSON.stringify(await (await api(`/sessions/${sessionId}/entries?turns=5`)).json())).not.toContain("zebra");

  // The attached client is carried over: no exit, a replace resync from the new child
  const resync = await until(() => reader.frames.find((frame) => frame.type === "resync"));
  expect(reader.frames.some((frame) => frame.type === "exit")).toBe(false);
  expect(resync.mode).toBe("replace");
  expect(resync.streamId).not.toBe(attached.streamId);
  expect(ids(resync.entries)).toEqual(["u2", "a2"]);
  expect(JSON.stringify(resync)).not.toContain("zebra");

  // The new child runs on the rewritten file and answers prompts
  reader.send({ type: "command", sessionId, command: { type: "prompt", message: "again" } });
  await until(() => reader.frames.find((frame) => frame.type === "pi" && frame.record.type === "agent_settled"));
  const lines = readFileSync(sessionPath, "utf8").trim().split("\n").map((line) => JSON.parse(line));
  expect(lines.slice(1, 3).map((entry) => entry.id)).toEqual(["u2", "a2"]);
  expect(lines[1].parentId).toBeNull();
  expect(lines.at(-1).message.content[0].text).toBe("echo: again");
});

test("deleting through the end keeps what came before and the child continues from there", async () => {
  const cwd = workdir("through-end");
  const { sessionId, sessionPath } = writeSessionFile(cwd, [
    message("u1", null, "user", "first"),
    message("a1", "u1", "assistant", "one"),
    message("u2", "a1", "user", "second"),
    message("a2", "u2", "assistant", "two"),
    message("u3", "a2", "user", "third"),
    message("a3", "u3", "assistant", "three"),
  ]);
  await start(cwd, sessionPath);
  const response = await deleteRequest(sessionId, "u2", { through: "end" });
  expect(await response.json()).toEqual({ ok: true, removedEntries: 4, removedMedia: [] });
  const prompted = await api(`/sessions/${sessionId}/prompt`, { method: "POST", body: JSON.stringify({ message: "edited" }) });
  expect(await prompted.json()).toEqual({ disposition: "started" });
  await until(() => (readFileSync(sessionPath, "utf8").includes("echo: edited") ? true : undefined));
  const entries = readFileSync(sessionPath, "utf8").trim().split("\n").slice(1).map((line) => JSON.parse(line));
  expect(entries.map((entry) => entry.message.content[0]?.text ?? entry.message.content)).toEqual(["first", "one", "edited", "echo: edited"]);
  expect(entries[2].parentId).toBe("a1");
});

test("a fork in the same cwd keeps the image and the paste, other cwds are not asked", async () => {
  const cwd = workdir("shared");
  const paste = ".pi/pastes/20261119-100000-shared.mp4";
  mkdirSync(join(cwd, ".pi", "pastes"), { recursive: true });
  writeFileSync(join(cwd, paste), "video bytes");
  const image = png("shared screenshot");
  const turn = [
    message("u1", null, "user", `see @${paste}`),
    message("a1", "u1", "assistant", "looking"),
    toolResult("t1", "a1", "shot", image),
    message("u2", "t1", "user", "next"),
  ];
  const { sessionId } = writeSessionFile(cwd, turn);
  // The fork sits in the archive root: archived sessions of the cwd count too
  writeSessionFile(cwd, turn, join(agentDir, "sessions-archive"));
  const hash = sha256(image);
  await api(`/sessions/${sessionId}/entries`); // stores the image
  expect(await mediaBytes(hash)).toBeDefined();
  const response = await deleteRequest(sessionId, "u1", { through: "turn" });
  expect(await response.json()).toEqual({ ok: true, removedEntries: 3, removedMedia: [] });
  expect(await mediaBytes(hash)).toBeDefined();
  expect(existsSync(join(cwd, paste))).toBe(true);
  forgetMedia([hash]);
});

test("refusals: unknown session or entry, no prompt, bad body, factory, running, dialog, compacted, locked", async () => {
  const cwd = workdir("refusals");
  const { sessionId, sessionPath } = writeSessionFile(cwd, [
    message("u1", null, "user", "first"),
    message("a1", "u1", "assistant", "one"),
    message("u2", "a1", "user", "second"),
    message("a2", "u2", "assistant", "two"),
    { type: "compaction", id: "c1", parentId: "a2", timestamp: new Date().toISOString(), summary: "s", firstKeptEntryId: "u2", tokensBefore: 1 },
    message("u3", "c1", "user", "third"),
  ]);
  const before = readFileSync(sessionPath, "utf8");
  const answer = async (response: Promise<Response>) => {
    const settled = await response;
    return [settled.status, await settled.json()];
  };
  expect(await answer(deleteRequest(crypto.randomUUID(), "u1", { through: "turn" }))).toEqual([404, { error: "not_found" }]);
  expect(await answer(deleteRequest(sessionId, "nope", { through: "turn" }))).toEqual([404, { error: "unknown" }]);
  expect(await answer(deleteRequest(sessionId, "a1", { through: "turn" }))).toEqual([400, { error: "not_prompt" }]);
  expect(await answer(deleteRequest(sessionId, "u1", { through: "all" }))).toEqual([400, { error: "invalid_body" }]);
  expect(await answer(deleteRequest(sessionId, "u1", {}))).toEqual([400, { error: "invalid_body" }]);
  // The compaction keeps u2 on, so it summarized u1
  expect(await answer(deleteRequest(sessionId, "u1", { through: "turn" }))).toEqual([409, { error: "compacted" }]);

  const holder = Bun.spawn(["flock", "--nonblock", "--no-fork", sessionPath, "sleep", "30"]);
  try {
    await Bun.sleep(200);
    expect(await answer(deleteRequest(sessionId, "u3", { through: "turn" }))).toEqual([409, { error: "locked" }]);
  } finally {
    holder.kill();
    await holder.exited;
  }
  expect(readFileSync(sessionPath, "utf8")).toBe(before);

  const factory = writeSessionFile(cwd, [
    message("u1", null, "user", "task"),
    { type: "custom", customType: "factory-state", id: "f1", parentId: "u1", timestamp: new Date().toISOString(), data: { state: "running", round: 1, reason: "" } },
  ], join(agentDir, "factory", "sessions"));
  expect(await answer(deleteRequest(factory.sessionId, "u1", { through: "turn" }))).toEqual([400, { error: "factory" }]);

  const live = writeSessionFile(workdir("running"), [message("u1", null, "user", "first"), message("a1", "u1", "assistant", "one")]);
  await start(workdir("running"), live.sessionPath);
  const driver = await client();
  driver.send({ type: "attach", sessionId: live.sessionId });
  await until(() => driver.frames.find((frame) => frame.type === "attached"));
  driver.send({ type: "command", sessionId: live.sessionId, command: { type: "prompt", message: "hold" } });
  await until(() => driver.frames.find((frame) => frame.type === "pi" && frame.record.type === "message_update"));
  expect(await answer(deleteRequest(live.sessionId, "u1", { through: "turn" }))).toEqual([409, { error: "running" }]);
  driver.send({ type: "command", sessionId: live.sessionId, command: { type: "abort" } });
  await until(() => driver.frames.find((frame) => frame.type === "pi" && frame.record.type === "agent_settled"));
  driver.send({ type: "command", sessionId: live.sessionId, command: { type: "prompt", message: "dialog" } });
  await until(() => driver.frames.find((frame) => frame.type === "pi" && frame.record.method === "select"));
  expect(await answer(deleteRequest(live.sessionId, "u1", { through: "turn" }))).toEqual([409, { error: "running" }]);
});

test("a file changed between read and rename is left as the other writer left it", async () => {
  const cwd = workdir("changed");
  const { sessionPath } = writeSessionFile(cwd, [message("u1", null, "user", "first"), message("a1", "u1", "assistant", "one")]);
  const appended = `${JSON.stringify(message("u2", "a1", "user", "from the TUI"))}\n`;
  const result = await deleteTurn({ sessionPath, entryId: "u1", through: "turn" }, async () => {
    writeFileSync(sessionPath, appended, { flag: "a" });
  });
  expect(result).toEqual({ ok: false, error: "changed" });
  expect(readFileSync(sessionPath, "utf8").endsWith(appended)).toBe(true);
  expect(readFileSync(sessionPath, "utf8")).toContain('"first"');
});
