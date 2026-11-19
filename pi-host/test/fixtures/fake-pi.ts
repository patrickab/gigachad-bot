/**
 * Fake `pi` binary for supervisor tests: `--version` and a scripted subset of `--mode rpc`, no model calls.
 *
 * - Consumes: `--session <path>`, `--session-dir <dir>`, `PI_CODING_AGENT_DIR`, `PI_CODING_AGENT_SESSION_DIR`, JSONL
 *   commands on stdin
 * - Provides: Pi-shaped responses and events. Prompt texts script the run: `hold` streams one delta and waits for
 *   `abort`, `dialog` opens a select dialog, `status <key> [text]` sets or clears a status, `compact` appends a
 *   compaction entry, `goto <entryId>` moves the leaf there (a branch switch), `orphan` starts a `sleep` that
 *   inherits stdout and stderr and answers its pid, anything else echoes. `bash` streams `bash_execution_update`s.
 *   `get_entries` honours `since` like Pi. The session file is written lazily at the first assistant `message_end`.
 *   `get_state` also answers `argv`, the arguments the fixture was started with, so tests can assert profile flags
 */
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";

type Command = { id?: string; type: string; [field: string]: unknown };
type Entry = { type: string; id: string; parentId: string | null; timestamp: string; [field: string]: unknown };

const args = process.argv.slice(2);
if (args.includes("--version")) {
  console.log("1.5.2");
  process.exit(0);
}

const cwd = process.cwd();
const sessionArg = args.indexOf("--session");
const sessionDirArg = args.indexOf("--session-dir");
const agentDir = process.env.PI_CODING_AGENT_DIR ?? join(homedir(), ".pi", "agent");
const sessionDir =
  (sessionDirArg >= 0 ? args[sessionDirArg + 1] : undefined) ??
  process.env.PI_CODING_AGENT_SESSION_DIR ??
  join(agentDir, "sessions", `--${resolve(cwd).replace(/^\//, "").replace(/[/\\:]/g, "-")}--`);

let sessionId = crypto.randomUUID();
let sessionFile = join(sessionDir, `${new Date().toISOString().replace(/[:.]/g, "-")}_${sessionId}.jsonl`);
const entries: Entry[] = [];
let leafId: string | null = null;
if (sessionArg >= 0) {
  sessionFile = resolve(args[sessionArg + 1]!);
  const [header, ...lines] = readFileSync(sessionFile, "utf8").split("\n").filter(Boolean);
  sessionId = JSON.parse(header!).id;
  entries.push(...lines.map((line) => JSON.parse(line)));
  leafId = entries.at(-1)?.id ?? null;
}

let model = { provider: "faux", id: "faux-1" };
let thinkingLevel = "medium";
let streaming = false;
let release: (() => void) | undefined;
let openDialog: string | undefined;

function emit(record: object): void {
  process.stdout.write(`${JSON.stringify(record)}\n`);
}

function respond(command: Command, fields: object = { success: true }): void {
  emit({ ...(command.id ? { id: command.id } : {}), type: "response", command: command.type, ...fields });
}

/** Append an entry; the file appears with the first assistant message, like Pi. */
function persist(entry: Entry, isAssistant: boolean): void {
  entries.push(entry);
  if (!existsSync(sessionFile)) {
    if (!isAssistant) return;
    mkdirSync(dirname(sessionFile), { recursive: true });
    const header = { type: "session", version: 3, id: sessionId, timestamp: new Date().toISOString(), cwd };
    writeFileSync(sessionFile, [header, ...entries].map((line) => `${JSON.stringify(line)}\n`).join(""));
    return;
  }
  appendFileSync(sessionFile, `${JSON.stringify(entry)}\n`);
}

function message(role: string, text: string, extra: object = {}): object {
  return { role, content: role === "user" ? text : [{ type: "text", text }], timestamp: Date.now(), ...extra };
}

function appendEntry(fields: { type: string; [field: string]: unknown }, isAssistant: boolean): Entry {
  const entry = { ...fields, id: crypto.randomUUID().slice(0, 8), parentId: leafId, timestamp: new Date().toISOString() };
  leafId = entry.id;
  persist(entry, isAssistant);
  return entry;
}

function endMessage(msg: object, isAssistant: boolean): void {
  emit({ type: "message_end", message: msg });
  appendEntry({ type: "message", message: msg }, isAssistant);
}

async function run(text: string): Promise<void> {
  streaming = true;
  emit({ type: "agent_start" });
  emit({ type: "turn_start" });
  const user = message("user", text);
  emit({ type: "message_start", message: user });
  endMessage(user, false);
  emit({ type: "message_start", message: message("assistant", "", { stopReason: "pending" }) });
  const aborted = text === "hold";
  const reply = aborted ? "partial " : `echo: ${text}`;
  emit({ type: "message_update", assistantMessageEvent: { type: "text_delta", contentIndex: 0, delta: reply } });
  if (aborted) {
    const held = Promise.withResolvers<void>();
    release = held.resolve;
    await held.promise;
  }
  const assistant = message("assistant", reply, { stopReason: aborted ? "aborted" : "stop" });
  endMessage(assistant, true);
  emit({ type: "turn_end", message: assistant, toolResults: [] });
  emit({ type: "agent_end", messages: [user, assistant], willRetry: false });
  streaming = false;
  emit({ type: "agent_settled", aborted });
}

function handle(command: Command): void {
  switch (command.type) {
    case "get_state":
      return respond(command, {
        success: true,
        data: {
          model,
          thinkingLevel,
          isStreaming: streaming,
          isCompacting: false,
          sessionFile,
          sessionId,
          messageCount: entries.length,
          pendingMessageCount: 0,
          argv: args,
        },
      });
    case "get_entries": {
      if (command.since === undefined) return respond(command, { success: true, data: { entries, leafId } });
      const since = entries.findIndex((entry) => entry.id === command.since);
      if (since < 0) return respond(command, { success: false, error: `Entry not found: ${command.since}` });
      return respond(command, { success: true, data: { entries: entries.slice(since + 1), leafId } });
    }
    case "bash": {
      const count = Number(command.updates ?? 3);
      for (let index = 0; index < count; index++) emit({ type: "bash_execution_update", output: `line ${index}\n` });
      return respond(command, { success: true, data: { output: "done", exitCode: 0 } });
    }
    case "set_model":
      if (command.provider === "bad") return respond(command, { success: false, error: "Model not found: bad" });
      model = { provider: String(command.provider), id: String(command.modelId) };
      return respond(command, { success: true, data: model });
    case "set_thinking_level":
      thinkingLevel = String(command.level);
      return respond(command);
    case "abort":
      release?.();
      release = undefined;
      return respond(command);
    case "extension_ui_response":
      if (command.id === openDialog) {
        openDialog = undefined;
        emit({ type: "extension_ui_request", id: crypto.randomUUID(), method: "notify", message: `chose ${command.value}` });
      }
      return;
    case "prompt": {
      const text = String(command.message);
      if (text === "dialog") {
        openDialog = crypto.randomUUID();
        respond(command, { success: true, data: { disposition: "handled" } });
        return emit({ type: "extension_ui_request", id: openDialog, method: "select", title: "Pick", options: ["a", "b"] });
      }
      if (text.startsWith("status ")) {
        const [, statusKey, statusText] = text.split(" ");
        respond(command, { success: true, data: { disposition: "handled" } });
        return emit({ type: "extension_ui_request", id: crypto.randomUUID(), method: "setStatus", statusKey, statusText });
      }
      if (text === "compact") {
        respond(command, { success: true, data: { disposition: "handled" } });
        emit({ type: "compaction_start", reason: "manual" });
        const compaction = appendEntry({ type: "compaction", summary: "summary", firstKeptEntryId: leafId, tokensBefore: 10 }, false);
        return emit({ type: "compaction_end", reason: "manual", result: { summary: "summary" }, entryId: compaction.id });
      }
      if (text.startsWith("goto ")) {
        leafId = text.slice("goto ".length);
        return respond(command, { success: true, data: { disposition: "handled" } });
      }
      if (text === "orphan") {
        const orphan = Bun.spawn(["sleep", "30"], { stdin: "ignore", stdout: "inherit", stderr: "inherit" });
        orphan.unref();
        return respond(command, { success: true, data: { disposition: "handled", pid: orphan.pid } });
      }
      if (streaming) return respond(command, { success: false, error: "Agent is streaming" });
      respond(command, { success: true, data: { disposition: "started" } });
      void run(text);
      return;
    }
    default:
      return respond(command);
  }
}

process.on("SIGTERM", () => process.exit(0));
let rest = "";
for await (const chunk of Bun.stdin.stream()) {
  const text = rest + new TextDecoder().decode(chunk);
  const lines = text.split("\n");
  rest = lines.pop() ?? "";
  for (const line of lines) if (line.trim()) handle(JSON.parse(line));
}
process.exit(0);
