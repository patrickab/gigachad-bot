/** The sessions watcher over a scratch agent dir: which factory files it reports (never the deployed agent dir). */
import { afterAll, beforeAll, expect, test } from "bun:test";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DEBOUNCE_MS, watchSessions } from "../src/host/local/watch.ts";

let scratch: string;
const saved = { agentDir: process.env.PI_CODING_AGENT_DIR, sessionDir: process.env.PI_CODING_AGENT_SESSION_DIR };

beforeAll(async () => {
  scratch = await mkdtemp(join(tmpdir(), "pi-host-watch-"));
  process.env.PI_CODING_AGENT_DIR = join(scratch, "agent");
  process.env.PI_CODING_AGENT_SESSION_DIR = join(scratch, "sessions");
  await mkdir(join(scratch, "agent", "factory", "tasks"), { recursive: true });
  await mkdir(join(scratch, "sessions"), { recursive: true });
});

afterAll(async () => {
  for (const [name, value] of [["PI_CODING_AGENT_DIR", saved.agentDir], ["PI_CODING_AGENT_SESSION_DIR", saved.sessionDir]] as const) {
    if (value === undefined) delete process.env[name];
    else process.env[name] = value;
  }
  await rm(scratch, { recursive: true, force: true });
});

test("a factory task's record and runner lock are reported, its log is not", async () => {
  const tasks = join(scratch, "agent", "factory", "tasks");
  const reported: string[] = [];
  const stop = watchSessions((paths) => reported.push(...paths));
  try {
    await Bun.sleep(50);
    await writeFile(join(tasks, "6f1c.lock"), "123\n");
    await writeFile(join(tasks, "6f1c.json"), "{}\n");
    await writeFile(join(tasks, "6f1c.log"), "output\n");
    await Bun.sleep(DEBOUNCE_MS + 300);
    await rm(join(tasks, "6f1c.lock"));
    await Bun.sleep(DEBOUNCE_MS + 300);
  } finally {
    stop();
  }
  expect(reported).toContain(join(tasks, "6f1c.lock"));
  expect(reported).toContain(join(tasks, "6f1c.json"));
  expect(reported).not.toContain(join(tasks, "6f1c.log"));
  expect(reported.filter((path) => path.endsWith(".lock")).length).toBeGreaterThanOrEqual(2);
});
