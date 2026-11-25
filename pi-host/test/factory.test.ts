/** Factory routes against a stand-in `factory` CLI and core in a scratch dir (never the installed package). */
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { chmod, mkdtemp, rm, utimes, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { handleFactory } from "../src/factory.ts";

/**
 * The stand-in `core.ts`: two actions, one per error path, a refusal with `detail`, the runner environment it was
 * given, and a throw that is no `FactoryError`.
 */
const STUB_CORE = `
export class FactoryError extends Error {
  constructor(code, message, detail) {
    super(message);
    this.code = code;
    if (detail !== undefined) this.detail = detail;
  }
}
export const ACTIONS = {
  "repos.list": { summary: "The registered repositories", params: { type: "object", required: [], properties: {} }, mutates: false },
};
export async function callAction(name, params, options) {
  if (name === "repos.list") return { repos: ["/git/repo"], params: params ?? null };
  if (name === "runner.env") return options.env;
  if (name === "remove unmerged") throw new FactoryError("refused", "task 6f1c has unmerged work", { force: true });
  if (name === "tasks.remove") return undefined;
  if (name === "crash") throw new Error("disk full");
  const [code, ...words] = name.split(" ");
  if (code === "unknown") throw new FactoryError("invalid", \`unknown action "\${name}"\`);
  throw new FactoryError(code, words.join(" "));
}
`;

let scratch: string;
let core: string;
let cli: string;
const savedCli = process.env.AGENT_FACTORY_CLI;

/** Make the stand-in CLI run `body` (a shell script) and count its runs in `<scratch>/cli-runs`. */
async function writeCli(body: string): Promise<void> {
  await writeFile(cli, `#!/bin/sh\necho run >> "${join(scratch, "cli-runs")}"\n${body}\n`);
  await chmod(cli, 0o755);
}

async function cliRuns(): Promise<number> {
  const runs = Bun.file(join(scratch, "cli-runs"));
  return (await runs.exists()) ? (await runs.text()).split("\n").filter(Boolean).length : 0;
}

beforeAll(async () => {
  scratch = await mkdtemp(join(tmpdir(), "pi-host-factory-"));
  core = join(scratch, "agent-factory", "src", "core.ts");
  cli = join(scratch, "factory");
  process.env.AGENT_FACTORY_CLI = cli;
});

afterAll(async () => {
  if (savedCli === undefined) delete process.env.AGENT_FACTORY_CLI;
  else process.env.AGENT_FACTORY_CLI = savedCli;
  await rm(scratch, { recursive: true, force: true });
});

/** Call the factory router and return status and JSON body. */
async function call(method: string, path: string, body?: string): Promise<{ status: number; json: Record<string, unknown> }> {
  const url = new URL(`http://pi-host${path}`);
  const response = await handleFactory(new Request(url, { method, ...(body === undefined ? {} : { body }) }), url);
  if (!response) throw new Error(`factory did not handle ${method} ${path}`);
  return { status: response.status, json: (await response.json()) as Record<string, unknown> };
}

function callAction(action: string, params?: unknown): Promise<{ status: number; json: Record<string, unknown> }> {
  return call("POST", "/factory/call", JSON.stringify({ action, params }));
}

describe("factory routes", () => {
  /** Both routes answer 503 `factory_plugin_missing` naming the package's repository and `reason`. */
  async function expectMissing(reason: string): Promise<void> {
    for (const answer of [await call("GET", "/factory/actions"), await callAction("repos.list")]) {
      expect(answer.status).toBe(503);
      expect(answer.json.error).toBe("factory_plugin_missing");
      expect(answer.json.message).toContain("https://github.com/patrickab/agent-factory");
      expect(answer.json.message).toContain(reason);
    }
  }

  test("without the factory CLI both routes answer 503 naming the package's source", async () => {
    await expectMissing(`posix_spawn '${cli}'`);
  });

  test("a failing CLI is a missing factory", async () => {
    await writeCli('echo "broken install" >&2; exit 3');
    await expectMissing("exited with 3: broken install");
  });

  test("a CLI naming a core that does not exist is a missing factory", async () => {
    await writeCli(`echo "${core}"`);
    await expectMissing(core);
  });

  test("call and actions go to the core the CLI names, its errors map to their status", async () => {
    await Bun.write(core, STUB_CORE);

    const actions = await call("GET", "/factory/actions");
    expect(actions.status).toBe(200);
    expect(Object.keys(actions.json.actions as object)).toEqual(["repos.list"]);

    expect(await callAction("repos.list", { repo: "/x" })).toEqual({ status: 200, json: { result: { repos: ["/git/repo"], params: { repo: "/x" } } } });
    expect(await callAction("tasks.remove")).toEqual({ status: 200, json: { result: null } });
    const statuses = { invalid: 400, not_found: 404, busy: 409, refused: 409, failed: 500 };
    for (const [code, status] of Object.entries(statuses)) {
      expect(await callAction(`${code} it went wrong`)).toEqual({ status, json: { error: code, message: "it went wrong" } });
    }
    expect(await callAction("unknown")).toEqual({ status: 400, json: { error: "invalid", message: 'unknown action "unknown"' } });
    expect(await callAction("crash")).toEqual({ status: 500, json: { error: "failed", message: "disk full" } });
    expect(await callAction("remove unmerged")).toEqual({
      status: 409,
      json: { error: "refused", message: "task 6f1c has unmerged work", detail: { force: true } },
    });
  });

  test("the core's path is asked once and kept across calls", async () => {
    const runs = await cliRuns();
    await writeCli("exit 1");
    expect((await call("GET", "/factory/actions")).status).toBe(200);
    expect((await callAction("repos.list")).status).toBe(200);
    expect(await cliRuns()).toBe(runs);
  });

  test("a changed core.ts is imported anew", async () => {
    await writeFile(core, STUB_CORE.replace('"repos.list": {', '"repos.add": {'));
    const later = new Date(Date.now() + 10_000);
    await utimes(core, later, later);

    expect(Object.keys((await call("GET", "/factory/actions")).json.actions as object)).toEqual(["repos.add"]);
  });

  test("a body without an action is invalid, a wrong method is refused, other paths are not the factory's", async () => {
    expect((await call("POST", "/factory/call", "not json")).json.error).toBe("invalid");
    expect((await call("POST", "/factory/call", JSON.stringify({ params: {} }))).status).toBe(400);
    expect((await call("GET", "/factory/call")).status).toBe(405);
    expect((await call("POST", "/factory/actions")).status).toBe(405);
    const url = new URL("http://pi-host/factory/draft");
    expect(await handleFactory(new Request(url), url)).toBeUndefined();
  });
});
