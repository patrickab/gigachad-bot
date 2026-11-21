import { afterAll, describe, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { parseFindings, readReviewerOutput, reviewPatch, runReview } from "./review.ts";
import type { CommandOptions, CommandResult } from "./run.ts";
import type { Finding, Task } from "./state.ts";

const line = (event: unknown) => JSON.stringify(event);
const verdictEnd = (findings: unknown, isError = false) =>
  line({ type: "tool_execution_end", toolName: "submit_verdict", isError, result: { details: { findings } } });
const assistantEnd = (usage: unknown) => line({ type: "message_end", message: { role: "assistant", usage } });

describe("reviewPatch", () => {
  test("lists the changed files before a patch within the cap", () => {
    expect(reviewPatch({ paths: ["src/a.ts", "bun.lock"], patch: "diff --git a b\n" }, 100)).toBe("Changed files (2):\n- src/a.ts\n- bun.lock\n\ndiff --git a b\n");
  });

  test("cuts a larger patch, says how much is missing and keeps the whole file list", () => {
    const capped = reviewPatch({ paths: ["src/a.ts", "tests/a.test.ts", "bun.lock"], patch: "x".repeat(150) }, 100);
    expect(capped.startsWith(`Changed files (3):\n- src/a.ts\n- tests/a.test.ts\n- bun.lock\n\n${"x".repeat(100)}\n`)).toBe(true);
    expect(capped).toContain("[patch cut here: 50 more characters. Read the files listed at the top that the patch does not show.]");
  });

  test("a huge file list is cut with a count", () => {
    const paths = Array.from({ length: 502 }, (_value, index) => `f${index}`);
    expect(reviewPatch({ paths, patch: "p" }, 10)).toContain("- f499\n- and 2 more\n\np");
  });
});

describe("parseFindings", () => {
  test("accepts findings, a missing line becomes null", () => {
    expect(parseFindings([{ file: "a.ts", severity: "minor", text: "typo" }, { file: "b.ts", line: 3, severity: "blocker", text: "crash" }])).toEqual([
      { file: "a.ts", line: null, severity: "minor", text: "typo" },
      { file: "b.ts", line: 3, severity: "blocker", text: "crash" },
    ]);
  });

  test.each([
    [{ findings: [] }, "findings is not an array"],
    [["x"], "findings[0] is not an object"],
    [[{ file: "", line: 1, severity: "minor", text: "t" }], "findings[0].file is not a string"],
    [[{ file: "a", line: 1.5, severity: "minor", text: "t" }], "findings[0].line is not an integer or null"],
    [[{ file: "a", line: 1, severity: "critical", text: "t" }], "findings[0].severity is not one of blocker, major, minor"],
    [[{ file: "a", line: 1, severity: "minor" }], "findings[0].text is not a string"],
  ])("rejects %j", (value, problem) => {
    expect(parseFindings(value)).toBe(problem);
  });
});

describe("readReviewerOutput", () => {
  const finding: Finding = { file: "src/cart.js", line: 5, severity: "blocker", text: "off by one" };

  test("reads session id, the verdict and the summed usage", () => {
    const stdout = [
      line({ type: "session", id: "abc-123" }),
      assistantEnd({ input: 4, output: 100, cacheRead: 10, cacheWrite: 50, cost: { total: 0.001 } }),
      line({ type: "message_end", message: { role: "user", usage: { input: 999 } } }),
      assistantEnd({ input: 6, output: 20, cacheRead: 60, cacheWrite: 0, cost: { total: 0.002 } }),
      verdictEnd([finding]),
    ].join("\n");
    expect(readReviewerOutput(stdout)).toEqual({
      findings: [finding],
      sessionId: "abc-123",
      usage: { input: 10, output: 120, cacheRead: 70, cacheWrite: 50, cost: 0.003 },
    });
  });

  test("a rejected call is no verdict, the last valid call wins", () => {
    expect(readReviewerOutput(verdictEnd([finding], true)).findings).toBeNull();
    expect(readReviewerOutput([verdictEnd([finding]), verdictEnd([])].join("\n")).findings).toEqual([]);
    expect(readReviewerOutput([verdictEnd([]), verdictEnd("nonsense")].join("\n")).findings).toEqual([]);
  });

  test("no call and broken lines give no verdict", () => {
    const output = readReviewerOutput(`${line({ type: "session", id: "s1" })}\n{"type":"tool_execution_end","toolNa`);
    expect(output.findings).toBeNull();
    expect(output.sessionId).toBe("s1");
  });
});

describe("runReview", () => {
  const reviewsDir = mkdtempSync(join(tmpdir(), "factory-reviews-test-"));
  afterAll(() => rmSync(reviewsDir, { recursive: true, force: true }));
  const task: Task = {
    title: "Fix the cart",
    request: "The total is off by one.",
    settings: { branch: "main", merge: "auto", maxRounds: 3, review: true, models: {} },
    worktree: false,
    branch: "main",
    mainBranch: "main",
    baseSha: "abc",
    baseTree: "tree",
  };
  const blocker: Finding = { file: "src/cart.js", line: 5, severity: "blocker", text: "off by one" };
  const child = (overrides: Partial<CommandResult> = {}): CommandResult => ({
    exitCode: 0,
    killed: false,
    timedOut: false,
    durationMs: 42,
    stdout: [line({ type: "session", id: "abc-123" }), verdictEnd([blocker])].join("\n"),
    stdoutTruncated: false,
    output: "",
    ...overrides,
  });
  const review = (result: CommandResult, calls: { argv: string[]; options: CommandOptions }[] = [], patch = "diff --git a/src/cart.js b/src/cart.js\n", followUps: string[] = []) =>
    runReview({
      exec: async (argv, options) => {
        calls.push({ argv, options });
        return result;
      },
      root: "/repo",
      round: 2,
      task,
      followUps,
      diff: { base: "tree", tree: "now", paths: ["src/cart.js"], patch },
      model: "anthropic/claude-haiku-5-5",
      rolePrompt: "Be strict.",
      reviewsDir,
      cancelled: () => false,
    });

  test("starts a read-only child that gets the patch and the request as files, and maps its verdict", async () => {
    writeFileSync(join(reviewsDir, "2024_abc-123.jsonl"), "");
    const calls: { argv: string[]; options: CommandOptions }[] = [];
    const data = await review(child(), calls);

    expect(data).toMatchObject({ round: 2, status: "ok", approved: false, findings: [blocker], model: "anthropic/claude-haiku-5-5", wallMs: 42 });
    expect(data.sessionFile).toBe(join(reviewsDir, "2024_abc-123.jsonl"));
    const [{ argv, options }] = calls;
    expect(argv.slice(0, 6)).toEqual(["pi", "--mode", "json", "-p", "--tools", "read,grep,find,ls,submit_verdict"]);
    expect(argv.slice(6, 9)).toEqual(["-ne", "-e", join(import.meta.dirname, "reviewer-tool.ts")]);
    expect(argv.join(" ")).toContain(`--session-dir ${reviewsDir} --model anthropic/claude-haiku-5-5 --append-system-prompt Be strict.`);
    expect(argv.at(-1)).toContain(join(reviewsDir, "round-2.patch"));
    expect(argv.at(-1)).not.toContain(task.request);
    expect(options).toMatchObject({ cwd: "/repo", timeoutMs: 300_000 });
    expect(options.maxStdoutChars).toBeGreaterThan(1_000_000);
    expect(readFileSync(join(reviewsDir, "round-2.patch"), "utf8")).toBe("Changed files (1):\n- src/cart.js\n\ndiff --git a/src/cart.js b/src/cart.js\n");
    expect(readFileSync(join(reviewsDir, "round-2.request.md"), "utf8")).toBe("# Fix the cart\n\nThe total is off by one.\n");
  });

  test.each([
    ["timeout", { timedOut: true, killed: true, exitCode: null }],
    ["aborted", { killed: true, exitCode: null }],
    ["failed", { exitCode: 1 }],
    ["malformed", { stdout: line({ type: "session", id: "abc-123" }) }],
  ] as const)("a child that ends %s yields no findings and no approval", async (status, result) => {
    expect(await review(child(result))).toMatchObject({ status, approved: false, findings: [] });
  });

  test("an empty patch is no_diff and starts no child", async () => {
    const calls: { argv: string[]; options: CommandOptions }[] = [];
    expect(await review(child(), calls, "\n")).toMatchObject({ status: "no_diff", approved: false, findings: [], sessionFile: null });
    expect(calls).toEqual([]);
  });

  test("the request file holds the user's follow-up messages after the request, and the prompt makes them part of the task", async () => {
    const calls: { argv: string[]; options: CommandOptions }[] = [];
    await review(child(), calls, undefined, ["also add a second line: follow-up", "and a third"]);
    expect(readFileSync(join(reviewsDir, "round-2.request.md"), "utf8")).toBe(
      "# Fix the cart\n\nThe total is off by one.\n\n## Follow-up messages from the user\n\n" +
        "### Message 1\n\nalso add a second line: follow-up\n\n### Message 2\n\nand a third\n",
    );
    expect(calls[0].argv.at(-1)).toContain("Follow-up messages from the user in the task file are part of the task");
  });
});
