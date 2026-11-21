import { describe, expect, test } from "bun:test";
import { applyEntry, approvedFrom, followUpMessages, gateData, mergeFailedPrompt, parseStartPayload, rebuild, RETRY_PROMPT, roundsUsed, wasInterrupted, type Finding, type Settings, type Task } from "./state.ts";

const task: Task = {
  title: "Add notes",
  request: "Create notes.txt",
  settings: { branch: "main", merge: "auto", maxRounds: 3, review: true, models: {} },
  worktree: false,
  branch: "main",
  mainBranch: "main",
  baseSha: "abc",
  baseTree: null,
};

const custom = (customType: string, data: unknown) => ({ type: "custom", customType, data });
const finding = (severity: Finding["severity"]): Finding => ({ file: "a.ts", line: 1, severity, text: "x" });

describe("rebuild", () => {
  test("a branch without a factory-task entry holds no task", () => {
    expect(rebuild([{ type: "message" }, custom("other", {}), custom("factory-state", { state: "done", round: 1, reason: "" })])).toBeUndefined();
  });

  test("follows rounds, states and merges in entry order", () => {
    const snapshot = rebuild([
      custom("factory-task", task),
      custom("factory-round", { round: 1, trigger: "start" }),
      custom("factory-state", { state: "running", round: 1, reason: "start" }),
      { type: "message" },
      custom("factory-gate", { round: 1 }),
      custom("factory-round", { round: 2, trigger: "gates" }),
      custom("factory-merge", { ok: false, kind: "commit", commit: null, error: "x", depsSynced: false }),
      custom("factory-merge", { ok: true, kind: "commit", commit: "def", error: null, depsSynced: false }),
      custom("factory-state", { state: "done", round: 2, reason: "committed" }),
    ]);
    expect(snapshot).toEqual({ task, state: "done", round: 2, runFirstRound: 1, reason: "committed", mergedBase: "def", head: "def", roundTree: null, mergeIntent: null, mergeError: null });
  });

  test("after a current-checkout commit the diff base is the tree it left, not the commit", () => {
    const entries = [custom("factory-task", task), custom("factory-merge", { ok: true, kind: "commit", commit: "def", error: null, depsSynced: false, tree: "left" })];
    expect(rebuild(entries)?.mergedBase).toBe("left");
  });

  test("message and retry open a run with a fresh round budget", () => {
    let snapshot = rebuild([custom("factory-task", task), custom("factory-round", { round: 1, trigger: "start" }), custom("factory-round", { round: 2, trigger: "gates" })]);
    expect(roundsUsed(snapshot!)).toBe(2);
    snapshot = applyEntry(snapshot, "factory-round", { round: 3, trigger: "message" });
    expect(roundsUsed(snapshot!)).toBe(1);
    snapshot = applyEntry(snapshot, "factory-round", { round: 4, trigger: "review" });
    snapshot = applyEntry(snapshot, "factory-round", { round: 5, trigger: "retry" });
    expect(snapshot).toMatchObject({ round: 5, runFirstRound: 5 });
  });

  test("a conflict round opens a run on a task at rest, inside a run or a merge it spends the run's budget", () => {
    const run = rebuild([custom("factory-task", task), custom("factory-round", { round: 1, trigger: "start" })])!;
    expect(applyEntry({ ...run, state: "blocked" }, "factory-round", { round: 2, trigger: "conflict" })).toMatchObject({ runFirstRound: 2 });
    expect(applyEntry({ ...run, state: "merging" }, "factory-round", { round: 2, trigger: "conflict" })).toMatchObject({ runFirstRound: 1 });
    expect(applyEntry(run, "factory-round", { round: 2, trigger: "conflict" })).toMatchObject({ runFirstRound: 1 });
  });

  test("running and merging count as interrupted, the other states do not", () => {
    const base = rebuild([custom("factory-task", task)])!;
    expect(wasInterrupted(base)).toBe(true);
    expect(wasInterrupted({ ...base, state: "merging" })).toBe(true);
    for (const state of ["waiting_approve", "blocked", "done", "cancelled"] as const) expect(wasInterrupted({ ...base, state })).toBe(false);
  });
});

describe("parseStartPayload", () => {
  const settings: Settings = { branch: "release", merge: "review", maxRounds: 2, review: false, models: { reviewer: "anthropic/claude-haiku-5-5" } };

  test("returns the payload", () => {
    expect(parseStartPayload(JSON.stringify({ request: "do it", title: "T", settings }))).toEqual({ request: "do it", title: "T", settings });
  });

  test.each([
    ["{", "payload is not JSON"],
    ["[]", "payload is not an object"],
    [JSON.stringify({ request: "", title: "T", settings }), "request must be a non-empty string"],
    [JSON.stringify({ request: "r", settings }), "title must be a non-empty string"],
    [JSON.stringify({ request: "r", title: "T" }), "settings must be an object"],
    [JSON.stringify({ request: "r", title: "T", settings: { ...settings, branch: undefined } }), "settings.branch must be a non-empty string"],
    [JSON.stringify({ request: "r", title: "T", settings: { ...settings, branch: "" } }), "settings.branch must be a non-empty string"],
    [JSON.stringify({ request: "r", title: "T", settings: { ...settings, merge: "manual" } }), 'settings.merge must be "auto" or "review"'],
    [JSON.stringify({ request: "r", title: "T", settings: { ...settings, maxRounds: 0 } }), "settings.maxRounds must be a positive integer"],
    [JSON.stringify({ request: "r", title: "T", settings: { ...settings, review: "yes" } }), "settings.review must be a boolean"],
    [JSON.stringify({ request: "r", title: "T", settings: { ...settings, models: { coder: "haiku" } } }), 'settings.models.coder must be "provider/model"'],
  ])("rejects %s", (text, message) => {
    expect(() => parseStartPayload(text)).toThrow(message);
  });
});

describe("derived values", () => {
  test("approved needs status ok and no blocker or major finding", () => {
    expect(approvedFrom("ok", [])).toBe(true);
    expect(approvedFrom("ok", [finding("minor")])).toBe(true);
    expect(approvedFrom("ok", [finding("minor"), finding("major")])).toBe(false);
    expect(approvedFrom("ok", [finding("blocker")])).toBe(false);
    expect(approvedFrom("malformed", [])).toBe(false);
    expect(approvedFrom("no_diff", [])).toBe(false);
  });

  test("a killed gate keeps a null exit code and only the tail of its output", () => {
    const result = { exitCode: null, killed: true, timedOut: false, durationMs: 12, stdout: "", stdoutTruncated: false, output: `${"x".repeat(5000)}end` };
    const data = gateData(2, "gates", ["sh", "gate.sh"], result);
    expect(data).toMatchObject({ round: 2, phase: "gates", argv: ["sh", "gate.sh"], exitCode: null, killed: true, durationMs: 12 });
    expect(data.outputTail).toHaveLength(4000);
    expect(data.outputTail.endsWith("end")).toBe(true);
  });
});

describe("followUpMessages", () => {
  const user = (content: unknown) => ({ type: "message", message: { role: "user", content } });

  test("the user's messages after the request in order, without commands, retry prompts and feedback", () => {
    const branch = [
      user("before the task"),
      custom("factory-task", task),
      custom("factory-round", { round: 1, trigger: "start" }),
      user("Create notes.txt"),
      { type: "message", message: { role: "assistant", content: [{ type: "text", text: "done" }] } },
      custom("factory-state", { state: "waiting_approve", round: 1, reason: "passed" }),
      custom("factory-round", { round: 2, trigger: "message" }),
      user([{ type: "text", text: "also add a second line: follow-up" }]),
      { type: "custom_message", customType: "factory-feedback", content: "Round 2: the review found problems." },
      user("/factory approve"),
      user(RETRY_PROMPT),
      user(mergeFailedPrompt("rebase onto main failed")),
      user("and a third line"),
    ];
    expect(followUpMessages(branch)).toEqual(["also add a second line: follow-up", "and a third line"]);
  });

  test("a branch without a task, or with only its request, has none", () => {
    expect(followUpMessages([user("hello")])).toEqual([]);
    expect(followUpMessages([custom("factory-task", task), user("Create notes.txt")])).toEqual([]);
  });
});
