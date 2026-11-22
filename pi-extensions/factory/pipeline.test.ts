import { describe, expect, test } from "bun:test";
import { DEFAULT_PROTECTED, type FactoryConfig } from "./config.ts";
import { MERGE_GRACE_ROUNDS, runMerge, runRound, type PipelineSteps } from "./pipeline.ts";
import type { CommandResult } from "./run.ts";
import { stateDraft, type Draft, type MergeData, type ReviewData, type ReviewStatus, type Settings, type Snapshot } from "./state.ts";

const config: FactoryConfig = {
  hooks: { after_code: [["fmt"]], after_merge: [] },
  gates: [["lint"], ["test"]],
  merge_gates: [["lint"], ["test"]],
  protected: DEFAULT_PROTECTED,
};

function snapshotOf(settings: Partial<Settings> = {}, round = 1, runFirstRound = 1, worktree = false): Snapshot {
  return {
    task: {
      title: "T",
      request: "R",
      settings: { branch: "main", merge: "auto", maxRounds: 3, review: true, models: {}, ...settings },
      worktree,
      branch: worktree ? "task" : "main",
      mainBranch: "main",
      baseSha: "abc",
      baseTree: "tree",
    },
    state: "running",
    round,
    runFirstRound,
    reason: "start",
    mergedBase: null,
    head: null,
    roundTree: null,
    mergeIntent: null,
    mergeError: null,
  };
}

function commandResult(exitCode: number | null, output = ""): CommandResult {
  return { exitCode, killed: exitCode === null, timedOut: false, durationMs: 5, stdout: output, stdoutTruncated: false, output };
}

function reviewOf(status: ReviewStatus, severities: ("blocker" | "major" | "minor")[] = []): ReviewData {
  const findings = severities.map((severity) => ({ file: "a.ts", line: 1, severity, text: `${severity} problem` }));
  const approved = status === "ok" && severities.every((severity) => severity === "minor");
  return { round: 1, status, approved, findings, model: "m", sessionFile: null, usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, cost: 0 }, wallMs: 1 };
}

const merged: MergeData = { ok: true, kind: "commit", commit: "def", error: null, depsSynced: false, mixed: [], tree: "left" };

/** Steps where everything passes, with the calls recorded in `calls`. */
function fakeSteps(overrides: Partial<PipelineSteps> = {}): PipelineSteps & { calls: string[] } {
  const calls: string[] = [];
  return {
    calls,
    status: () => undefined,
    cancelled: () => false,
    pendingMessages: () => false,
    config: () => config,
    runCommand: async (argv) => {
      calls.push(argv.join(" "));
      return commandResult(0);
    },
    diff: async () => ({ base: "tree", tree: "now", paths: ["src/a.ts"], patch: "diff" }),
    review: async () => {
      calls.push("review");
      return reviewOf("ok");
    },
    merge: async () => {
      calls.push("merge");
      return { gates: [], merge: merged, conflicts: null, failedGate: null };
    },
    openRebase: async () => null,
    ...overrides,
  };
}

const kinds = (entries: Draft[]) => entries.map((entry) => (entry.type === "custom" ? entry.customType : entry.type));
const lastState = (entries: Draft[]) => entries.flatMap((entry) => (entry.type === "custom" && entry.customType === "factory-state" ? [entry.data] : [])).at(-1);

describe("runRound", () => {
  test("a passing auto round runs hooks, gates and review in order and ends merging, the merge itself is the caller's", async () => {
    const steps = fakeSteps();
    const result = await runRound(snapshotOf(), steps);
    expect(steps.calls).toEqual(["fmt", "lint", "test", "review"]);
    expect(kinds(result.entries)).toEqual(["factory-gate", "factory-gate", "factory-gate", "factory-review", "factory-state"]);
    expect(result.entries[0]).toMatchObject({ data: { round: 1, phase: "hooks", argv: ["fmt"], exitCode: 0 } });
    expect(result.entries[1]).toMatchObject({ data: { phase: "gates", argv: ["lint"] } });
    expect(lastState(result.entries)).toEqual({ state: "merging", round: 1, reason: "passed" });
    expect(result.continue).toBe(false);
  });

  test.each(["auto", "review"] as const)("a message queued during a passing %s round opens a message round instead of the merge or the wait", async (merge) => {
    const steps = fakeSteps({ pendingMessages: () => true });
    const result = await runRound(snapshotOf({ merge }), steps);
    expect(kinds(result.entries).slice(-2)).toEqual(["factory-round", "factory-state"]);
    expect(result.entries.at(-2)).toMatchObject({ data: { round: 2, trigger: "message" } });
    expect(lastState(result.entries)).toEqual({ state: "running", round: 2, reason: "message" });
    expect(result.entries.some((entry) => entry.type === "custom" && ["merging", "waiting_approve"].includes((entry.data as { state?: string }).state ?? ""))).toBe(false);
    expect(result.continue).toBe(true);
  });

  test("a message queued during a round that ends blocked opens a message round after the blocked state", async () => {
    const steps = fakeSteps({ pendingMessages: () => true, review: async () => reviewOf("malformed") });
    const result = await runRound(snapshotOf(), steps);
    const states = result.entries.flatMap((entry) => (entry.type === "custom" && entry.customType === "factory-state" ? [(entry.data as { state: string; reason: string }).reason] : []));
    expect(states).toEqual(["review_malformed", "message"]);
    expect(result.continue).toBe(true);
  });

  test("a cancelled round opens no message round", async () => {
    const result = await runRound(snapshotOf(), fakeSteps({ pendingMessages: () => true, cancelled: () => true }));
    expect(lastState(result.entries)).toEqual({ state: "cancelled", round: 1, reason: "cancelled" });
    expect(result.continue).toBe(false);
  });

  test("merge review waits for approve without merging", async () => {
    const steps = fakeSteps();
    const result = await runRound(snapshotOf({ merge: "review" }), steps);
    expect(steps.calls).not.toContain("merge");
    expect(lastState(result.entries)).toEqual({ state: "waiting_approve", round: 1, reason: "passed" });
  });

  test("review off skips the reviewer", async () => {
    const steps = fakeSteps();
    await runRound(snapshotOf({ review: false }), steps);
    expect(steps.calls).toEqual(["fmt", "lint", "test"]);
  });

  test("a failing gate stops the round and gives the coder its output in the next round", async () => {
    const steps = fakeSteps({ runCommand: async (argv) => commandResult(argv[0] === "lint" ? 2 : 0, "E501 line too long") });
    const result = await runRound(snapshotOf(), steps);
    expect(kinds(result.entries)).toEqual(["factory-gate", "factory-gate", "factory-round", "custom_message"]);
    expect(result.entries[2]).toMatchObject({ data: { round: 2, trigger: "gates" } });
    expect(result.entries[3]).toMatchObject({ customType: "factory-feedback", display: true, details: { round: 1, trigger: "gates" } });
    expect((result.entries[3] as { content: string }).content).toContain("`lint` failed (exit 2)");
    expect((result.entries[3] as { content: string }).content).toContain("E501 line too long");
    expect(result.continue).toBe(true);
  });

  test("a gate that exits 75 blocks the task as infra, with no feedback and no round for the coder", async () => {
    const steps = fakeSteps({ runCommand: async (argv) => commandResult(argv[0] === "lint" ? 75 : 0, "noise\neslint is missing\n") });
    const result = await runRound(snapshotOf(), steps);
    expect(kinds(result.entries)).toEqual(["factory-gate", "factory-gate", "factory-state"]);
    expect(lastState(result.entries)).toMatchObject({ state: "blocked", reason: "infra: eslint is missing" });
    expect(result.continue).toBe(false);
  });

  test("a failing hook is a round problem with trigger hooks", async () => {
    const result = await runRound(snapshotOf(), fakeSteps({ runCommand: async () => commandResult(1) }));
    expect(kinds(result.entries)).toEqual(["factory-gate", "factory-round", "custom_message"]);
    expect(result.entries[1]).toMatchObject({ data: { round: 2, trigger: "hooks" } });
  });

  test("the last round of the budget ends blocked, with the feedback and no new round", async () => {
    const result = await runRound(snapshotOf({ maxRounds: 3 }, 3), fakeSteps({ runCommand: async () => commandResult(1) }));
    expect(kinds(result.entries)).toEqual(["factory-gate", "custom_message", "factory-state"]);
    expect(lastState(result.entries)).toEqual({ state: "blocked", round: 3, reason: "max_rounds: hooks" });
    expect(result.continue).toBe(false);
  });

  test("the budget counts from the first round of the run", async () => {
    const result = await runRound(snapshotOf({ maxRounds: 2 }, 4, 4), fakeSteps({ runCommand: async () => commandResult(1) }));
    expect(result.continue).toBe(true);
    expect(result.entries[1]).toMatchObject({ data: { round: 5, trigger: "hooks" } });
  });

  test("a diff that touches a protected path is a round problem and is never reviewed or merged", async () => {
    const steps = fakeSteps({ diff: async () => ({ base: "tree", tree: "now", paths: ["src/a.ts", "factory.yaml", ".factory/agents/coder.md"], patch: "diff" }) });
    const snapshot = snapshotOf();
    const result = await runRound({ ...snapshot, task: { ...snapshot.task, worktree: true, baseTree: null } }, steps);
    expect(steps.calls).toEqual(["fmt", "lint", "test"]);
    expect(result.entries.at(-2)).toMatchObject({ data: { round: 2, trigger: "protected" } });
    expect((result.entries.at(-1) as { content: string }).content).toContain("- factory.yaml\n- .factory/agents/coder.md");
    expect(result.continue).toBe(true);
  });

  test("in the user's checkout a protected path blocks the task with the paths: no model is asked to restore it", async () => {
    const steps = fakeSteps({ diff: async () => ({ base: "tree", tree: "now", paths: ["factory.yaml", "src/a.ts"], patch: "diff" }) });
    const result = await runRound(snapshotOf(), steps);
    expect(result).toEqual({ entries: [...result.entries.slice(0, -1), stateDraft("blocked", 1, "protected: factory.yaml")], continue: false });
    expect(result.entries.some((draft) => draft.type === "custom_message")).toBe(false);
    const snapshot = snapshotOf();
    const worktree = (await runRound({ ...snapshot, task: { ...snapshot.task, worktree: true, baseTree: null } }, steps)).entries.at(-1) as { content: string };
    expect(worktree.content).toContain("Restore them to their original state, then stop.");
  });

  test("blocker and major findings go back to the coder, minor ones do not stop the merge", async () => {
    const rejected = await runRound(snapshotOf(), fakeSteps({ review: async () => reviewOf("ok", ["minor", "major"]) }));
    expect(kinds(rejected.entries).slice(-3)).toEqual(["factory-review", "factory-round", "custom_message"]);
    expect(rejected.entries.at(-2)).toMatchObject({ data: { round: 2, trigger: "review" } });
    expect((rejected.entries.at(-1) as { content: string }).content).toContain("- [major] a.ts:1: major problem");

    const approved = await runRound(snapshotOf(), fakeSteps({ review: async () => reviewOf("ok", ["minor"]) }));
    expect(lastState(approved.entries)).toMatchObject({ state: "merging" });
  });

  test.each(["malformed", "timeout", "failed"] as const)("a %s review blocks the task without a coder round", async (status) => {
    const steps = fakeSteps({ review: async () => reviewOf(status) });
    const result = await runRound(snapshotOf(), steps);
    expect(lastState(result.entries)).toEqual({ state: "blocked", round: 1, reason: `review_${status}` });
    expect(steps.calls).not.toContain("merge");
    expect(result.continue).toBe(false);
  });

  test("a review without a diff counts as passed", async () => {
    const result = await runRound(snapshotOf(), fakeSteps({ review: async () => reviewOf("no_diff") }));
    expect(lastState(result.entries)).toMatchObject({ state: "merging" });
  });

  test.each(["start", "message", "retry"] as const)("a %s round without a change ends done/no_changes: no review, no merge, no approve", async (trigger) => {
    const steps = fakeSteps({ diff: async () => ({ base: "tree", tree: "tree", paths: [], patch: "" }) });
    const result = await runRound({ ...snapshotOf({ merge: "review" }), reason: trigger }, steps);
    expect(lastState(result.entries)).toEqual({ state: "done", round: 1, reason: "no_changes" });
    expect(kinds(result.entries)).not.toContain("factory-review");
    expect(steps.calls).not.toContain("review");
    expect(steps.calls).not.toContain("merge");
    expect(result.continue).toBe(false);
  });

  test("a review that was killed ends cancelled", async () => {
    const result = await runRound(snapshotOf(), fakeSteps({ review: async () => reviewOf("aborted") }));
    expect(lastState(result.entries)).toEqual({ state: "cancelled", round: 1, reason: "cancelled" });
  });

  test("a step that throws while cancelled ends cancelled, not as a pipeline error", async () => {
    const steps = fakeSteps({
      cancelled: () => true,
      runCommand: async () => commandResult(0),
      diff: async () => {
        throw new Error("git add -A: killed");
      },
    });
    expect(lastState((await runRound(snapshotOf({}), { ...steps, config: () => ({ ...config, hooks: { after_code: [], after_merge: [] }, gates: [] }) })).entries)).toEqual({
      state: "cancelled",
      round: 1,
      reason: "cancelled",
    });
  });

  test("a cancel during a gate ends cancelled, with the killed gate recorded", async () => {
    let cancelled = false;
    const steps = fakeSteps({
      cancelled: () => cancelled,
      runCommand: async () => {
        cancelled = true;
        return commandResult(null);
      },
    });
    const result = await runRound(snapshotOf(), steps);
    expect(kinds(result.entries)).toEqual(["factory-gate", "factory-state"]);
    expect(result.entries[0]).toMatchObject({ data: { exitCode: null, killed: true } });
    expect(lastState(result.entries)).toEqual({ state: "cancelled", round: 1, reason: "cancelled" });
  });

  test("a broken factory.yaml blocks the task with the error", async () => {
    const steps = fakeSteps({
      config: () => {
        throw new Error("factory.yaml: gates must be a list of argv lists");
      },
    });
    const result = await runRound(snapshotOf(), steps);
    expect(lastState(result.entries)).toEqual({ state: "blocked", round: 1, reason: "pipeline_error: factory.yaml: gates must be a list of argv lists" });
  });
});

test("a rebase the coder left open goes back to the coder before any hook or gate runs", async () => {
  const steps = fakeSteps({ openRebase: async () => ["notes.txt"] });
  const result = await runRound(snapshotOf({}, 2, 1, true), steps);
  expect(result.continue).toBe(true);
  expect(result.entries[0]).toMatchObject({ data: { round: 3, trigger: "conflict" } });
  expect((result.entries[1] as { content: string }).content).toContain("A rebase is still in progress in the task worktree");
  expect(steps.calls).toEqual([]);
  // A current-checkout task never asks.
  const checkout = fakeSteps({ openRebase: async () => ["notes.txt"] });
  expect((await runRound(snapshotOf({}, 2), checkout)).continue).toBe(false);
});

describe("runMerge", () => {
  const gate = { round: 2, phase: "merge_gates" as const, argv: ["lint"], exitCode: 0, killed: false, durationMs: 1, outputTail: "" };

  test("records the merge gates before the merge and reports a worktree merge as merged", async () => {
    const steps = fakeSteps({ merge: async () => ({ gates: [gate], merge: { ...merged, kind: "merge", tree: null }, conflicts: null, failedGate: null }) });
    const entries = await runMerge(snapshotOf({}, 2), config, steps);
    expect(kinds(entries)).toEqual(["factory-gate", "factory-merge", "factory-state"]);
    expect(lastState(entries)).toEqual({ state: "done", round: 2, reason: "merged" });
  });

  test("a current-checkout commit ends committed", async () => {
    expect(lastState(await runMerge(snapshotOf(), config, fakeSteps()))).toEqual({ state: "done", round: 1, reason: "committed" });
  });

  test("a failed after_merge hook is recorded and leaves the task done with a reason that says so", async () => {
    const hook = { ...gate, phase: "after_merge" as const, argv: ["uv", "sync"], exitCode: 1, outputTail: "no network" };
    const data: MergeData = { ok: true, kind: "merge", commit: "def", error: "after_merge failed: uv sync: no network", depsSynced: false, tree: null, mixed: [] };
    const entries = await runMerge(snapshotOf({}, 2), config, fakeSteps({ merge: async () => ({ gates: [hook], merge: data, conflicts: null, failedGate: null }) }));
    expect(entries[0]).toMatchObject({ customType: "factory-gate", data: { phase: "after_merge", exitCode: 1 } });
    expect(entries[1]).toMatchObject({ customType: "factory-merge", data: { ok: true, error: "after_merge failed: uv sync: no network" } });
    expect(lastState(entries)).toEqual({ state: "done", round: 2, reason: "merged, after_merge failed" });
  });

  const failed: MergeData = { ok: false, kind: "merge", commit: null, error: "fast-forward of main refused", depsSynced: false, tree: null, mixed: [] };

  test("a failed merge blocks the task", async () => {
    const entries = await runMerge(snapshotOf(), config, fakeSteps({ merge: async () => ({ gates: [], merge: failed, conflicts: null, failedGate: null }) }));
    expect(entries.at(-2)).toMatchObject({ customType: "factory-merge", data: failed });
    expect(lastState(entries)).toEqual({ state: "blocked", round: 1, reason: "merge_failed" });
  });

  test("a rebase conflict merges nothing and opens a conflict round, past maxRounds by the merge grace, then blocks with max_rounds: conflict", async () => {
    const conflicting = fakeSteps({ merge: async () => ({ gates: [], merge: failed, conflicts: ["notes.txt"], failedGate: null }) });
    const entries = await runMerge(snapshotOf({ maxRounds: 3 }, 3, 1, true), config, conflicting);
    expect(kinds(entries)).toEqual(["factory-round", "custom_message", "factory-state"]);
    expect(entries[0]).toMatchObject({ data: { round: 4, trigger: "conflict" } });
    expect(entries[1]).toMatchObject({ details: { round: 3, trigger: "conflict" } });
    expect((entries[1] as { content: string }).content).toContain("- notes.txt\nRun `git rebase refs/heads/main`, resolve the conflicts");
    expect(lastState(entries)).toEqual({ state: "running", round: 4, reason: "conflict" });

    const last = await runMerge(snapshotOf({ maxRounds: 3 }, 3 + MERGE_GRACE_ROUNDS, 1, true), config, conflicting);
    expect(kinds(last)).toEqual(["custom_message", "factory-state"]);
    expect(lastState(last)).toEqual({ state: "blocked", round: 6, reason: "max_rounds: conflict" });
  });

  test("a merge gate that fails on the rebased branch goes back to the coder as a gates round, nothing merged", async () => {
    const failedGate = { argv: ["test"], result: commandResult(1, "FAILED test_new_thing") };
    const entries = await runMerge(snapshotOf({ maxRounds: 3 }, 3, 1, true), config, fakeSteps({ merge: async () => ({ gates: [], merge: failed, conflicts: null, failedGate }) }));
    expect(kinds(entries)).toEqual(["factory-round", "custom_message", "factory-state"]);
    expect((entries[1] as { content: string }).content).toContain("After the rebase onto main");
    expect((entries[1] as { content: string }).content).toContain("FAILED test_new_thing");
    expect(lastState(entries)).toEqual({ state: "running", round: 4, reason: "gates" });
  });

  test("a merge gate that exits 75 blocks the task as infra", async () => {
    const failedGate = { argv: ["test"], result: commandResult(75, "pi-host/node_modules is missing") };
    const entries = await runMerge(snapshotOf({}, 1, 1, true), config, fakeSteps({ merge: async () => ({ gates: [], merge: failed, conflicts: null, failedGate }) }));
    expect(lastState(entries)).toEqual({ state: "blocked", round: 1, reason: "infra: pi-host/node_modules is missing" });
  });

  test("a round after a merge's grace round keeps the grace budget", async () => {
    const steps = fakeSteps({ runCommand: async (argv) => commandResult(argv[0] === "lint" ? 1 : 0) });
    const result = await runRound(snapshotOf({ maxRounds: 3 }, 4, 1, true), steps);
    expect(result.continue).toBe(true);
    expect(result.entries.at(-2)).toMatchObject({ data: { round: 5, trigger: "gates" } });
  });

  test("a merge that fails under a cancel ends cancelled", async () => {
    const entries = await runMerge(snapshotOf(), config, fakeSteps({ cancelled: () => true, merge: async () => ({ gates: [], merge: failed, conflicts: null, failedGate: null }) }));
    expect(lastState(entries)).toEqual({ state: "cancelled", round: 1, reason: "cancelled" });
  });
});
