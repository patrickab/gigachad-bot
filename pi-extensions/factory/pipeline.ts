/**
 * The pipeline of one round, as control flow over injected steps. No Pi, no git, no spawn of its own.
 *
 * - Consumes the task `Snapshot` and `PipelineSteps` (the side effects: commands, diff, review, merge)
 * - Provides `runRound`: the entries of the round in order, and whether the coder gets another turn. A passed `auto`
 *   round ends in `merging` and does not merge: the caller merges once it knows the run was not aborted
 * - Provides `runMerge`: the entries of a merge, used after a passed `auto` round and by `/factory approve`. A worktree
 *   task is rebased onto its target branch first. A rebase that conflicts gets a `conflict` round, a merge gate that
 *   fails on the rebased branch a `gates` round: the coder resolves it and the task merges once that round passes
 * - Rounds a merge opens may go `MERGE_GRACE_ROUNDS` beyond `maxRounds`, so do the rounds after them: a task that
 *   passed is not blocked for what the target branch moved meanwhile
 * - A rebase the coder left open in a worktree task goes back to the coder as a `conflict` round before anything runs
 * - A hook or gate that exits `INFRA_EXIT` (75, EX_TEMPFAIL) says the environment is broken, not the change: the task
 *   ends `blocked`, `infra: <last output line>`, without feedback to the coder. A retry starts the next round once it is fixed
 */
import { protectedPaths, type FactoryConfig } from "./config.ts";
import type { TaskDiff } from "./git.ts";
import type { MergeOutcome } from "./merge.ts";
import type { CommandResult } from "./run.ts";
import {
  commandFeedback,
  conflictFeedback,
  customDraft,
  feedbackDraft,
  gateData,
  passed,
  protectedFeedback,
  rebaseOnto,
  reviewFeedback,
  roundsUsed,
  stateDraft,
  type Draft,
  type ReviewData,
  type Snapshot,
  type State,
  type Trigger,
} from "./state.ts";

export interface PipelineSteps {
  status(text: string): void;
  cancelled(): boolean;
  /** True when a user message is queued: it starts a coder turn as soon as the round's pipeline returns. */
  pendingMessages(): boolean;
  config(): FactoryConfig;
  /** Runs a hook or gate in the task's checkout. */
  runCommand(argv: string[]): Promise<CommandResult>;
  diff(): Promise<TaskDiff>;
  review(round: number, diff: TaskDiff): Promise<ReviewData>;
  merge(config: FactoryConfig): Promise<MergeOutcome>;
  /** Worktree task: the conflicting files of a rebase in progress in its checkout, null when none is. */
  openRebase(): Promise<string[] | null>;
}

export interface RoundResult {
  entries: Draft[];
  /** True when the entries end with feedback and the coder runs the next round. */
  continue: boolean;
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** Exit code of a hook or gate whose environment is broken (missing packages, tools, services): not the coder's to fix. */
export const INFRA_EXIT = 75;

function lastLine(output: string): string {
  const lines = output.trim().split("\n");
  return (lines[lines.length - 1] ?? "").slice(0, 200);
}

/**
 * The round a queued message gets after round `round` ended. That message starts a coder turn right after the
 * boundary, with no `before_agent_start`.
 */
export function messageRoundDrafts(round: number): Draft[] {
  return [customDraft("factory-round", { round: round + 1, trigger: "message" }), stateDraft("running", round + 1, "message")];
}

/** Rounds a merge's conflict or gate failure may take beyond `maxRounds`. */
export const MERGE_GRACE_ROUNDS = 3;

/**
 * A problem found in the snapshot's round: the next coder round and the feedback, or only the feedback when the run
 * used up its `maxRounds` (the caller ends it `blocked`, `max_rounds: <trigger>`). `grace` (a merge's problem, or a
 * round already past `maxRounds`, which only a merge opens) extends the budget by `MERGE_GRACE_ROUNDS`.
 */
function nextRound(snapshot: Snapshot, trigger: Trigger, text: string, grace = false): RoundResult {
  const feedback = feedbackDraft(snapshot.round, trigger, text);
  const { maxRounds } = snapshot.task.settings;
  const budget = grace || roundsUsed(snapshot) > maxRounds ? maxRounds + MERGE_GRACE_ROUNDS : maxRounds;
  if (roundsUsed(snapshot) >= budget) return { entries: [feedback], continue: false };
  return { entries: [customDraft("factory-round", { round: snapshot.round + 1, trigger }), feedback], continue: true };
}

/** The entries after `merging`: the merge gates, the merge and the state it ends in. */
export async function runMerge(snapshot: Snapshot, config: FactoryConfig, steps: PipelineSteps): Promise<Draft[]> {
  const { round } = snapshot;
  steps.status(`round ${round}: merge`);
  const { gates, merge, conflicts, failedGate } = await steps.merge(config);
  const drafts: Draft[] = gates.map((gate) => customDraft("factory-gate", gate));
  // Nothing was merged: the coder resolves the conflict or the failed gate in a round of its own, unless cancelled.
  if ((conflicts || failedGate) && steps.cancelled()) return [...drafts, stateDraft("cancelled", round, "cancelled")];
  if (failedGate?.result.exitCode === INFRA_EXIT && !failedGate.result.killed) return [...drafts, stateDraft("blocked", round, `infra: ${lastLine(failedGate.result.output)}`)];
  const problem = conflicts
    ? { trigger: "conflict" as const, text: conflictFeedback(round, snapshot.task.mainBranch, conflicts, false, rebaseOnto(snapshot.task)) }
    : failedGate
      ? { trigger: "gates" as const, text: `After the rebase onto ${snapshot.task.mainBranch}: ${commandFeedback(round, failedGate.argv, failedGate.result)}` }
      : null;
  if (problem) {
    const next = nextRound(snapshot, problem.trigger, problem.text, true);
    const state = next.continue ? stateDraft("running", round + 1, problem.trigger) : stateDraft("blocked", round, `max_rounds: ${problem.trigger}`);
    return [...drafts, ...next.entries, state];
  }
  drafts.push(customDraft("factory-merge", merge));
  // A failed `after_merge` hook cannot undo the merge: the task is done, and says what is left to do.
  if (merge.ok) drafts.push(stateDraft("done", round, merge.kind === "commit" ? "committed" : merge.error ? "merged, after_merge failed" : "merged"));
  else if (steps.cancelled()) drafts.push(stateDraft("cancelled", round, "cancelled"));
  else drafts.push(stateDraft("blocked", round, "merge_failed"));
  return drafts;
}

/** Everything after a coder turn: hooks, gates, protected paths, review, then `merging` or the wait for approve. */
export async function runRound(snapshot: Snapshot, steps: PipelineSteps): Promise<RoundResult> {
  const { round, task } = snapshot;
  const drafts: Draft[] = [];
  const messageRound = (): RoundResult => {
    drafts.push(...messageRoundDrafts(round));
    return { entries: drafts, continue: true };
  };
  const end = (state: State, reason: string): RoundResult => {
    drafts.push(stateDraft(state, round, reason));
    return state !== "cancelled" && steps.pendingMessages() ? messageRound() : { entries: drafts, continue: false };
  };
  const problem = (trigger: Trigger, text: string): RoundResult => {
    const next = nextRound(snapshot, trigger, text);
    drafts.push(...next.entries);
    return next.continue ? { entries: drafts, continue: true } : end("blocked", `max_rounds: ${trigger}`);
  };

  try {
    const config = steps.config();
    const conflicts = task.worktree ? await steps.openRebase() : null;
    if (conflicts) return problem("conflict", conflictFeedback(round, task.mainBranch, conflicts, true, rebaseOnto(task)));
    for (const [phase, commands] of [["hooks", config.hooks.after_code], ["gates", config.gates]] as const) {
      for (const argv of commands) {
        steps.status(`round ${round}: ${phase} ${argv.join(" ")}`);
        const result = await steps.runCommand(argv);
        drafts.push(customDraft("factory-gate", gateData(round, phase, argv, result)));
        if (steps.cancelled()) return end("cancelled", "cancelled");
        if (result.exitCode === INFRA_EXIT && !result.killed) return end("blocked", `infra: ${lastLine(result.output)}`);
        if (!passed(result)) return problem(phase, commandFeedback(round, argv, result));
      }
    }

    steps.status(`round ${round}: diff`);
    const diff = await steps.diff();
    // Nothing differs from the diff base, whatever opened the round: nothing to review, approve or merge.
    if (diff.paths.length === 0) return end("done", "no_changes");
    const touched = protectedPaths(diff.paths, config.protected);
    if (touched.length > 0) {
      // In the user's checkout a protected path may hold the user's own edit: no model is asked to restore it.
      return task.worktree ? problem("protected", protectedFeedback(round, touched)) : end("blocked", `protected: ${touched.join(", ")}`);
    }

    if (task.settings.review) {
      steps.status(`round ${round}: review`);
      const review = await steps.review(round, diff);
      drafts.push(customDraft("factory-review", review));
      if (steps.cancelled() || review.status === "aborted") return end("cancelled", "cancelled");
      if (review.status === "ok" && !review.approved) return problem("review", reviewFeedback(round, review.findings));
      if (review.status !== "ok" && review.status !== "no_diff") return end("blocked", `review_${review.status}`);
    }

    // The message is about a change that is neither approved nor merged yet: it goes first.
    if (steps.pendingMessages()) return messageRound();
    return end(task.settings.merge === "review" ? "waiting_approve" : "merging", "passed");
  } catch (error) {
    return steps.cancelled() ? end("cancelled", "cancelled") : end("blocked", `pipeline_error: ${errorText(error)}`);
  }
}
