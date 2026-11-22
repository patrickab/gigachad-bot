/**
 * Agent factory: the pipeline of a factory task, run inside the task's own coding session.
 *
 *   /factory start {"request", "title", "settings"}   record the task and start the coder
 *   /factory approve                                  merge a task that waits for approve
 *   /factory retry                                    another coder turn on a blocked or cancelled task
 *   /factory cancel                                   kill the running command group, end `cancelled`
 *
 * - After every coder turn (`agent_before_settle`) it runs `hooks.after_code`, `gates`, the protected-path check and
 *   the review. A problem goes back to the coder as a `factory-feedback` message and starts the next round, up to
 *   `settings.maxRounds`. A passed round waits for approve or merges, a round with an empty diff ends `done`/`no_changes`
 * - The merge of a passed round starts in `agent_settled`, the first place that knows whether the run was aborted: an
 *   abort is invisible inside `agent_before_settle`, so the steps already started there run to their end, but nothing
 *   is merged after it and the task ends `blocked`/`stopped`. The merge is not awaited there: Pi defers every prompt,
 *   `/factory cancel` included, until the `agent_settled` handlers return
 * - `factory.yaml` and the role prompts come from a commit of the main branch (a worktree task: its base, `taskBase`,
 *   at the run's start, a current-checkout task: its base commit), never from the tree the coder edits
 * - A worktree task works on its own branch, which gigachad made off `settings.sourceBranch`, and stays up to date with
 *   the target branch `settings.branch`: a run that opens commits the worktree and rebases it onto that branch. With a
 *   source that is not the target, it rebases onto their merge instead (the upstream, `nextUpstream`): the source's
 *   commits are never the task's change, and reach the target with its merge. A conflicting rebase is aborted and the
 *   run opens with a `conflict` round instead, so does a rebase the coder left open, which is never aborted. A rebase
 *   or commit that fails otherwise (a hook, a deleted branch, an untracked file in the way, a source that does not
 *   merge into the target) opens no run and ends the task `blocked`, `rebase_failed: <git error>`: a later retry or
 *   message tries again
 * - A current-checkout task shares its tree with the user. A state that ends a run records the tree and the HEAD
 *   (`tree`, `head`). A run opened on a done task measures from the tree as it is then. On any other state a run is
 *   refused while a path changed since. A HEAD that moved (the user committed or pulled) is taken as the new HEAD:
 *   the task's paths keep their old base, everything else in the tree now is the user's, so is a task path the user
 *   changed since. At rest that happens when a run opens and on approve, during a coder turn before its round
 * - Every step is a `custom` session entry (`state.ts`). The task is rebuilt from them on `session_start`, where a
 *   merge that landed while the session went down is recorded from its `factory-merge-intent`
 * - A plain message to a waiting, blocked, cancelled or done task starts new rounds
 * - Loaded into every coding session: without a `factory-task` entry no handler does anything
 */
import { basename, join } from "node:path";
import { getAgentDir, type ExtensionAPI, type ExtensionCommandContext, type ExtensionContext } from "@earendil-works/pi-coding-agent";
import { loadTaskFiles, type TaskFiles } from "./config.ts";
import { abortLeftoverRebase, diffBase, git, keepTree, LOCKFILES, nulList, primaryCheckout, rebaseInProgress, snapshotTree, taskBase, taskDiff, taskGit, treeWithPathsOf } from "./git.ts";
import { mergeTask, rebaseWorktree } from "./merge.ts";
import { messageRoundDrafts, runMerge, runRound, type PipelineSteps, type RoundResult } from "./pipeline.ts";
import { runReview } from "./review.ts";
import { CommandRunner } from "./run.ts";
import {
  applyEntry,
  conflictFeedback,
  customDraft,
  FEEDBACK_TYPE,
  followUpMessages,
  mergeFailedPrompt,
  parseStartPayload,
  rebaseOnto,
  rebuild,
  RETRY_PROMPT,
  stateDraft,
  wasInterrupted,
  type CustomDraft,
  type Draft,
  type FeedbackDraft,
  type MergeData,
  type Snapshot,
  type State,
  type StateData,
  type Task,
  type Trigger,
} from "./state.ts";

/** Shows what the pipeline does: set while a round, a review or a merge runs. */
const STATUS_KEY = "factory";
/** pi-host keeps a session with this status from idle eviction. A merge sets it: `/factory approve` starts no agent run. */
const LIVENESS_KEY = "session-liveness";
/** Tag of the system prompt section that carries `.factory/agents/coder.md`. */
const ROLE_SECTION = "factory_role";
/** States a user message opens a new run from. */
const RESTING_STATES: readonly State[] = ["waiting_approve", "blocked", "done", "cancelled"];

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * What a run opens with. A current-checkout task: a new diff base and HEAD when it opens on a done task or after its
 * HEAD moved. A worktree task: the feedback of a rebase that conflicts, then the run opens with a `conflict` round.
 */
type RunOpening = Pick<StateData, "base" | "head"> & { conflict?: string; failed?: string };

/** The feedback message of a `conflict` round that opens a run, as `pi.sendMessage` and `before_agent_start` take it. */
function conflictMessage(round: number, content: string): Omit<FeedbackDraft, "type"> {
  return { customType: FEEDBACK_TYPE, content, display: true, details: { round, trigger: "conflict" } };
}

export default function (pi: ExtensionAPI) {
  let snapshot: Snapshot | undefined;
  /** `factory.yaml` and the role prompts the current run is held to, read once per run. */
  let files: Promise<TaskFiles> | undefined;
  let starting = false;
  /** Set on `session_shutdown`: `pi` is stale from then on and a merge still in flight must not write through it. */
  let disposed = false;
  /** The state the last `agent_before_settle` pipeline ended in, until `agent_settled` has acted on it. */
  let roundEnd: State | undefined;
  /** The working tree the current round's diff was taken of. */
  let roundTree: string | undefined;
  /** What the `input` handler checked for the user message whose run `before_agent_start` opens next. */
  let opened: RunOpening | undefined;
  const runner = new CommandRunner();

  const record = (customType: string, data: unknown) => {
    pi.appendEntry(customType, data);
    snapshot = applyEntry(snapshot, customType, data);
  };
  const keep = (ctx: ExtensionContext, name: "base" | "round", tree: string) => keepTree(runner.cleanup, ctx.cwd, ctx.sessionManager.getSessionId(), name, tree);
  /**
   * `tree` and `head` of a state that ends a run of a current-checkout task: `tree`, else the working tree now. Left
   * out when git fails: then a later merge takes the tree as it is by then.
   */
  const checkpoint = async (ctx: ExtensionContext, current: Snapshot, tree?: string): Promise<Pick<StateData, "tree" | "head">> => {
    if (current.task.worktree || !current.head) return {};
    try {
      const ended = tree ?? (await snapshotTree(runner.cleanup, ctx.cwd));
      await keep(ctx, "round", ended);
      return { tree: ended, head: current.head };
    } catch {
      return {};
    }
  };
  const endState = async (ctx: ExtensionContext, current: Snapshot, state: State, reason: string, tree?: string) => {
    const stamp = await checkpoint(ctx, current, tree);
    record("factory-state", { state, round: current.round, reason, ...stamp } satisfies StateData);
  };
  const startRun = (current: Snapshot, trigger: Trigger, { conflict, failed: _failed, ...stamp }: RunOpening) => {
    runner.reset();
    files = undefined;
    const opens = conflict === undefined ? trigger : "conflict";
    record("factory-round", { round: current.round + 1, trigger: opens });
    record("factory-state", { state: "running", round: current.round + 1, reason: opens, ...stamp } satisfies StateData);
  };
  const taskFiles = (ctx: ExtensionContext, task: Task) =>
    (files ??= (async () => {
      const rules = task.worktree ? await taskBase(runner.cleanup, ctx.cwd, task) : task.baseSha;
      return loadTaskFiles(runner.cleanup, ctx.cwd, rules);
    })());

  const changedPaths = async (ctx: ExtensionContext, from: string, to: string, only: string[] = []) =>
    nulList(await git(runner.cleanup, ctx.cwd, ["--literal-pathspecs", "diff", "--no-renames", "--name-only", "-z", from, to, "--", ...only]));

  /**
   * The diff base of a current-checkout task at rest taken again on its moved HEAD `head`, with the tree now: the
   * paths in which the last round's tree differs from the old base keep that base, unless they changed since (the
   * user's commit or pull holds them then). Everything else is the tree now.
   */
  const retakeBase = async (current: Snapshot, ctx: ExtensionContext, head: string): Promise<Required<Pick<StateData, "base" | "head" | "tree">>> => {
    const now = await snapshotTree(runner.cleanup, ctx.cwd);
    const oldBase = await diffBase(runner.cleanup, ctx.cwd, current.task, current.mergedBase);
    const touched = await changedPaths(ctx, oldBase, current.roundTree ?? now);
    const since = new Set(current.roundTree && touched.length > 0 ? await changedPaths(ctx, current.roundTree, now, touched) : []);
    const base = await treeWithPathsOf(runner.cleanup, ctx.cwd, now, oldBase, touched.filter((path) => !since.has(path)));
    await keep(ctx, "base", base);
    return { base, head, tree: now };
  };

  /**
   * A current-checkout task whose HEAD moved during the coder's turn (the user committed or pulled): its running state
   * on the new HEAD with the base taken again, else undefined. A path the move changed that the tree now holds as the
   * new HEAD does is the user's, the base takes it from there. Every other path keeps the old base.
   */
  const movedDuringTurn = async (current: Snapshot, ctx: ExtensionContext): Promise<StateData | undefined> => {
    if (current.task.worktree || !current.head) return undefined;
    const head = await git(runner.cleanup, ctx.cwd, ["rev-parse", "HEAD"]);
    if (head === current.head) return undefined;
    const now = await snapshotTree(runner.cleanup, ctx.cwd);
    const pulled = await changedPaths(ctx, current.head, head);
    const edited = new Set(pulled.length > 0 ? await changedPaths(ctx, head, now, pulled) : []);
    const oldBase = await diffBase(runner.cleanup, ctx.cwd, current.task, current.mergedBase);
    const base = await treeWithPathsOf(runner.cleanup, ctx.cwd, oldBase, head, pulled.filter((path) => !edited.has(path)));
    await keep(ctx, "base", base);
    return { state: "running", round: current.round, reason: current.reason, base, head };
  };

  /**
   * The checks before a run opens on a task at rest. A worktree task: a rebase in progress (the coder's) is left as
   * it is and gets a `conflict` round, else the worktree is committed and rebased onto the main branch, a conflicting
   * rebase aborted into a `conflict` round, any other failure of it returned as `failed`. A current-checkout task: on
   * `done` the tree now becomes the diff base. After a HEAD move the base is taken again (`retakeBase`). Without
   * one, a task path changed since its last state refuses the run (throws).
   */
  const openRun = async (current: Snapshot, ctx: ExtensionContext): Promise<RunOpening> => {
    const { task } = current;
    if (task.worktree) {
      const open = await rebaseInProgress(runner.cleanup, ctx.cwd);
      if (open) return { conflict: conflictFeedback(current.round, task.mainBranch, open, true, rebaseOnto(task)) };
      const protectedGlobs = (await taskFiles(ctx, task).catch(() => undefined))?.config.protected ?? [];
      try {
        const conflicts = await rebaseWorktree(runner.cleanup, runner.cleanup, ctx.cwd, task, protectedGlobs);
        return conflicts ? { conflict: conflictFeedback(current.round, task.mainBranch, conflicts, false, rebaseOnto(task)) } : {};
      } catch (error) {
        return { failed: `rebase_failed: ${errorText(error)}` };
      }
    }
    const head = await git(runner.cleanup, ctx.cwd, ["rev-parse", "HEAD"]);
    if (current.state === "done") {
      const base = await snapshotTree(runner.cleanup, ctx.cwd);
      await keep(ctx, "base", base);
      return { base, head };
    }
    if (current.head && head !== current.head) {
      const { base } = await retakeBase(current, ctx, head);
      return { base, head };
    }
    if (current.roundTree) {
      const now = await snapshotTree(runner.cleanup, ctx.cwd);
      const changed = await changedPaths(ctx, current.roundTree, now);
      if (changed.length > 0) throw new Error(`changed since the task's last round, by someone else: ${changed.join(", ")}`);
    }
    return {};
  };

  const steps = (ctx: ExtensionContext, task: Task, taskFilesNow: TaskFiles, liveness: boolean): PipelineSteps => {
    const root = ctx.cwd;
    const status = (text: string) => {
      ctx.ui.setStatus(STATUS_KEY, text);
      if (liveness) ctx.ui.setStatus(LIVENESS_KEY, text);
    };
    const base = () => diffBase(runner.run, root, task, snapshot?.mergedBase ?? null);
    return {
      status,
      cancelled: () => runner.cancelled,
      pendingMessages: () => ctx.hasPendingMessages(),
      config: () => taskFilesNow.config,
      runCommand: (argv) => runner.run(argv, { cwd: root }),
      diff: async () => {
        const diff = await taskDiff(runner.run, root, await base());
        roundTree = diff.tree;
        return diff;
      },
      review: (round, diff) =>
        runReview({
          exec: runner.run,
          root,
          round,
          task,
          followUps: followUpMessages(ctx.sessionManager.getBranch()),
          diff,
          model: task.settings.models.reviewer ?? (ctx.model ? `${ctx.model.provider}/${ctx.model.id}` : null),
          rolePrompt: taskFilesNow.prompts.reviewer,
          reviewsDir: join(getAgentDir(), "factory-reviews", ctx.sessionManager.getSessionId()),
          cancelled: () => runner.cancelled,
        }),
      merge: async (config) => {
        const round = snapshot?.round ?? 0;
        const mergeStatus = (text: string) => status(`round ${round}: merge, ${text}`);
        const checkpoint = snapshot?.roundTree && snapshot.head ? { tree: snapshot.roundTree, head: snapshot.head } : null;
        const intent = (data: unknown) => {
          if (!disposed) record("factory-merge-intent", data);
        };
        return mergeTask({ exec: runner.run, cleanup: runner.cleanup, root, round, task, config, base: await base(), checkpoint, intent, status: mergeStatus });
      },
      openRebase: () => rebaseInProgress(runner.run, root),
    };
  };

  const start = async (json: string, ctx: ExtensionCommandContext) => {
    if (snapshot || starting) throw new Error("this session already holds a task");
    if (!ctx.isIdle()) throw new Error("the session is busy");
    starting = true;
    try {
      runner.reset();
      const payload = parseStartPayload(json);
      const taskGitFacts = await taskGit(runner.run, ctx.cwd, payload.settings.branch);
      const loaded = await loadTaskFiles(runner.run, ctx.cwd, taskGitFacts.baseSha);
      const coder = payload.settings.models.coder;
      if (coder) {
        const slash = coder.indexOf("/");
        const model = ctx.modelRegistry.find(coder.slice(0, slash), coder.slice(slash + 1));
        if (!model || !(await pi.setModel(model))) throw new Error(`model ${coder} is not available`);
      }
      if (taskGitFacts.baseTree) await keep(ctx, "base", taskGitFacts.baseTree);
      const task: Task = { title: payload.title, request: payload.request, settings: payload.settings, ...taskGitFacts };
      files = Promise.resolve(loaded);
      record("factory-task", task);
      record("factory-round", { round: 1, trigger: "start" });
      record("factory-state", { state: "running", round: 1, reason: "start" });
      // Everything that can fail from here on runs after this prompt: entries reach the file only once a message exists.
      pi.sendUserMessage(task.request);
    } finally {
      starting = false;
    }
  };

  /** Records a merge and the state it ends in. A current-checkout commit's tree is the diff base from then on. */
  const recordMerge = async (ctx: ExtensionContext, current: Snapshot, drafts: Draft[]) => {
    const merge = drafts.find((draft): draft is CustomDraft => draft.type === "custom" && draft.customType === "factory-merge")?.data as MergeData | undefined;
    const committed = merge?.ok && merge.kind === "commit" && merge.tree ? merge.tree : undefined;
    if (committed) await keep(ctx, "base", committed).catch(() => undefined);
    // A failed merge leaves the round's tree as it was: what changed since is still not the task's.
    const after = { ...current, head: merge?.ok && merge.kind === "commit" ? merge.commit : current.head };
    const stamp = await checkpoint(ctx, after, committed ?? current.roundTree ?? undefined);
    if (disposed) return;
    for (const draft of drafts) {
      if (draft.type !== "custom") continue;
      record(draft.customType, draft.customType === "factory-state" ? { ...(draft.data as StateData), ...stamp } : draft.data);
    }
  };

  /**
   * The merge of a task in `merging`, with the entries it ends in. Started without being awaited, from `agent_settled`
   * after a passed `auto` round and from `/factory approve`. A rebase that conflicts, or a merge gate that fails on the
   * rebased branch, sends its feedback, which starts the coder's round unless the run's rounds are used up.
   */
  const finishMerge = async (current: Snapshot, ctx: ExtensionContext) => {
    let drafts: Draft[];
    try {
      const mergeSteps = steps(ctx, current.task, await taskFiles(ctx, current.task), true);
      drafts = await runMerge(current, mergeSteps.config(), mergeSteps);
    } catch (error) {
      drafts = [runner.cancelled ? stateDraft("cancelled", current.round, "cancelled") : stateDraft("blocked", current.round, `pipeline_error: ${errorText(error)}`)];
    }
    // The task then stays `merging` in the file and is reconciled on `session_start`.
    if (disposed) return;
    await recordMerge(ctx, current, drafts);
    if (disposed) return;
    ctx.ui.setStatus(STATUS_KEY, undefined);
    ctx.ui.setStatus(LIVENESS_KEY, undefined);
    runner.reset();
    const feedback = drafts.find((draft): draft is FeedbackDraft => draft.type === "custom_message");
    if (feedback) {
      const { type: _type, ...message } = feedback;
      pi.sendMessage(message, { triggerTurn: snapshot?.state === "running" });
    }
  };

  /**
   * The merge entries of a task found `merging` on resume whose merge landed after its intent was written, else
   * undefined. A current-checkout commit also gets the user's index back in line for the committed paths whose
   * entry is still the one before the commit.
   */
  const landedMerge = async (current: Snapshot, ctx: ExtensionContext): Promise<Draft[] | undefined> => {
    const intent = current.mergeIntent;
    if (!intent) return undefined;
    const exec = runner.cleanup;
    const repo = intent.kind === "merge" ? (await primaryCheckout(exec, ctx.cwd)).primary : ctx.cwd;
    const target = intent.kind === "merge" ? `refs/heads/${current.task.mainBranch}` : "HEAD";
    if ((await exec(["git", "merge-base", "--is-ancestor", intent.commit, target], { cwd: repo })).exitCode !== 0) return undefined;
    const changed = nulList(await git(exec, repo, ["diff", "--no-renames", "--name-only", "-z", intent.before, intent.commit]));
    const merge: MergeData = { ok: true, kind: intent.kind, commit: intent.commit, error: null, depsSynced: false, tree: null, mixed: [] };
    if (intent.kind === "merge") {
      if (changed.some((path) => LOCKFILES.includes(basename(path)))) merge.error = "after_merge did not run: the session ended during the merge";
      return [{ type: "custom", customType: "factory-merge", data: merge }, stateDraft("done", current.round, merge.error ? "merged, after_merge failed" : "merged")];
    }
    if (changed.length > 0) {
      const restaged = new Set(nulList(await git(exec, repo, ["--literal-pathspecs", "diff", "--cached", "--name-only", "-z", intent.before, "--", ...changed])));
      const stale = changed.filter((path) => !restaged.has(path));
      if (stale.length > 0) await git(exec, repo, ["--literal-pathspecs", "restore", "--staged", `--source=${intent.commit}`, "--", ...stale]);
    }
    merge.tree = await snapshotTree(exec, repo);
    return [{ type: "custom", customType: "factory-merge", data: merge }, stateDraft("done", current.round, "committed")];
  };

  const resume = async (current: Snapshot, ctx: ExtensionContext) => {
    if (current.state === "merging") {
      const landed = await landedMerge(current, ctx).catch(() => undefined);
      if (landed) return recordMerge(ctx, current, landed);
      if (current.task.worktree) await abortLeftoverRebase(runner.cleanup, ctx.cwd).catch(() => undefined);
      // A current-checkout merge does not touch the tree: the round's tree still tells the task's paths apart.
      return endState(ctx, current, "blocked", "interrupted", current.roundTree ?? undefined);
    }
    await endState(ctx, current, "blocked", "interrupted");
  };

  const command = async (args: string, ctx: ExtensionCommandContext) => {
    const text = args.trim();
    const verb = text.split(/\s+/, 1)[0];
    if (verb === "start") return start(text.slice(verb.length).trim(), ctx);
    const current = snapshot;
    if (!current) throw new Error("this session holds no task");
    const { state } = current;
    if (verb === "approve") {
      if (state !== "waiting_approve") throw new Error("the task is not waiting for approve");
      if (!ctx.isIdle()) throw new Error("the session is busy");
      runner.reset();
      // A current-checkout task merges onto a HEAD the user moved: the commit then builds on it, with the tree now.
      const head = current.task.worktree || !current.head ? null : await git(runner.cleanup, ctx.cwd, ["rev-parse", "HEAD"]);
      const moved = head && head !== current.head ? await retakeBase(current, ctx, head) : {};
      record("factory-state", { state: "merging", round: current.round, reason: "approved", ...moved } satisfies StateData);
      void finishMerge(snapshot ?? current, ctx).catch(() => undefined);
    } else if (verb === "retry") {
      if (state !== "blocked" && state !== "cancelled") throw new Error("retry needs a blocked or cancelled task");
      if (!ctx.isIdle()) throw new Error("the session is busy");
      const opening = await openRun(current, ctx);
      if (opening.failed !== undefined) {
        await endState(ctx, current, "blocked", opening.failed);
        throw new Error(opening.failed);
      }
      startRun(current, "retry", opening);
      // A failed merge is in no message the coder sees.
      const failed = current.reason === "merge_failed" && current.mergeError;
      if (opening.conflict !== undefined) pi.sendMessage(conflictMessage(current.round, opening.conflict), { triggerTurn: true });
      else pi.sendUserMessage(failed ? mergeFailedPrompt(failed) : RETRY_PROMPT);
    } else if (verb === "cancel") {
      if (state === "done" || state === "cancelled") throw new Error("the task is not active");
      runner.cancel();
      // A merge or a round in flight ends itself and writes `cancelled`. An idle task is cancelled here.
      if (state === "merging") return;
      if (!ctx.isIdle()) return ctx.abort();
      await endState(ctx, current, "cancelled", "cancelled");
      runner.reset();
    } else {
      throw new Error(`unknown command "${verb}"`);
    }
  };

  pi.registerCommand("factory", {
    description: "Agent factory task: start <json> | approve | retry | cancel",
    handler: async (args, ctx) => {
      try {
        await command(args, ctx);
      } catch (error) {
        ctx.ui.notify(`factory: ${errorText(error)}`, "error");
      }
    },
  });

  pi.on("session_start", async (_event, ctx) => {
    snapshot = rebuild(ctx.sessionManager.getBranch());
    files = undefined;
    if (snapshot && wasInterrupted(snapshot)) await resume(snapshot, ctx);
  });

  pi.on("session_tree", (_event, ctx) => {
    snapshot = rebuild(ctx.sessionManager.getBranch());
    files = undefined;
  });

  pi.on("session_shutdown", () => {
    disposed = true;
    runner.cancel();
  });

  pi.on("input", async (_event, ctx) => {
    if (!snapshot) return;
    if (snapshot.state === "merging") {
      ctx.ui.notify("factory: a merge is running, send the message when it has ended", "warning");
      return { action: "handled" };
    }
    if (!RESTING_STATES.includes(snapshot.state)) return;
    const atRest = snapshot;
    try {
      opened = await openRun(atRest, ctx);
    } catch (error) {
      ctx.ui.notify(`factory: the message starts no run: ${errorText(error)}`, "error");
      return { action: "handled" };
    }
    if (opened.failed !== undefined) {
      const { failed } = opened;
      opened = undefined;
      await endState(ctx, atRest, "blocked", failed);
      ctx.ui.notify(`factory: the message starts no run: ${failed}`, "error");
      return { action: "handled" };
    }
  });

  pi.on("before_agent_start", async (event, ctx) => {
    if (!snapshot) return;
    const opening = opened ?? {};
    opened = undefined;
    const atRest = snapshot;
    const opens = event.messageType === "user" && RESTING_STATES.includes(atRest.state);
    if (opens) startRun(atRest, "message", opening);
    // A `factory.yaml` that does not parse is reported by the round, not here.
    const taskFilesNow = await taskFiles(ctx, snapshot.task).catch(() => undefined);
    if (taskFilesNow?.prompts.coder) event.systemPromptOptions.sections[ROLE_SECTION] = taskFilesNow.prompts.coder;
    // The user's message and the conflict both reach the coder in this run.
    if (opens && opening.conflict !== undefined) return { message: conflictMessage(atRest.round, opening.conflict) };
  });

  pi.on("agent_start", (_event, ctx) => {
    if (snapshot?.state === "running") ctx.ui.setStatus(STATUS_KEY, `round ${snapshot.round}: coder`);
  });

  pi.on("agent_before_settle", async (event, ctx) => {
    if (!snapshot || snapshot.state !== "running") return;
    // An aborted run is classified in `agent_settled`, which is also reached when this boundary is skipped.
    if (event.outcome === "aborted") return;
    let current = snapshot;
    const { round, task } = current;
    roundTree = undefined;
    const stopped = (reason: string): RoundResult => {
      const entries: Draft[] = [stateDraft("blocked", round, reason)];
      // A message queued meanwhile runs right after this boundary: it gets its round.
      return ctx.hasPendingMessages() ? { entries: [...entries, ...messageRoundDrafts(round)], continue: true } : { entries, continue: false };
    };
    let result: RoundResult;
    if (event.outcome === "error") {
      result = stopped("agent_error");
    } else {
      let moved: StateData | undefined;
      try {
        // The user committed or pulled during the turn: the round builds on the new HEAD and its base taken again.
        moved = await movedDuringTurn(current, ctx);
        if (moved) current = snapshot = applyEntry(current, "factory-state", moved) ?? current;
        result = await runRound(current, steps(ctx, task, await taskFiles(ctx, task), false));
      } catch (error) {
        result = stopped(`pipeline_error: ${errorText(error)}`);
      }
      if (moved) result = { ...result, entries: [customDraft("factory-state", moved), ...result.entries] };
    }
    const ended = result.entries.find((draft) => draft.type === "custom" && draft.customType === "factory-state" && (draft.data as StateData).state !== "running");
    if (ended?.type === "custom") ended.data = { ...(ended.data as StateData), ...(await checkpoint(ctx, current, roundTree)) };
    for (const draft of result.entries) if (draft.type === "custom") snapshot = applyEntry(snapshot, draft.customType, draft.data);
    roundEnd = snapshot?.state;
    const entries = [...event.entries, ...result.entries];
    return result.continue ? { entries, continue: true } : { entries };
  });

  pi.on("agent_settled", async (event, ctx) => {
    if (!snapshot) return;
    const current = snapshot;
    const ended = roundEnd === current.state ? roundEnd : undefined;
    roundEnd = undefined;
    // `running`: no step ended the run. `merging`: the round passed and its merge is due. An aborted run keeps neither,
    // and no approve either: the abort may have come while the round's steps ran, which they cannot see.
    const open = current.state === "running" || ended === "merging" || (ended === "waiting_approve" && event.aborted);
    if (open && runner.cancelled) await endState(ctx, current, "cancelled", "cancelled");
    else if (open && event.aborted) await endState(ctx, current, "blocked", "stopped");
    else if (ended === "merging") {
      // Not awaited: Pi holds back every prompt until this handler returns, `/factory cancel` too.
      void finishMerge(current, ctx).catch(() => undefined);
      return;
    } else if (current.state === "running") await endState(ctx, current, "blocked", "interrupted");
    ctx.ui.setStatus(STATUS_KEY, undefined);
    runner.reset();
  });
}
