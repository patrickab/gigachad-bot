/**
 * The merge step of a passed task.
 *
 * - Consumes the task, the repo's `FactoryConfig`, the task's diff base and two `Exec`s: `exec` is killed by a cancel,
 *   `cleanup` is not
 * - Worktree task: commit, rebase onto the main branch (the target branch), gates again (`merge_gates`, skipped when
 *   the rebase moved nothing), then under the merge lock (a cancel ends the wait for it, never the update): a target
 *   checked out in a worktree (the primary or another one) is fast-forwarded there, and `hooks.after_merge` run when
 *   that is the primary and the merge changed a lockfile; a target checked out nowhere is moved with a
 *   compare-and-swap `update-ref`, no hooks. A
 *   conflicting rebase is aborted and reported with its files (`MergeOutcome.conflicts`), not as a failure. A merge
 *   gate that fails on the rebased branch is reported with its command (`MergeOutcome.failedGate`): the rebase stays,
 *   so the coder fixes the change on top of the target branch
 * - Provides `rebaseWorktree`, which a run that opens uses to bring the task branch up to date the same way. It
 *   squashes the task branch onto its merge base first: the rebase replays one commit, the task's net change
 * - The factory's own commits run no commit hooks and are not signed (`--no-verify --no-gpg-sign`)
 * - Current-checkout task: commit the paths in which the round's tree differs from the task's base onto the HEAD
 *   the task builds on, compare-and-swap, no commit hooks. What the user had in the tree or staged before the task
 *   stays as it was. A task path that changed since the round ended, or a HEAD that moved, fails the merge
 * - Reports the step that cannot be undone (`intent`) right before it, so a resume can tell whether it happened
 * - Provides `mergeTask`: the `factory-merge` data and the `factory-gate` data of the commands it ran. It never
 *   throws, a failure is `ok: false` with the reason
 */
import { writeFile } from "node:fs/promises";
import { basename, join } from "node:path";
import { protectedPaths, type FactoryConfig } from "./config.ts";
import { addAllArgs, assertCheckoutReady, git, LOCKFILES, nulList, primaryCheckout, rebaseInProgress, snapshotTree, stageTree, withScratchIndex } from "./git.ts";
import { joinQueue, leaveQueue, queueAhead } from "./queue.ts";
import type { CommandResult, Exec } from "./run.ts";
import { gateData, passed, type GateData, type MergeData, type MergeIntentData, type Task } from "./state.ts";

const ERROR_TAIL_CHARS = 1000;
/** How long a merge waits for the merge lock, polling once a second. */
const LOCK_WAIT_MS = 60_000;
/** `flock -n -E` exit code of a lock someone else holds. */
const LOCK_BUSY_EXIT = 75;
const AFTER_MERGE_TIMEOUT_MS = 10 * 60_000;
/** How often a queued merge looks whether it is its turn, or the target moved. */
const QUEUE_POLL_MS = 2_000;
/** Exit code of `MOVE_TARGET_SCRIPT` when the main branch moved past the task's base: rebase again and retry. */
const TARGET_MOVED_EXIT = 4;
/** How often a merge rebases again after the main branch moved during its merge gates. */
export const MERGE_ATTEMPTS = 3;
/**
 * Moves the main branch to the task's commit in one lock holder: `$1` primary, `$2` main branch, `$3` task branch,
 * `$4` the task's commit, `$5` the main branch's commit as the merge read it. A main branch that is no longer an
 * ancestor of the task's commit (another merge landed meanwhile) exits `TARGET_MOVED_EXIT`, nothing moved. Checked
 * out in a worktree (the primary or another one, by `git worktree list`): fast-forward there. Else compare-and-swap of
 * the ref. Prints the commit before and `primary`, `checkout` (another worktree) or `ref`.
 */
const MOVE_TARGET_SCRIPT =
  `git -C "$1" merge-base --is-ancestor "refs/heads/$2" "$4" || { echo "$2 moved past the task's base" >&2; exit ${TARGET_MOVED_EXIT}; }; ` +
  'list=$(git -C "$1" worktree list --porcelain) || exit; ' +
  'where=$(printf "%s\\n" "$list" | awk -v ref="branch refs/heads/$2" ' +
  "'/^worktree /{path=substr($0,10);n++} $0==ref{print (n==1?\"primary\":\"checkout\") \" \" path; exit}'); " +
  'if test -n "$where"; then kind=${where%% *}; dir=${where#* }; ' +
  'before=$(git -C "$dir" rev-parse HEAD) && git -C "$dir" merge --ff-only "$3" >&2 && echo "$before $kind"; exit; fi; ' +
  'git -C "$1" merge-base --is-ancestor "$5" "$4" || { echo "$4 does not descend from $5" >&2; exit 3; }; ' +
  'git -C "$1" update-ref -m "factory: merge $3" "refs/heads/$2" "$4" "$5" >&2 && echo "$5 ref"';

export interface MergeInput {
  exec: Exec;
  /** Runs what a cancel must not stop or prevent. */
  cleanup: Exec;
  root: string;
  round: number;
  task: Task;
  config: FactoryConfig;
  /** Current-checkout task: the commit or tree its change is taken against (`diffBase`). */
  base: string;
  /**
   * Current-checkout task: the working tree its round ended on and the HEAD it builds on (`Snapshot.roundTree`,
   * `Snapshot.head`). Null for a task whose states predate them: then the tree and HEAD as they are now.
   */
  checkpoint: { tree: string; head: string } | null;
  /** Called with the intent right before the merge's irreversible step. */
  intent: (data: MergeIntentData) => void;
  /** Called with what the merge does next. */
  status?: (text: string) => void;
}

export interface MergeOutcome {
  gates: GateData[];
  merge: MergeData;
  /** The files of a rebase onto the main branch that conflicted and was aborted, else null. Nothing was merged then. */
  conflicts: string[] | null;
  /** The merge gate that failed on the rebased branch, else null. Nothing was merged then. */
  failedGate: GateFailure | null;
}

export type GateFailure = { argv: string[]; result: CommandResult };

/**
 * Brings a worktree task's branch up to date: commits the worktree as `commitWorktree` does, squashes the branch onto
 * its merge base with the main branch, then rebases it onto the main branch. A conflicting rebase is aborted with
 * `cleanup` and its conflicting files returned, else null. Any other failure throws. `--autostash` carries what the
 * commit leaves out (protected paths) over the rebase.
 */
export async function rebaseWorktree(exec: Exec, cleanup: Exec, root: string, task: Task, protectedGlobs: string[]): Promise<string[] | null> {
  await commitWorktree(exec, root, task, protectedGlobs);
  await squashOntoMergeBase(exec, root, task);
  const rebase = await exec(["git", "rebase", "--autostash", `refs/heads/${task.mainBranch}`], { cwd: root });
  if (passed(rebase)) {
    // An autostash that did not apply cleanly leaves conflict markers, which the next commit would take.
    const unmerged = nulList(await git(exec, root, ["diff", "--name-only", "--diff-filter=U", "-z"]));
    if (unmerged.length > 0) throw new Error(`the autostash did not apply after the rebase onto ${task.mainBranch} (kept in git stash list): ${unmerged.join(", ")}`);
    return null;
  }
  const open = await rebaseInProgress(cleanup, root);
  const conflicts = open ?? [];
  if (open !== null || rebase.killed) await cleanup(["git", "rebase", "--abort"], { cwd: root });
  if (conflicts.length > 0 && !rebase.killed) return conflicts;
  throw new Error(`rebase onto ${task.mainBranch} failed: ${rebase.output.trim().slice(-ERROR_TAIL_CHARS) || "killed"}`);
}

/**
 * Makes the task branch one commit on top of its merge base with the main branch, of the tree it holds now. A rebase
 * then conflicts only on what the task changes in net, and a merge of the main branch the coder made is folded in.
 */
async function squashOntoMergeBase(exec: Exec, root: string, task: Task): Promise<void> {
  const head = await git(exec, root, ["rev-parse", "HEAD"]);
  const base = await git(exec, root, ["merge-base", "HEAD", `refs/heads/${task.mainBranch}`]);
  if (head === base || (await git(exec, root, ["rev-list", "--parents", "-n", "1", "HEAD"])) === `${head} ${base}`) return;
  const commit = await git(exec, root, ["commit-tree", "--no-gpg-sign", `${head}^{tree}`, "-p", base, "-m", task.title]);
  await git(exec, root, ["update-ref", "-m", `factory: squash ${task.branch}`, "HEAD", commit, head]);
}

/** 0 means nothing staged, 1 something. Any other exit is a git failure, never "nothing to commit". */
async function hasStagedChanges(exec: Exec, root: string, env?: Record<string, string>): Promise<boolean> {
  const staged = await exec(["git", "diff", "--cached", "--quiet"], { cwd: root, env });
  if (staged.killed || (staged.exitCode !== 0 && staged.exitCode !== 1)) throw new Error(`git diff --cached: ${staged.output.trim().slice(-ERROR_TAIL_CHARS) || "killed"}`);
  return staged.exitCode === 1;
}

async function commitCheckout({ exec, cleanup, root, task, config, base, checkpoint, intent }: MergeInput): Promise<MergeData> {
  await assertCheckoutReady(exec, root, task.branch);
  return withScratchIndex(exec, root, async (env, dir) => {
    const now = await stageTree(exec, root, env);
    const tree = checkpoint?.tree ?? now;
    const head = checkpoint?.head ?? (await git(exec, root, ["rev-parse", "HEAD"]));
    const changed = async (from: string, to: string, only: string[] = ["."]) =>
      nulList(await git(exec, root, ["--literal-pathspecs", "diff", "--no-renames", "--name-only", "-z", from, to, "--", ...only]));
    // The task's paths are those in which the round's tree differs from the base. Of them, the commit takes what
    // differs from HEAD.
    const ofTask = new Set(await changed(base, tree));
    const candidates = (await changed(head, tree)).filter((path) => ofTask.has(path));
    const excluded = new Set(protectedPaths(candidates, config.protected));
    const paths = candidates.filter((path) => !excluded.has(path));
    const done = (commit: string, left: string, mixed: string[]): MergeData => ({ ok: true, kind: "commit", commit, error: null, depsSynced: false, tree: left, mixed });
    if (paths.length === 0) return done(head, now, []);

    const pathsFile = join(dir, "paths");
    await writeFile(pathsFile, paths.join("\0"));
    const since = await changed(tree, now, paths);
    if (since.length > 0) throw new Error(`changed since the round ended, by someone else: ${since.join(", ")}`);
    // Committed whole, though the user had uncommitted edits in them before the task.
    const mixed = await changed(head, base, paths);
    const stagePaths = (source: string) => ["--literal-pathspecs", "restore", "--staged", `--source=${source}`, `--pathspec-from-file=${pathsFile}`, "--pathspec-file-nul"];
    // The commit is built in the scratch index: HEAD plus the task's paths. The user's index is not part of it.
    await git(exec, root, ["read-tree", head], env);
    await git(exec, root, stagePaths(tree), env);
    const commitTree = await git(exec, root, ["write-tree"], env);
    const commit = await git(exec, root, ["commit-tree", "--no-gpg-sign", commitTree, "-p", head, "-m", task.title]);
    intent({ kind: "commit", commit, before: head });
    // From here on a cancel stops nothing: the user's index still holds the old entries of the committed paths, which
    // would read as staged reverts. The update fails when HEAD is no longer the commit the task builds on.
    const moved = await cleanup(["git", "update-ref", "-m", `factory: ${task.title}`, "HEAD", commit, head], { cwd: root });
    if (!passed(moved)) throw new Error(`HEAD moved away from ${head}, nothing was committed: ${moved.output.trim().slice(-ERROR_TAIL_CHARS)}`);
    await git(cleanup, root, stagePaths(commit));
    // What the commit left in the working tree (the user's edits, protected paths) is the base of later rounds.
    return done(commit, await snapshotTree(cleanup, root), mixed);
  });
}

/** Commits everything in the task worktree but protected paths with the task title. Returns HEAD, also when there was nothing to commit. */
async function commitWorktree(exec: Exec, root: string, task: Task, protectedGlobs: string[]): Promise<string> {
  await git(exec, root, [...(await addAllArgs(exec, root)), ...protectedGlobs.map((glob) => `:(exclude,glob)${glob}`)]);
  if (await hasStagedChanges(exec, root)) await git(exec, root, ["commit", "--quiet", "--no-verify", "--no-gpg-sign", "-m", task.title]);
  return git(exec, root, ["rev-parse", "HEAD"]);
}

function tryMoveTarget(cleanup: Exec, lock: string, root: string, primary: string, task: Task, head: string, before: string) {
  const argv = ["flock", "-n", "-E", String(LOCK_BUSY_EXIT), lock, "sh", "-c", MOVE_TARGET_SCRIPT, "sh", primary, task.mainBranch, task.branch, head, before];
  return cleanup(argv, { cwd: root, maxStdoutChars: 100 });
}

async function mergeWorktree(input: MergeInput, gates: GateData[]): Promise<MergeData | string[] | GateFailure> {
  const { exec, cleanup, root, round, task, config, intent } = input;
  const status = input.status ?? (() => undefined);
  const result = (ok: boolean, commit: string | null, error: string | null, depsSynced: boolean): MergeData => ({ ok, kind: "merge", commit, error, depsSynced, tree: null, mixed: [] });
  const fail = (error: string) => result(false, null, error, false);
  const tail = (output: string) => output.trim().slice(-ERROR_TAIL_CHARS);

  const { primary, gitDir } = await primaryCheckout(exec, root);
  const lock = join(gitDir, "factory-merge.lock");
  // A rebase in progress may hold the coder's resolution: it fails the merge here, it is never aborted.
  await assertCheckoutReady(exec, root, task.branch);

  status("commit");
  await commitWorktree(exec, root, task, config.protected);
  let ontoBefore = await git(exec, root, ["merge-base", "HEAD", `refs/heads/${task.mainBranch}`]);
  const sameGates = JSON.stringify(config.merge_gates) === JSON.stringify(config.gates);
  const targetMoved = async () =>
    (await git(exec, root, ["merge-base", "HEAD", `refs/heads/${task.mainBranch}`])) !== (await git(exec, root, ["rev-parse", `refs/heads/${task.mainBranch}`]));

  // Rebased before it queues: a conflict goes to the coder at once, without a place in the queue.
  status(`rebase onto ${task.mainBranch}`);
  const conflicts = await rebaseWorktree(exec, cleanup, root, task, config.protected);
  if (conflicts) return conflicts;
  const ticket = await joinQueue(gitDir);
  try {
    for (let ahead = await queueAhead(ticket); ahead > 0; ahead = await queueAhead(ticket)) {
      status(`queued for merge, ${ahead} ahead`);
      if ((await exec(["sleep", String(QUEUE_POLL_MS / 1000)], { cwd: root })).killed) return fail("cancelled while queued for merge");
      // A merge ahead landed: rebase onto it now, a conflict leaves the queue and goes to the coder.
      if (!(await targetMoved())) continue;
      status(`rebase onto ${task.mainBranch}, a merge ahead landed`);
      const moved = await rebaseWorktree(exec, cleanup, root, task, config.protected);
      if (moved) return moved;
    }
    return await turn();
  } finally {
    await leaveQueue(ticket);
  }

  /** The ticket's turn: rebase, merge gates, move the target. Again from the rebase when the target moved anyway. */
  async function turn(): Promise<MergeData | string[] | GateFailure> {
    // Rebase, gates, move: again from the rebase while another merge lands on the main branch during the gates.
    let merged: CommandResult;
    let head: string;
    for (let attempt = 1; ; attempt++) {
      status(attempt === 1 ? `rebase onto ${task.mainBranch}` : `rebase onto ${task.mainBranch} again, it moved (attempt ${attempt})`);
      const conflicts = await rebaseWorktree(exec, cleanup, root, task, config.protected);
      if (conflicts) return conflicts;
      head = await git(exec, root, ["rev-parse", "HEAD"]);
      // Merge gates that are the round's gates, on the base the round passed them on (the squash keeps the tree), would
      // only repeat that run.
      const onto = await git(exec, root, ["merge-base", "HEAD", `refs/heads/${task.mainBranch}`]);
      for (const argv of onto === ontoBefore && sameGates ? [] : config.merge_gates) {
        status(`merge gate ${argv.join(" ")}`);
        const gate = await exec(argv, { cwd: root });
        gates.push(gateData(round, "merge_gates", argv, gate));
        if (!passed(gate)) return { argv, result: gate };
      }
      ontoBefore = onto;

      status("fast-forward");
      await assertCheckoutReady(exec, root, task.branch);
      const target = await git(exec, primary, ["rev-parse", `refs/heads/${task.mainBranch}`]);
      intent({ kind: "merge", commit: head, before: target });
      // Each try takes the lock or gives up at once. The update is not killed by a cancel: a fast-forward cut in half
      // leaves the target's checkout partly updated. The wait between tries is, and a lock holder that hangs ends in a timeout.
      const deadline = Date.now() + LOCK_WAIT_MS;
      merged = await tryMoveTarget(cleanup, lock, root, primary, task, head, target);
      while (merged.exitCode === LOCK_BUSY_EXIT) {
        if (Date.now() > deadline) return fail(`the merge lock ${lock} stayed taken for ${LOCK_WAIT_MS / 1000} s`);
        status("wait for the merge lock");
        if ((await exec(["sleep", "1"], { cwd: root })).killed) return fail("cancelled while waiting for the merge lock");
        merged = await tryMoveTarget(cleanup, lock, root, primary, task, head, target);
      }
      if (merged.exitCode === TARGET_MOVED_EXIT && attempt < MERGE_ATTEMPTS) continue;
      if (!passed(merged)) return fail(`update of ${task.mainBranch} refused: ${tail(merged.output)}`);
      break;
    }
    const [before, moved] = merged.stdout.trim().split(" ");
    const commit = await git(cleanup, primary, ["rev-parse", `refs/heads/${task.mainBranch}`]);
    if (commit !== head) return fail(`${task.mainBranch} is at ${commit} after the update, not at the task's commit ${head}`);

    // Hooks install into the primary only: another worktree, or a target checked out nowhere, gets none.
    if (moved !== "primary") return result(true, commit, null, false);
    const changed = (await git(cleanup, primary, ["diff", "--name-only", before, commit])).split("\n");
    if (!changed.some((path) => LOCKFILES.includes(basename(path)))) return result(true, commit, null, false);
    for (const argv of config.hooks.after_merge) {
      status(`after_merge ${argv.join(" ")}`);
      // Under the merge lock, so two tasks never install into the primary checkout at once. `-o`: a process the hook
      // leaves behind does not hold the lock.
      const hook = await exec(["flock", "-w", String(LOCK_WAIT_MS / 1000), "-o", lock, ...argv], { cwd: primary, timeoutMs: AFTER_MERGE_TIMEOUT_MS });
      gates.push(gateData(round, "after_merge", argv, hook));
      if (!passed(hook)) return result(true, commit, `after_merge failed: ${argv.join(" ")}: ${tail(hook.output)}`, false);
    }
    return result(true, commit, null, true);

  }
}

export async function mergeTask(input: MergeInput): Promise<MergeOutcome> {
  const kind = input.task.worktree ? "merge" : "commit";
  const gates: GateData[] = [];
  const failed = (error: string): MergeData => ({ ok: false, kind, commit: null, error, depsSynced: false, tree: null, mixed: [] });
  try {
    const merge = input.task.worktree ? await mergeWorktree(input, gates) : await commitCheckout(input);
    if (Array.isArray(merge)) return { gates, merge: failed(`rebase onto ${input.task.mainBranch} conflicts`), conflicts: merge, failedGate: null };
    if ("argv" in merge) return { gates, merge: failed(`merge gate failed: ${merge.argv.join(" ")}`), conflicts: null, failedGate: merge };
    return { gates, merge, conflicts: null, failedGate: null };
  } catch (error) {
    return { gates, merge: failed(error instanceof Error ? error.message : String(error)), conflicts: null, failedGate: null };
  }
}
