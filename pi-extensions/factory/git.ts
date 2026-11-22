/**
 * Git facts a factory task needs, as thin calls over the command runner.
 *
 * - Consumes an `Exec` and a checkout path
 * - Provides the task's git identity (`taskGit`), its primary checkout, the snapshot of a working tree as a tree id
 *   (`snapshotTree`), the task's diff base and its diff (changed paths and patch, untracked files included)
 * - Provides the checks a merge needs before it writes: the checkout is on its branch with no git operation in
 *   progress (`assertCheckoutReady`). Tells a rebase in progress and its conflicting files (`rebaseInProgress`), and
 *   aborts one a killed merge left behind (`abortLeftoverRebase`)
 * - Provides a current-checkout task's diff base taken again after its HEAD moved (`treeWithPathsOf`)
 * - Provides where a worktree task's own change starts (`taskBase`), and for a task with its own source branch the
 *   upstream it is rebased onto: the merge of its target and source branch (`nextUpstream`, `upstreamMoved`)
 * - Keeps a tree a task still needs from git's garbage collection (`keepTree`)
 * - Nothing here writes the checkout's own index or its files: staging goes through a scratch index
 */
import { existsSync, realpathSync } from "node:fs";
import { copyFile, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import type { Exec } from "./run.ts";
import { hasOwnSource, upstreamRef, type Task } from "./state.ts";

const ERROR_TAIL_CHARS = 1000;
/** Upper bound of what one git call may print: path lists and file contents, never a whole patch. */
const GIT_STDOUT_MAX_CHARS = 4_000_000;
/** What a diff keeps of its patch. The review caps it again, far lower. */
const PATCH_KEEP_CHARS = 1_000_000;

/** Their bodies are noise in a review and go last in a patch. A change to one makes a merge run `hooks.after_merge`. */
export const LOCKFILES = ["uv.lock", "package-lock.json", "bun.lock", "bun.lockb"];

/** Stdout of `git <args>` in `cwd`, trimmed. Throws with the end of the output when git fails. */
export async function git(exec: Exec, cwd: string, args: string[], env?: Record<string, string>): Promise<string> {
  const result = await exec(["git", ...args], { cwd, env, maxStdoutChars: GIT_STDOUT_MAX_CHARS });
  if (result.exitCode !== 0 || result.killed) {
    throw new Error(`git ${args.join(" ")}: ${result.output.trim().slice(-ERROR_TAIL_CHARS) || "killed"}`);
  }
  if (result.stdoutTruncated) throw new Error(`git ${args.join(" ")}: more than ${GIT_STDOUT_MAX_CHARS} characters of output`);
  return result.stdout.trim();
}

/** The paths of a `-z` listing. */
export function nulList(text: string): string[] {
  return text.split("\0").filter((path) => path !== "");
}

export type TaskGit = Pick<Task, "worktree" | "branch" | "mainBranch" | "baseSha" | "baseTree">;

/** Primary checkout of the repository `cwd` belongs to, and its git directory (where the merge lock lives). */
export async function primaryCheckout(exec: Exec, cwd: string): Promise<{ primary: string; gitDir: string }> {
  const gitDir = await git(exec, cwd, ["rev-parse", "--path-format=absolute", "--git-common-dir"]);
  return { primary: dirname(gitDir), gitDir };
}

/** A scratch index for `root` that starts as a copy of its real one. `GIT_INDEX_FILE` in `env` points git at it. */
export async function withScratchIndex<T>(exec: Exec, root: string, use: (env: { GIT_INDEX_FILE: string }, dir: string) => Promise<T>): Promise<T> {
  const dir = await mkdtemp(join(tmpdir(), "factory-index-"));
  try {
    const index = await git(exec, root, ["rev-parse", "--path-format=absolute", "--git-path", "index"]);
    const env = { GIT_INDEX_FILE: join(dir, "index") };
    if (existsSync(index)) await copyFile(index, env.GIT_INDEX_FILE);
    return await use(env, dir);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

/**
 * `git add -A` arguments that leave out untracked nested repositories: one without a commit makes `add` fail, one
 * with a commit would be staged as a gitlink. Git lists both with a trailing slash.
 */
export async function addAllArgs(exec: Exec, root: string, env?: Record<string, string>): Promise<string[]> {
  const nested = nulList(await git(exec, root, ["ls-files", "--others", "--exclude-standard", "-z"], env)).filter((path) => path.endsWith("/"));
  return ["add", "-A", "--", ".", ...nested.map((path) => `:(exclude,literal)${path.slice(0, -1)}`)];
}

/** Stages the whole working tree into the scratch index of `env` and returns its tree id. */
export async function stageTree(exec: Exec, root: string, env: { GIT_INDEX_FILE: string }): Promise<string> {
  await git(exec, root, await addAllArgs(exec, root, env), env);
  return git(exec, root, ["write-tree"], env);
}

/**
 * Points `refs/factory/<session id>/<name>` at a commit of `tree`. A snapshot tree is referenced by nothing else, and
 * `git gc` prunes it once it is older than `gc.pruneExpire`. The backend deletes the refs with the task.
 */
export async function keepTree(exec: Exec, root: string, sessionId: string, name: "base" | "round", tree: string): Promise<void> {
  const commit = await git(exec, root, ["commit-tree", "--no-gpg-sign", tree, "-m", `factory ${name}`]);
  await git(exec, root, ["update-ref", `refs/factory/${sessionId}/${name}`, commit]);
}

/** Tree id of the working tree of `root` as it is now: tracked and untracked files, ignored ones left out. */
export function snapshotTree(exec: Exec, root: string): Promise<string> {
  return withScratchIndex(exec, root, (env) => stageTree(exec, root, env));
}

/**
 * Throws when `cwd` is not the root of a git checkout on a branch with at least one commit. A worktree task's main
 * branch is `target` (`settings.branch`), which must exist. A current-checkout task is its own main branch and also
 * gets `baseTree`, the working tree at this moment: what is already there is the user's, not the task's.
 */
export async function taskGit(exec: Exec, cwd: string, target: string): Promise<TaskGit> {
  const lines = await git(exec, cwd, ["rev-parse", "--path-format=absolute", "--show-toplevel", "--git-dir", "--git-common-dir", "HEAD"]);
  const [toplevel, gitDir, commonDir, baseSha] = lines.split("\n");
  if (realpathSync(toplevel) !== realpathSync(cwd)) throw new Error(`${cwd} is not the root of its git checkout (${toplevel})`);
  const worktree = gitDir !== commonDir;
  const branch = await git(exec, cwd, ["symbolic-ref", "--short", "HEAD"]);
  if (worktree && (await exec(["git", "rev-parse", "--verify", "--quiet", `refs/heads/${target}`], { cwd })).exitCode !== 0) {
    throw new Error(`the target branch ${target} does not exist`);
  }
  const mainBranch = worktree ? target : branch;
  return { worktree, branch, mainBranch, baseSha, baseTree: worktree ? null : await snapshotTree(exec, cwd) };
}

/**
 * The commit or tree a task's diff is taken against. `baseSha` goes stale once the pipeline rebases the branch, so a
 * worktree task uses where its own change starts (`taskBase`). A current-checkout task uses the tree its last commit
 * left in the working directory, before that the tree it started in.
 */
export async function diffBase(exec: Exec, root: string, task: Task, mergedBase: string | null): Promise<string> {
  if (task.worktree) return taskBase(exec, root, task);
  return mergedBase ?? task.baseTree ?? task.baseSha;
}

/** Whether `ancestor` is an ancestor of (or is) `commit`. Throws when git fails otherwise. */
async function isAncestor(exec: Exec, root: string, ancestor: string, commit: string): Promise<boolean> {
  const result = await exec(["git", "merge-base", "--is-ancestor", ancestor, commit], { cwd: root });
  if (result.exitCode === 0 || result.exitCode === 1) return result.exitCode === 0 && !result.killed;
  throw new Error(`git merge-base --is-ancestor ${ancestor} ${commit}: ${result.output.trim().slice(-ERROR_TAIL_CHARS) || "killed"}`);
}

/** The commit `ref` points at, null without one. */
async function refCommit(exec: Exec, root: string, ref: string): Promise<string | null> {
  const result = await exec(["git", "rev-parse", "--verify", "--quiet", `${ref}^{commit}`], { cwd: root, maxStdoutChars: 100 });
  return result.exitCode === 0 && !result.killed ? result.stdout.trim() : null;
}

/**
 * The commit a worktree task's own change sits on: what is in it is not the task's. A task on its target alone: the
 * merge base of HEAD and the target. A task with its own source: the newest of its upstreams (`next`, then `base`)
 * that HEAD holds, before the first rebase the merge base of HEAD and the source branch it started from.
 */
export async function taskBase(exec: Exec, root: string, task: Task): Promise<string> {
  if (!hasOwnSource(task)) return git(exec, root, ["merge-base", "HEAD", `refs/heads/${task.mainBranch}`]);
  for (const name of ["next", "base"] as const) {
    const upstream = await refCommit(exec, root, upstreamRef(task, name));
    if (upstream && (await isAncestor(exec, root, upstream, "HEAD"))) return upstream;
  }
  const fallback = (await refCommit(exec, root, upstreamRef(task, "base"))) ?? `refs/heads/${task.settings.sourceBranch}`;
  return git(exec, root, ["merge-base", "HEAD", fallback]);
}

/**
 * A task with its own source: the upstream to rebase onto now, the target and the source branch as they are, merged.
 * One that holds the other is the upstream itself. The merge is a commit with the target as first parent; one of the
 * task's upstreams that already merges these two is taken again, so an upstream changes only when a branch moved.
 * Throws when the two do not merge cleanly.
 */
export async function nextUpstream(exec: Exec, root: string, task: Task): Promise<string> {
  const source = task.settings.sourceBranch as string;
  const target = await git(exec, root, ["rev-parse", "--verify", `refs/heads/${task.mainBranch}^{commit}`]);
  const tip = await git(exec, root, ["rev-parse", "--verify", `refs/heads/${source}^{commit}`]);
  if (await isAncestor(exec, root, tip, target)) return target;
  if (await isAncestor(exec, root, target, tip)) return tip;
  for (const name of ["next", "base"] as const) {
    const upstream = await refCommit(exec, root, upstreamRef(task, name));
    if (upstream && (await git(exec, root, ["rev-list", "--parents", "-n", "1", upstream])) === `${upstream} ${target} ${tip}`) return upstream;
  }
  const merged = await exec(["git", "merge-tree", "--write-tree", "--name-only", "--no-messages", target, tip], { cwd: root, maxStdoutChars: GIT_STDOUT_MAX_CHARS });
  const [tree, ...conflicts] = merged.stdout.trim().split("\n");
  if (merged.exitCode === 1 && !merged.killed) throw new Error(`the source branch ${source} does not merge cleanly into ${task.mainBranch}: ${conflicts.join(", ")}`);
  if (merged.exitCode !== 0 || merged.killed) throw new Error(`git merge-tree: ${merged.output.trim().slice(-ERROR_TAIL_CHARS) || "killed"}`);
  return git(exec, root, ["commit-tree", "--no-gpg-sign", tree, "-p", target, "-p", tip, "-m", `Merge branch '${source}' into ${task.mainBranch}`]);
}

/** Whether the task branch no longer holds its target branch, or for a task with its own source its source branch. */
export async function upstreamMoved(exec: Exec, root: string, task: Task): Promise<boolean> {
  const branches = hasOwnSource(task) ? [task.mainBranch, task.settings.sourceBranch as string] : [task.mainBranch];
  for (const branch of branches) if (!(await isAncestor(exec, root, `refs/heads/${branch}`, "HEAD"))) return true;
  return false;
}

const GIT_OPERATIONS = ["MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "rebase-merge", "rebase-apply"];

/** The git operations (merge, cherry-pick, revert, rebase) that are in progress in `root`, by their state path. */
async function operationsInProgress(exec: Exec, root: string): Promise<string[]> {
  const paths = await git(exec, root, ["rev-parse", "--path-format=absolute", ...GIT_OPERATIONS.flatMap((name) => ["--git-path", name])]);
  return GIT_OPERATIONS.filter((_name, index) => existsSync(paths.split("\n")[index]));
}

/** Throws unless `root` is on `branch` with no merge, cherry-pick, revert or rebase in progress. */
export async function assertCheckoutReady(exec: Exec, root: string, branch: string): Promise<void> {
  // First: a rebase in progress also detaches HEAD, and the operation is the useful thing to name.
  const operations = await operationsInProgress(exec, root);
  if (operations.length > 0) throw new Error(`a git operation is in progress in the task checkout (${operations.join(", ")})`);
  const current = await git(exec, root, ["rev-parse", "--abbrev-ref", "HEAD"]);
  if (current !== branch) throw new Error(`the task checkout is on ${current === "HEAD" ? "a detached HEAD" : current}, not on ${branch}`);
}

/** The conflicting (unmerged) files of a rebase in progress in `root`, or null when no rebase is in progress. */
export async function rebaseInProgress(exec: Exec, root: string): Promise<string[] | null> {
  const operations = await operationsInProgress(exec, root);
  if (!operations.some((name) => name.startsWith("rebase-"))) return null;
  return nulList(await git(exec, root, ["diff", "--name-only", "--diff-filter=U", "-z"]));
}

/**
 * `tree` with `paths` as they are in `base`, a path missing there left out. A current-checkout task whose HEAD moved
 * takes its diff base again this way: its own paths keep the old base, everything else is the tree as it is now.
 */
export async function treeWithPathsOf(exec: Exec, root: string, tree: string, base: string, paths: string[]): Promise<string> {
  if (paths.length === 0) return tree;
  return withScratchIndex(exec, root, async (env) => {
    await git(exec, root, ["read-tree", tree], env);
    await git(exec, root, ["--literal-pathspecs", "rm", "--cached", "--force", "--quiet", "--ignore-unmatch", "-r", "--", ...paths], env);
    const entries = nulList(await git(exec, root, ["--literal-pathspecs", "ls-tree", "-r", "-z", base, "--", ...paths]));
    const cacheinfo = entries.flatMap((entry) => {
      const [meta, path] = entry.split("\t");
      const [mode, , sha] = meta.split(" ");
      return ["--cacheinfo", `${mode},${sha},${path}`];
    });
    if (cacheinfo.length > 0) await git(exec, root, ["update-index", "--add", ...cacheinfo], env);
    return git(exec, root, ["write-tree"], env);
  });
}

/** Aborts a rebase that a killed merge left behind in `root`. `exec` must be one a cancel does not stop. */
export async function abortLeftoverRebase(exec: Exec, root: string): Promise<void> {
  const operations = await operationsInProgress(exec, root);
  if (operations.some((name) => name.startsWith("rebase-"))) await git(exec, root, ["rebase", "--abort"]);
}

export interface TaskDiff {
  /** What the diff is taken against, a commit or a tree. */
  base: string;
  /** The working tree the diff was taken of. */
  tree: string;
  /** Changed paths relative to the repo root. A rename lists both of its paths. */
  paths: string[];
  /** Source first, lockfiles last. Cut at a size no review reads. */
  patch: string;
}

/** The working tree of `root` against `base`, untracked files included. */
export async function taskDiff(exec: Exec, root: string, base: string): Promise<TaskDiff> {
  const tree = await snapshotTree(exec, root);
  const paths = nulList(await git(exec, root, ["diff", "--no-renames", "--name-only", "-z", base, tree]));
  // Fixed prefixes and no external diff tool: the patch must not depend on the user's git config.
  const patchOf = async (pathspecs: string[]) => {
    const argv = ["git", "diff", "--no-color", "--no-ext-diff", "--src-prefix=a/", "--dst-prefix=b/", base, tree, "--", ...pathspecs];
    const result = await exec(argv, { cwd: root, maxStdoutChars: PATCH_KEEP_CHARS });
    if (result.exitCode !== 0 || result.killed) throw new Error(`git diff: ${result.output.trim().slice(-ERROR_TAIL_CHARS) || "killed"}`);
    return result.stdout;
  };
  const source = await patchOf([".", ...LOCKFILES.map((name) => `:(exclude,glob)**/${name}`)]);
  const locks = await patchOf(LOCKFILES.map((name) => `:(glob)**/${name}`));
  return { base, tree, paths, patch: source + locks };
}
