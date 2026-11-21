/**
 * Integration tests of the side-effect modules (`run.ts`, `git.ts`, `merge.ts`) against throwaway git repos in a temp
 * dir. Every git call names its repo with `-C`, the user's global git config is left out.
 */
import { afterAll, beforeEach, describe, expect, test } from "bun:test";
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DEFAULT_PROTECTED, protectedPaths, type FactoryConfig } from "./config.ts";
import { abortLeftoverRebase, diffBase, taskDiff, taskGit } from "./git.ts";
import { mergeTask, type MergeInput } from "./merge.ts";
import { runRound, type PipelineSteps } from "./pipeline.ts";
import { CommandRunner, type Exec } from "./run.ts";
import type { Settings, Snapshot, Task } from "./state.ts";

process.env.GIT_CONFIG_GLOBAL = "/dev/null";
process.env.GIT_CONFIG_SYSTEM = "/dev/null";

const scratch = realpathSync(mkdtempSync(join(tmpdir(), "factory-merge-test-")));
afterAll(() => rmSync(scratch, { recursive: true, force: true }));

const runner = new CommandRunner();
const settings: Settings = { branch: "main", merge: "auto", maxRounds: 3, review: false, models: {} };
const noGates: FactoryConfig = { hooks: { after_code: [], after_merge: [] }, gates: [], protected: DEFAULT_PROTECTED };

function gitIn(repo: string, ...args: string[]): string {
  return execFileSync("git", ["-C", repo, ...args], { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }).trim();
}

let repo: string;
let counter = 0;

/** A fresh repo on `main` with one commit holding `README.md` and `factory.yaml`. */
beforeEach(() => {
  counter += 1;
  repo = join(scratch, `repo-${counter}`);
  mkdirSync(repo);
  gitIn(repo, "init", "--quiet", "--initial-branch", "main");
  gitIn(repo, "config", "user.name", "Factory Test");
  gitIn(repo, "config", "user.email", "factory@test.invalid");
  writeFileSync(join(repo, "README.md"), "readme\n");
  writeFileSync(join(repo, "factory.yaml"), "gates: []\n");
  gitIn(repo, "add", "-A");
  gitIn(repo, "commit", "--quiet", "-m", "base");
});

/** A linked worktree of `repo` on a new branch, as a worktree task gets one. */
function addWorktree(branch: string): string {
  const path = join(scratch, `wt-${counter}-${branch}`);
  gitIn(repo, "worktree", "add", "--quiet", "-b", branch, path);
  return path;
}

async function taskIn(root: string, title = "Task title"): Promise<Task> {
  return { title, request: "do it", settings, ...(await taskGit(runner.run, root, "main")) };
}

function input(root: string, task: Task, config: FactoryConfig, round = 1): MergeInput {
  return { exec: runner.run, cleanup: runner.cleanup, root, round, task, config, base: task.baseTree ?? task.baseSha, checkpoint: null, intent: () => undefined };
}

describe("CommandRunner", () => {
  test("reports exit code, stdout and the combined output", async () => {
    const result = await runner.run(["sh", "-c", "echo out; echo err >&2; exit 3"], { cwd: scratch, maxStdoutChars: 100 });
    expect(result).toMatchObject({ exitCode: 3, killed: false, timedOut: false, stdout: "out\n", stdoutTruncated: false });
    expect(result.output).toContain("err");
  });

  test("a command that cannot start has no exit code", async () => {
    const result = await runner.run(["factory-no-such-command"], { cwd: scratch });
    expect(result.exitCode).toBeNull();
    expect(result.output).toContain("factory-no-such-command");
  });

  test("cancel kills the whole process group and never reports exit 0", async () => {
    const own = new CommandRunner();
    const pidFile = join(scratch, `grandchild-${counter}.pid`);
    const running = own.run(["sh", "-c", `sleep 60 & echo $! > ${pidFile}; wait`], { cwd: scratch });
    while (!existsSync(pidFile) || readFileSync(pidFile, "utf8").trim() === "") await Bun.sleep(20);
    const grandchild = Number(readFileSync(pidFile, "utf8"));
    own.cancel();
    const result = await running;
    expect(result).toMatchObject({ exitCode: null, killed: true });
    await Bun.sleep(100);
    expect(() => process.kill(grandchild, 0)).toThrow();
    expect(await own.run(["true"], { cwd: scratch })).toMatchObject({ exitCode: null, killed: true });
    own.reset();
    expect(await own.run(["true"], { cwd: scratch })).toMatchObject({ exitCode: 0, killed: false });
  });

  test("keeps the end of the output and only the asked start of stdout, whatever the command prints", async () => {
    const chatty = ["sh", "-c", "head -c 3000000 /dev/zero | tr '\\0' x; echo; echo end"];
    const unkept = await runner.run(chatty, { cwd: scratch });
    expect(unkept.stdout).toBe("");
    expect(unkept.output.length).toBe(64_000);
    expect(unkept.output.endsWith("x\nend\n")).toBe(true);
    const kept = await runner.run(chatty, { cwd: scratch, maxStdoutChars: 1000 });
    expect(kept).toMatchObject({ exitCode: 0, stdout: "x".repeat(1000), stdoutTruncated: true });
  });

  test("a multi-byte character split over two chunks stays whole", async () => {
    const result = await runner.run(["sh", "-c", "printf '\\303'; sleep 0.2; printf '\\244\\n'"], { cwd: scratch, maxStdoutChars: 10 });
    expect(result.stdout).toBe("ä\n");
  });

  test("a command that leaves a daemon holding its pipes ends when the command exits", async () => {
    const result = await runner.run(["sh", "-c", "setsid sleep 30 & echo $!"], { cwd: scratch, maxStdoutChars: 100 });
    process.kill(Number(result.stdout), "SIGKILL");
    expect(result).toMatchObject({ exitCode: 0, killed: false });
    expect(result.durationMs).toBeLessThan(5000);
  });

  test("a group member that ignores SIGTERM does not outlive a cancelled command", async () => {
    const own = new CommandRunner();
    const pidFile = join(scratch, `stubborn-${counter}.pid`);
    const running = own.run(["sh", "-c", `(trap "" TERM; echo $$ > /dev/null; sleep 60) & echo $! > ${pidFile}; wait`], { cwd: scratch });
    while (!existsSync(pidFile) || readFileSync(pidFile, "utf8").trim() === "") await Bun.sleep(20);
    await Bun.sleep(100);
    own.cancel();
    expect(await running).toMatchObject({ exitCode: null, killed: true });
    await Bun.sleep(100);
    expect(() => process.kill(Number(readFileSync(pidFile, "utf8")), 0)).toThrow();
  });

  test("cleanup runs after a cancel and is not killed by one", async () => {
    const own = new CommandRunner();
    own.cancel();
    const cleanup = own.cleanup(["sh", "-c", "sleep 0.2; echo cleaned"], { cwd: scratch, maxStdoutChars: 100 });
    own.cancel();
    expect(await cleanup).toMatchObject({ exitCode: 0, killed: false, stdout: "cleaned\n" });
  });

  test("a timeout kills the command and marks it", async () => {
    const result = await runner.run(["sleep", "60"], { cwd: scratch, timeoutMs: 100 });
    expect(result).toMatchObject({ exitCode: null, killed: true, timedOut: true });
  });
});

describe("taskGit and taskDiff", () => {
  test("a current-checkout task is its own main branch", async () => {
    expect(await taskGit(runner.run, repo, "elsewhere")).toEqual({ worktree: false, branch: "main", mainBranch: "main", baseSha: gitIn(repo, "rev-parse", "HEAD"), baseTree: gitIn(repo, "rev-parse", "HEAD^{tree}") });
  });

  test("the base tree of a current-checkout task is its working tree, and taking it leaves index and files alone", async () => {
    writeFileSync(join(repo, "README.md"), "user edit\n");
    writeFileSync(join(repo, "notes.txt"), "user notes\n");
    writeFileSync(join(repo, "staged.txt"), "user staged\n");
    gitIn(repo, "add", "staged.txt");
    const { baseTree } = await taskGit(runner.run, repo, "main");
    expect(gitIn(repo, "ls-tree", "-r", "--name-only", baseTree!)).toBe("README.md\nfactory.yaml\nnotes.txt\nstaged.txt");
    expect(gitIn(repo, "show", `${baseTree}:README.md`)).toBe("user edit");
    expect(gitIn(repo, "status", "--porcelain")).toBe("M README.md\nA  staged.txt\n?? notes.txt");
  });

  test("a linked worktree is a worktree task whose main branch is the target branch, which must exist", async () => {
    gitIn(repo, "branch", "release");
    const worktree = addWorktree("task-a");
    expect(await taskGit(runner.run, worktree, "release")).toEqual({ worktree: true, branch: "task-a", mainBranch: "release", baseSha: gitIn(repo, "rev-parse", "HEAD"), baseTree: null });
    await expect(taskGit(runner.run, worktree, "nowhere")).rejects.toThrow("the target branch nowhere does not exist");
  });

  test("refuses a directory that is not the root of a checkout", async () => {
    mkdirSync(join(repo, "sub"));
    await expect(taskGit(runner.run, join(repo, "sub"), "main")).rejects.toThrow("is not the root of its git checkout");
    await expect(taskGit(runner.run, scratch, "main")).rejects.toThrow("git rev-parse");
  });

  test("the diff holds tracked changes and untracked files, and leaves the index alone", async () => {
    const task = await taskIn(repo);
    writeFileSync(join(repo, "README.md"), "readme changed\n");
    writeFileSync(join(repo, "new.txt"), "brand new\n");
    const diff = await taskDiff(runner.run, repo, await diffBase(runner.run, repo, task, null));
    expect(diff.base).toBe(task.baseTree!);
    expect(diff.paths.sort()).toEqual(["README.md", "new.txt"]);
    expect(diff.patch).toContain("+readme changed");
    expect(diff.patch).toContain("+brand new");
    expect(gitIn(repo, "status", "--porcelain")).toBe("M README.md\n?? new.txt");
  });

  test("the patch puts source before lockfiles", async () => {
    const task = await taskIn(repo);
    mkdirSync(join(repo, "zz"));
    writeFileSync(join(repo, "zz", "source.ts"), "export const answer = 42;\n");
    writeFileSync(join(repo, "bun.lock"), "lock body\n");
    mkdirSync(join(repo, "app"));
    writeFileSync(join(repo, "app", "package-lock.json"), "{}\n");
    const diff = await taskDiff(runner.run, repo, task.baseTree!);
    expect(diff.paths).toEqual(["app/package-lock.json", "bun.lock", "zz/source.ts"]);
    const at = (text: string) => diff.patch.indexOf(text);
    expect(at("b/zz/source.ts")).toBeGreaterThanOrEqual(0);
    expect(at("b/zz/source.ts")).toBeLessThan(at("b/app/package-lock.json"));
    expect(at("b/app/package-lock.json")).toBeLessThan(at("b/bun.lock"));
    expect(diff.patch.endsWith("+lock body\n")).toBe(true);
  });

  test("a worktree task's diff base follows a rebase onto the moved main branch", async () => {
    const worktree = addWorktree("task-b");
    const task = await taskIn(worktree);
    writeFileSync(join(repo, "other.txt"), "someone else\n");
    gitIn(repo, "add", "-A");
    gitIn(repo, "commit", "--quiet", "-m", "other task");
    writeFileSync(join(worktree, "mine.txt"), "mine\n");
    expect(await diffBase(runner.run, worktree, task, null)).toBe(task.baseSha);
    gitIn(worktree, "rebase", "--quiet", "--autostash", "main");
    const diff = await taskDiff(runner.run, worktree, await diffBase(runner.run, worktree, task, null));
    expect(diff.paths).toEqual(["mine.txt"]);
  });
});

describe("mergeTask, worktree task", () => {
  test("commits with the title, rebases onto main, runs the gates again and fast-forwards the primary", async () => {
    const worktree = addWorktree("task-c");
    const task = await taskIn(worktree, "Add feature");
    writeFileSync(join(repo, "other.txt"), "someone else\n");
    gitIn(repo, "add", "-A");
    gitIn(repo, "commit", "--quiet", "-m", "other task");
    writeFileSync(join(worktree, "feature.txt"), "feature\n");
    const config: FactoryConfig = { ...noGates, gates: [["test", "-f", "other.txt"], ["test", "-f", "feature.txt"]], hooks: { after_code: [], after_merge: [["touch", "synced"]] } };

    const outcome = await mergeTask(input(worktree, task, config, 2));

    expect(outcome.merge).toEqual({ ok: true, kind: "merge", commit: gitIn(repo, "rev-parse", "HEAD"), error: null, depsSynced: false, tree: null, mixed: [] });
    expect(outcome.gates.map((gate) => [gate.round, gate.phase, gate.argv.join(" "), gate.exitCode])).toEqual([
      [2, "merge_gates", "test -f other.txt", 0],
      [2, "merge_gates", "test -f feature.txt", 0],
    ]);
    expect(gitIn(repo, "log", "--format=%s")).toBe("Add feature\nother task\nbase");
    expect(readFileSync(join(repo, "feature.txt"), "utf8")).toBe("feature\n");
    expect(existsSync(join(repo, ".git", "factory-merge.lock"))).toBe(true);
    expect(existsSync(join(repo, "synced"))).toBe(false);
  });

  test("a merge that changes a lockfile runs after_merge in the primary checkout", async () => {
    const worktree = addWorktree("task-d");
    const task = await taskIn(worktree);
    mkdirSync(join(worktree, "pi-host"));
    writeFileSync(join(worktree, "pi-host", "bun.lock"), "lock\n");
    const config: FactoryConfig = { ...noGates, hooks: { after_code: [], after_merge: [["touch", "synced"]] } };

    const outcome = await mergeTask(input(worktree, task, config, 1));

    expect(outcome.merge).toMatchObject({ ok: true, depsSynced: true, error: null });
    expect(outcome.gates).toMatchObject([{ round: 1, phase: "after_merge", argv: ["touch", "synced"], exitCode: 0 }]);
    expect(existsSync(join(repo, "synced"))).toBe(true);
    expect(existsSync(join(worktree, "synced"))).toBe(false);
  });

  test("a failing after_merge hook leaves the merge done and reports the hook", async () => {
    const worktree = addWorktree("task-e");
    const task = await taskIn(worktree);
    writeFileSync(join(worktree, "uv.lock"), "lock\n");
    const config: FactoryConfig = { ...noGates, hooks: { after_code: [], after_merge: [["sh", "-c", "echo no network >&2; exit 1"]] } };

    const outcome = await mergeTask(input(worktree, task, config, 1));

    expect(outcome.merge).toMatchObject({ ok: true, depsSynced: false, commit: gitIn(repo, "rev-parse", "HEAD") });
    expect(outcome.merge.error).toContain("after_merge failed: sh -c echo no network >&2; exit 1: no network");
    expect(outcome.gates).toMatchObject([{ phase: "after_merge", exitCode: 1, outputTail: "no network\n" }]);
  });

  test("refuses when the main branch moved after the rebase: no fast-forward, main untouched", async () => {
    const worktree = addWorktree("task-f");
    const task = await taskIn(worktree);
    gitIn(repo, "commit", "--quiet", "--allow-empty", "-m", "earlier");
    writeFileSync(join(worktree, "feature.txt"), "feature\n");
    // The merge gate plays a second task that merges while this one's gates run.
    const config: FactoryConfig = { ...noGates, gates: [["git", "-C", repo, "commit", "--quiet", "--allow-empty", "-m", "raced"]] };

    const outcome = await mergeTask(input(worktree, task, config, 1));

    expect(outcome.merge).toMatchObject({ ok: false, kind: "merge", commit: null, depsSynced: false });
    expect(outcome.merge.error).toContain("update of main refused");
    expect(gitIn(repo, "log", "--format=%s")).toBe("raced\nearlier\nbase");
    expect(existsSync(join(repo, "feature.txt"))).toBe(false);
  });

  test("a rebase conflict is aborted and reported with its files, a failing merge gate stops before the merge", async () => {
    const worktree = addWorktree("task-g");
    const task = await taskIn(worktree);
    writeFileSync(join(repo, "README.md"), "main version\n");
    gitIn(repo, "commit", "--quiet", "-am", "main edit");
    writeFileSync(join(worktree, "README.md"), "task version\n");

    const conflict = await mergeTask(input(worktree, task, noGates));
    expect(conflict.merge.ok).toBe(false);
    expect(conflict.conflicts).toEqual(["README.md"]);
    expect(gitIn(worktree, "status", "--porcelain")).toBe("");
    expect(gitIn(worktree, "symbolic-ref", "--short", "HEAD")).toBe("task-g");
    expect(readFileSync(join(worktree, "README.md"), "utf8")).toBe("task version\n");

    const other = addWorktree("task-h");
    gitIn(repo, "commit", "--quiet", "--allow-empty", "-m", "moved");
    writeFileSync(join(other, "feature.txt"), "feature\n");
    const gated = await mergeTask(input(other, await taskIn(other), { ...noGates, gates: [["false"]] }));
    expect(gated.merge).toMatchObject({ ok: false, error: "merge gate failed: false" });
    expect(gated.gates).toHaveLength(1);
    expect(gitIn(repo, "log", "--format=%s")).toBe("moved\nmain edit\nbase");
  });

  test("a target checked out nowhere is moved by compare-and-swap: no working tree touched, after_merge skipped", async () => {
    const worktree = addWorktree("task-i");
    const task = await taskIn(worktree);
    gitIn(repo, "commit", "--quiet", "--allow-empty", "-m", "main moved");
    gitIn(repo, "switch", "--quiet", "-c", "elsewhere");
    writeFileSync(join(worktree, "uv.lock"), "lock\n");
    const config: FactoryConfig = { ...noGates, hooks: { after_code: [], after_merge: [["touch", "hook-ran"]] } };

    const outcome = await mergeTask(input(worktree, task, config));

    expect(outcome.merge).toMatchObject({ ok: true, kind: "merge", error: null, depsSynced: false });
    expect(gitIn(repo, "rev-parse", "main")).toBe(gitIn(worktree, "rev-parse", "HEAD"));
    expect(gitIn(repo, "log", "--format=%s", "main")).toBe("Task title\nmain moved\nbase");
    expect(gitIn(repo, "symbolic-ref", "--short", "HEAD")).toBe("elsewhere");
    expect(existsSync(join(repo, "uv.lock"))).toBe(false);
    expect(existsSync(join(repo, "hook-ran"))).toBe(false);
    expect(outcome.gates).toEqual([]);
  });

  test("a target checked out in another worktree is fast-forwarded there: its files updated, after_merge skipped", async () => {
    const worktree = addWorktree("task-o");
    const task = await taskIn(worktree);
    gitIn(repo, "switch", "--quiet", "-c", "elsewhere");
    const other = join(scratch, `wt-${counter}-main`);
    gitIn(repo, "worktree", "add", "--quiet", other, "main");
    writeFileSync(join(worktree, "feature.txt"), "feature\n");
    writeFileSync(join(worktree, "uv.lock"), "lock\n");
    const config: FactoryConfig = { ...noGates, hooks: { after_code: [], after_merge: [["touch", "hook-ran"]] } };

    const outcome = await mergeTask(input(worktree, task, config));

    expect(outcome.merge).toMatchObject({ ok: true, commit: gitIn(worktree, "rev-parse", "HEAD"), error: null, depsSynced: false });
    expect(gitIn(other, "log", "--format=%s")).toBe("Task title\nbase");
    expect(readFileSync(join(other, "feature.txt"), "utf8")).toBe("feature\n");
    expect(gitIn(other, "status", "--porcelain")).toBe("");
    expect(existsSync(join(other, "hook-ran"))).toBe(false);
    expect(existsSync(join(repo, "feature.txt"))).toBe(false);
    expect(outcome.gates).toEqual([]);
  });

  test("a worktree holding the target whose uncommitted change the merge would overwrite refuses it: nothing moved", async () => {
    const worktree = addWorktree("task-q");
    const task = await taskIn(worktree);
    gitIn(repo, "switch", "--quiet", "-c", "elsewhere");
    const other = join(scratch, `wt-${counter}-main`);
    gitIn(repo, "worktree", "add", "--quiet", other, "main");
    writeFileSync(join(worktree, "README.md"), "task version\n");
    writeFileSync(join(other, "README.md"), "user work in the other worktree\n");

    const outcome = await mergeTask(input(worktree, task, noGates));

    expect(outcome.merge).toMatchObject({ ok: false, kind: "merge", commit: null });
    expect(outcome.merge.error).toContain("update of main refused");
    expect(outcome.merge.error).toContain("README.md");
    expect(readFileSync(join(other, "README.md"), "utf8")).toBe("user work in the other worktree\n");
    expect(gitIn(repo, "log", "--format=%s", "main")).toBe("base");
  });

  test("two tasks on the same target checked out in another worktree merge one after the other, the second on top", async () => {
    const first = addWorktree("task-r");
    const second = addWorktree("task-s");
    const firstTask = await taskIn(first, "First");
    const secondTask = await taskIn(second, "Second");
    gitIn(repo, "switch", "--quiet", "-c", "elsewhere");
    const other = join(scratch, `wt-${counter}-main`);
    gitIn(repo, "worktree", "add", "--quiet", other, "main");
    writeFileSync(join(first, "first.txt"), "first\n");
    writeFileSync(join(second, "second.txt"), "second\n");

    const one = await mergeTask(input(first, firstTask, noGates));
    const two = await mergeTask(input(second, secondTask, noGates));

    expect(one.merge.ok).toBe(true);
    expect(two.merge).toMatchObject({ ok: true, commit: gitIn(other, "rev-parse", "HEAD") });
    expect(gitIn(other, "log", "--format=%s")).toBe("Second\nFirst\nbase");
    expect(readFileSync(join(other, "first.txt"), "utf8")).toBe("first\n");
    expect(readFileSync(join(other, "second.txt"), "utf8")).toBe("second\n");
    expect(gitIn(other, "status", "--porcelain")).toBe("");
  });
});

describe("mergeTask, worktree task: what a merge must not get wrong", () => {
  test("skips the merge gates when the rebase did not move HEAD", async () => {
    const worktree = addWorktree("task-k");
    const task = await taskIn(worktree);
    writeFileSync(join(worktree, "feature.txt"), "feature\n");
    const outcome = await mergeTask(input(worktree, task, { ...noGates, gates: [["false"]] }));
    expect(outcome).toMatchObject({ gates: [], merge: { ok: true, commit: gitIn(repo, "rev-parse", "HEAD") } });
    expect(gitIn(repo, "log", "--format=%s")).toBe("Task title\nbase");
  });

  test("a cancel during the rebase still aborts it, and a retry merges from the task branch", async () => {
    const worktree = addWorktree("task-l");
    const task = await taskIn(worktree);
    writeFileSync(join(repo, "README.md"), "main version\n");
    gitIn(repo, "commit", "--quiet", "-am", "main edit");
    writeFileSync(join(worktree, "README.md"), "task version\n");
    // The rebase stops on the conflict, and the cancel arrives before the pipeline can clean up.
    const own = new CommandRunner();
    const exec: Exec = async (argv, options) => {
      const result = await own.run(argv, options);
      if (argv[1] === "rebase") own.cancel();
      return result;
    };

    const cancelled = await mergeTask({ ...input(worktree, task, noGates), exec, cleanup: own.cleanup });

    expect(cancelled.merge.ok).toBe(false);
    expect(existsSync(gitIn(worktree, "rev-parse", "--path-format=absolute", "--git-path", "rebase-merge"))).toBe(false);
    expect(gitIn(worktree, "symbolic-ref", "--short", "HEAD")).toBe("task-l");

    // Retry: the coder moves its work onto main and resolves the conflict by hand.
    gitIn(worktree, "reset", "--quiet", "--soft", "main");
    writeFileSync(join(worktree, "README.md"), "main and task version\n");
    writeFileSync(join(worktree, "feature.txt"), "feature\n");
    const retried = await mergeTask(input(worktree, task, noGates));
    expect(retried.merge).toMatchObject({ ok: true, commit: gitIn(repo, "rev-parse", "HEAD") });
    expect(readFileSync(join(repo, "feature.txt"), "utf8")).toBe("feature\n");
  });

  test("abortLeftoverRebase puts a worktree a killed process left mid-rebase back on its branch", async () => {
    const worktree = addWorktree("task-m");
    writeFileSync(join(repo, "README.md"), "main version\n");
    gitIn(repo, "commit", "--quiet", "-am", "main edit");
    writeFileSync(join(worktree, "README.md"), "task version\n");
    gitIn(worktree, "commit", "--quiet", "-am", "task edit");
    const tip = gitIn(worktree, "rev-parse", "HEAD");
    expect(() => gitIn(worktree, "rebase", "--quiet", "main")).toThrow();
    await abortLeftoverRebase(runner.cleanup, worktree);
    expect(gitIn(worktree, "symbolic-ref", "--short", "HEAD")).toBe("task-m");
    expect(gitIn(worktree, "rev-parse", "HEAD")).toBe(tip);
  });

  test("a rebase in progress at merge start fails the merge and keeps the coder's resolution and edits", async () => {
    const worktree = addWorktree("task-r");
    const task = await taskIn(worktree);
    writeFileSync(join(repo, "README.md"), "main version\n");
    gitIn(repo, "commit", "--quiet", "-am", "main edit");
    writeFileSync(join(worktree, "README.md"), "task version\n");
    gitIn(worktree, "commit", "--quiet", "-am", "task edit");
    expect(() => gitIn(worktree, "rebase", "--quiet", "main")).toThrow();
    // The coder resolved the conflict and edited another file, but did not continue the rebase.
    writeFileSync(join(worktree, "README.md"), "resolved\n");
    writeFileSync(join(worktree, "factory.yaml"), "gates: [[true]]\n");

    const outcome = await mergeTask(input(worktree, task, noGates));

    expect(outcome.merge.error).toBe("a git operation is in progress in the task checkout (rebase-merge)");
    expect(readFileSync(join(worktree, "README.md"), "utf8")).toBe("resolved\n");
    expect(readFileSync(join(worktree, "factory.yaml"), "utf8")).toBe("gates: [[true]]\n");
    expect(gitIn(repo, "log", "--format=%s", "main")).toBe("main edit\nbase");
  });

  test("a cancel reaches the wait for a merge lock someone else holds: nothing is merged, also once the lock frees", async () => {
    const worktree = addWorktree("task-w");
    const task = await taskIn(worktree);
    writeFileSync(join(worktree, "feature.txt"), "feature\n");
    const lock = join(repo, ".git", "factory-merge.lock");
    const holder = Bun.spawn(["flock", lock, "sleep", "30"]);
    await Bun.sleep(200);
    const own = new CommandRunner();
    const intents: unknown[] = [];
    setTimeout(() => own.cancel(), 1500);
    const started = Date.now();
    const outcome = await mergeTask({ ...input(worktree, task, noGates), exec: own.run, cleanup: own.cleanup, intent: (data) => intents.push(data) });
    holder.kill();
    await holder.exited;
    expect(Date.now() - started).toBeLessThan(4000);
    expect(outcome.merge).toMatchObject({ ok: false, error: "cancelled while waiting for the merge lock" });
    expect(intents).toEqual([{ kind: "merge", commit: gitIn(worktree, "rev-parse", "HEAD"), before: task.baseSha }]);
    await Bun.sleep(300);
    expect(gitIn(repo, "rev-parse", "main")).toBe(task.baseSha);
  });

  test("untracked nested repositories are left out of the diff and the commit", async () => {
    const worktree = addWorktree("task-n");
    const task = await taskIn(worktree);
    mkdirSync(join(worktree, "empty"));
    gitIn(join(worktree, "empty"), "init", "--quiet");
    mkdirSync(join(worktree, "sub"));
    gitIn(join(worktree, "sub"), "init", "--quiet");
    writeFileSync(join(worktree, "sub", "x"), "x\n");
    gitIn(join(worktree, "sub"), "add", "x");
    gitIn(join(worktree, "sub"), "-c", "user.name=a", "-c", "user.email=a@b.invalid", "commit", "--quiet", "-m", "x");
    writeFileSync(join(worktree, "feature.txt"), "feature\n");
    expect((await taskDiff(runner.run, worktree, task.baseSha)).paths).toEqual(["feature.txt"]);
    const outcome = await mergeTask(input(worktree, task, noGates));
    expect(outcome.merge.ok).toBe(true);
    expect(gitIn(repo, "show", "--format=", "--name-only", "main")).toBe("feature.txt");
  });

  test("the primary switched to another branch during the merge gates: main is moved by ref, the primary's tree untouched", async () => {
    const worktree = addWorktree("task-n");
    const task = await taskIn(worktree);
    gitIn(repo, "commit", "--quiet", "--allow-empty", "-m", "moved");
    const main = gitIn(repo, "rev-parse", "main");
    writeFileSync(join(worktree, "feature.txt"), "feature\n");
    const config: FactoryConfig = { ...noGates, gates: [["git", "-C", repo, "switch", "--quiet", "-c", "wip"]] };

    const outcome = await mergeTask(input(worktree, task, config));

    expect(outcome.merge).toMatchObject({ ok: true, commit: gitIn(worktree, "rev-parse", "HEAD"), depsSynced: false });
    expect(gitIn(repo, "rev-parse", "main^")).toBe(main);
    expect(gitIn(repo, "rev-parse", "wip")).toBe(main);
    expect(existsSync(join(repo, "feature.txt"))).toBe(false);
  });

  test("a primary checkout whose uncommitted work the merge would overwrite refuses the fast-forward", async () => {
    const worktree = addWorktree("task-p");
    const task = await taskIn(worktree);
    writeFileSync(join(worktree, "README.md"), "task version\n");
    writeFileSync(join(repo, "README.md"), "user work in the primary\n");
    const outcome = await mergeTask(input(worktree, task, noGates));
    expect(outcome.merge.ok).toBe(false);
    expect(outcome.merge.error).toContain("update of main refused");
    expect(readFileSync(join(repo, "README.md"), "utf8")).toBe("user work in the primary\n");
    expect(gitIn(repo, "log", "--format=%s")).toBe("base");
  });

  test("a worktree that left the task branch is not merged and not reported as merged", async () => {
    const worktree = addWorktree("task-o");
    const task = await taskIn(worktree);
    gitIn(worktree, "switch", "--quiet", "-c", "fix-x");
    writeFileSync(join(worktree, "feature.txt"), "feature\n");
    const onOther = await mergeTask(input(worktree, task, noGates));
    expect(onOther.merge).toMatchObject({ ok: false, commit: null, error: "the task checkout is on fix-x, not on task-o" });
    gitIn(worktree, "switch", "--quiet", "--detach");
    const detached = await mergeTask(input(worktree, task, noGates));
    expect(detached.merge.error).toBe("the task checkout is on a detached HEAD, not on task-o");
    expect(gitIn(worktree, "status", "--porcelain")).toBe("?? feature.txt");
    expect(gitIn(repo, "log", "--format=%s")).toBe("base");
  });
});

describe("mergeTask, current-checkout task in a tree the user was working in", () => {
  /** The user's work in progress: an edit to a tracked file, to `factory.yaml`, an untracked file and a staged one. */
  function userEdits(): Record<string, string> {
    const edits = { "README.md": "user readme\n", "factory.yaml": "gates: [[user, tuning]]\n", "notes.txt": "user notes\n", "staged.txt": "user staged\n" };
    for (const [path, content] of Object.entries(edits)) writeFileSync(join(repo, path), content);
    gitIn(repo, "add", "staged.txt");
    return edits;
  }
  const userStatus = "M README.md\n M factory.yaml\nA  staged.txt\n?? notes.txt";

  test("the commit holds only the task's file, the user's edits stay uncommitted, staged as they were and byte-identical", async () => {
    const edits = userEdits();
    const task = await taskIn(repo, "Add task file");
    writeFileSync(join(repo, "task.txt"), "task\n");
    expect((await taskDiff(runner.run, repo, task.baseTree!)).paths).toEqual(["task.txt"]);

    const outcome = await mergeTask(input(repo, task, noGates));

    expect(outcome.merge).toMatchObject({ ok: true, kind: "commit", commit: gitIn(repo, "rev-parse", "HEAD") });
    expect(gitIn(repo, "show", "--format=%s", "--name-only", "HEAD")).toBe("Add task file\n\ntask.txt");
    expect(gitIn(repo, "status", "--porcelain")).toBe(userStatus);
    for (const [path, content] of Object.entries(edits)) expect(readFileSync(join(repo, path), "utf8")).toBe(content);
  });

  test("after the commit a message round's diff holds only the new change", async () => {
    userEdits();
    const task = await taskIn(repo);
    writeFileSync(join(repo, "task.txt"), "task\n");
    const { merge } = await mergeTask(input(repo, task, noGates));
    expect(merge.tree).not.toBe(gitIn(repo, "rev-parse", `${merge.commit}^{tree}`));
    expect(gitIn(repo, "ls-tree", "-r", "--name-only", merge.tree!)).toBe("README.md\nfactory.yaml\nnotes.txt\nstaged.txt\ntask.txt");

    writeFileSync(join(repo, "task.txt"), "task, second round\n");
    const diff = await taskDiff(runner.run, repo, await diffBase(runner.run, repo, task, merge.tree));
    expect(diff.paths).toEqual(["task.txt"]);
    expect(diff.patch).toContain("+task, second round");
    expect(protectedPaths(diff.paths, noGates.protected)).toEqual([]);

    const second = await mergeTask({ ...input(repo, task, noGates), base: merge.tree! });
    expect(gitIn(repo, "show", "--format=", "--name-only", second.merge.commit!)).toBe("task.txt");
    expect(gitIn(repo, "status", "--porcelain")).toBe(userStatus);
  });

  test("a file of the user that the task deleted, and one it changed, are committed as the task left them", async () => {
    writeFileSync(join(repo, "notes.txt"), "user notes\n");
    const task = await taskIn(repo);
    rmSync(join(repo, "notes.txt"));
    rmSync(join(repo, "README.md"));
    const outcome = await mergeTask(input(repo, task, noGates));
    expect(outcome.merge.ok).toBe(true);
    expect(gitIn(repo, "show", "--format=", "--name-status", "HEAD")).toBe("D\tREADME.md");
    expect(gitIn(repo, "status", "--porcelain")).toBe("");
  });

  test("a cancel that arrives during the commit still leaves the user's index in line with it", async () => {
    const task = await taskIn(repo);
    writeFileSync(join(repo, "README.md"), "task readme\n");
    writeFileSync(join(repo, "task.txt"), "task\n");
    const own = new CommandRunner();
    const exec: Exec = async (argv, options) => {
      const result = await own.run(argv, options);
      if (argv[1] === "commit-tree") own.cancel();
      return result;
    };
    const outcome = await mergeTask({ ...input(repo, task, noGates), exec, cleanup: own.cleanup });
    expect(outcome.merge).toMatchObject({ ok: true, commit: gitIn(repo, "rev-parse", "HEAD") });
    expect(gitIn(repo, "log", "--format=%s")).toBe("Task title\nbase");
    expect(gitIn(repo, "status", "--porcelain")).toBe("");
  });

  test("a HEAD that moves while the commit is built fails the merge: the other commit is not reverted", async () => {
    const task = await taskIn(repo);
    writeFileSync(join(repo, "task.txt"), "task\n");
    const exec: Exec = async (argv, options) => {
      // Someone else commits right before the task's commit object is written.
      if (argv[1] === "commit-tree") gitIn(repo, "commit", "--quiet", "--allow-empty", "-m", "other");
      return runner.run(argv, options);
    };
    const intents: unknown[] = [];
    const outcome = await mergeTask({ ...input(repo, task, noGates), exec, intent: (data) => intents.push(data) });
    expect(outcome.merge).toMatchObject({ ok: false, kind: "commit", commit: null });
    expect(outcome.merge.error).toStartWith(`HEAD moved away from ${task.baseSha}`);
    expect(intents).toMatchObject([{ kind: "commit", before: task.baseSha }]);
    expect(gitIn(repo, "log", "--format=%s")).toBe("other\nbase");
    expect(gitIn(repo, "status", "--porcelain")).toBe("?? task.txt");
  });

  test("a task path that changed since the round's tree fails the merge, a path the user also had changed is reported as mixed", async () => {
    writeFileSync(join(repo, "README.md"), "user readme\n");
    const task = await taskIn(repo);
    writeFileSync(join(repo, "README.md"), "user readme\ntask line\n");
    writeFileSync(join(repo, "task.txt"), "task\n");
    const tree = (await taskDiff(runner.run, repo, task.baseTree!)).tree;
    writeFileSync(join(repo, "task.txt"), "the user's later edit\n");
    const changed = await mergeTask({ ...input(repo, task, noGates), checkpoint: { tree, head: task.baseSha } });
    expect(changed.merge.error).toBe("changed since the round ended, by someone else: task.txt");
    expect(gitIn(repo, "log", "--format=%s")).toBe("base");

    writeFileSync(join(repo, "task.txt"), "task\n");
    const committed = await mergeTask({ ...input(repo, task, noGates), checkpoint: { tree, head: task.baseSha } });
    expect(committed.merge).toMatchObject({ ok: true, mixed: ["README.md"] });
  });

  test("refuses to commit on another branch or into a merge in progress", async () => {
    const task = await taskIn(repo);
    writeFileSync(join(repo, "task.txt"), "task\n");
    gitIn(repo, "switch", "--quiet", "-c", "elsewhere");
    const switched = await mergeTask(input(repo, task, noGates));
    expect(switched.merge).toMatchObject({ ok: false, kind: "commit", commit: null, error: "the task checkout is on elsewhere, not on main" });
    gitIn(repo, "switch", "--quiet", "main");
    writeFileSync(gitIn(repo, "rev-parse", "--path-format=absolute", "--git-path", "MERGE_HEAD"), `${task.baseSha}\n`);
    const merging = await mergeTask(input(repo, task, noGates));
    expect(merging.merge.error).toBe("a git operation is in progress in the task checkout (MERGE_HEAD)");
    expect(gitIn(repo, "log", "--all", "--format=%s")).toBe("base");
    expect(gitIn(repo, "status", "--porcelain")).toBe("?? task.txt");
  });
});

describe("mergeTask, current-checkout task", () => {
  test("commits everything but protected paths to the current branch", async () => {
    const task = await taskIn(repo, "Checkout change");
    writeFileSync(join(repo, "README.md"), "readme changed\n");
    writeFileSync(join(repo, "new.txt"), "new\n");
    writeFileSync(join(repo, "factory.yaml"), "gates: [[true]]\n");
    mkdirSync(join(repo, ".factory", "agents"), { recursive: true });
    writeFileSync(join(repo, ".factory", "agents", "coder.md"), "be nice\n");

    const outcome = await mergeTask(input(repo, task, noGates));

    expect(outcome).toEqual({ gates: [], merge: { ok: true, kind: "commit", commit: gitIn(repo, "rev-parse", "HEAD"), error: null, depsSynced: false, tree: expect.any(String), mixed: [] }, conflicts: null });
    expect(gitIn(repo, "log", "--format=%s")).toBe("Checkout change\nbase");
    expect(gitIn(repo, "show", "--format=", "--name-only", "HEAD").split("\n").sort()).toEqual(["README.md", "new.txt"]);
    expect(gitIn(repo, "status", "--porcelain")).toBe("M factory.yaml\n?? .factory/");
  });

  test("nothing to commit is a merge without a new commit", async () => {
    const task = await taskIn(repo);
    const outcome = await mergeTask(input(repo, task, noGates));
    expect(outcome.merge).toMatchObject({ ok: true, kind: "commit", commit: task.baseSha });
    expect(gitIn(repo, "log", "--format=%s")).toBe("base");
  });
});

test("a round whose diff touches a protected path is refused before any merge", async () => {
  const worktree = addWorktree("task-j");
  const task = await taskIn(worktree);
  writeFileSync(join(worktree, "feature.txt"), "feature\n");
  writeFileSync(join(worktree, "factory.yaml"), "gates: [[true]]\n");
  const snapshot: Snapshot = { task, state: "running", round: 1, runFirstRound: 1, reason: "start", mergedBase: null, head: null, roundTree: null, mergeIntent: null, mergeError: null };
  let merges = 0;
  const steps: PipelineSteps = {
    status: () => undefined,
    cancelled: () => false,
    pendingMessages: () => false,
    config: () => noGates,
    runCommand: (argv) => runner.run(argv, { cwd: worktree }),
    diff: async () => taskDiff(runner.run, worktree, await diffBase(runner.run, worktree, task, null)),
    review: async () => {
      throw new Error("review is off");
    },
    merge: (config) => {
      merges += 1;
      return mergeTask(input(worktree, task, config));
    },
    openRebase: async () => null,
  };

  const result = await runRound(snapshot, steps);

  expect(result.continue).toBe(true);
  expect(result.entries[0]).toMatchObject({ customType: "factory-round", data: { round: 2, trigger: "protected" } });
  expect((result.entries[1] as { content: string }).content).toContain("- factory.yaml");
  expect(merges).toBe(0);
  expect(gitIn(repo, "log", "--format=%s")).toBe("base");
  expect(protectedPaths(["factory.yaml"], noGates.protected)).toEqual(["factory.yaml"]);
});
