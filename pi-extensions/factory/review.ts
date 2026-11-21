/**
 * The review step: a fresh, read-only reviewer child that can only answer through `submit_verdict`.
 *
 * - Consumes the task's diff, the reviewer model and role prompt, and the directory for this task's reviews
 * - Provides `runReview` (the `factory-review` data) and the pure parts it is built from: the patch file's text, the
 *   findings check and the reading of the child's `--mode json` stdout
 */
import { mkdir, readdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import type { TaskDiff } from "./git.ts";
import type { Exec } from "./run.ts";
import { approvedFrom, type Finding, type ReviewData, type ReviewStatus, type ReviewUsage, type Severity, type Task } from "./state.ts";

export const VERDICT_TOOL = "submit_verdict";
const REVIEWER_TOOL_EXTENSION = join(import.meta.dirname, "reviewer-tool.ts");
const SEVERITIES: readonly Severity[] = ["blocker", "major", "minor"];
const PATCH_MAX_CHARS = 200_000;
const PATHS_MAX = 500;
const REVIEW_TIMEOUT_MS = 300_000;
/** The child's JSONL stream: the session line, the verdict call and the usage of every message. */
const REVIEWER_STDOUT_MAX_CHARS = 32_000_000;

/**
 * The patch file as the reviewer gets it: every changed path first, then the patch cut at `maxChars` with a line that
 * says so. The list survives any cut, so the reviewer knows which files it has not seen.
 */
export function reviewPatch(diff: Pick<TaskDiff, "paths" | "patch">, maxChars: number): string {
  const listed = diff.paths.slice(0, PATHS_MAX).map((path) => `- ${path}`);
  if (diff.paths.length > PATHS_MAX) listed.push(`- and ${diff.paths.length - PATHS_MAX} more`);
  const header = `Changed files (${diff.paths.length}):\n${listed.join("\n")}\n\n`;
  if (diff.patch.length <= maxChars) return header + diff.patch;
  return (
    `${header}${diff.patch.slice(0, maxChars)}\n\n` +
    `[patch cut here: ${diff.patch.length - maxChars} more characters. Read the files listed at the top that the patch does not show.]\n`
  );
}

/** The request file as the reviewer gets it: the task's request, then every follow-up message of the user in order. */
export function reviewRequest(task: Pick<Task, "title" | "request">, followUps: string[]): string {
  const request = `# ${task.title}\n\n${task.request}\n`;
  if (followUps.length === 0) return request;
  const messages = followUps.map((text, index) => `### Message ${index + 1}\n\n${text}\n`);
  return `${request}\n## Follow-up messages from the user\n\n${messages.join("\n")}`;
}

/** The findings of a `submit_verdict` call, or a string that says what is wrong with `value`. */
export function parseFindings(value: unknown): Finding[] | string {
  if (!Array.isArray(value)) return "findings is not an array";
  const findings: Finding[] = [];
  for (const [index, raw] of value.entries()) {
    if (typeof raw !== "object" || raw === null) return `findings[${index}] is not an object`;
    const finding = raw as Record<string, unknown>;
    if (typeof finding.file !== "string" || finding.file === "") return `findings[${index}].file is not a string`;
    const line = finding.line ?? null;
    if (line !== null && !(typeof line === "number" && Number.isInteger(line))) return `findings[${index}].line is not an integer or null`;
    if (!SEVERITIES.includes(finding.severity as Severity)) return `findings[${index}].severity is not one of ${SEVERITIES.join(", ")}`;
    if (typeof finding.text !== "string" || finding.text === "") return `findings[${index}].text is not a string`;
    findings.push({ file: finding.file, line, severity: finding.severity as Severity, text: finding.text });
  }
  return findings;
}

export interface ReviewerOutput {
  /** Findings of the last valid `submit_verdict` call, null without one. */
  findings: Finding[] | null;
  sessionId: string | null;
  usage: ReviewUsage;
}

/** Reads the reviewer's JSONL stdout. Lines that are not JSON are skipped (a killed run ends mid-line). */
export function readReviewerOutput(stdout: string): ReviewerOutput {
  const output: ReviewerOutput = { findings: null, sessionId: null, usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, cost: 0 } };
  for (const line of stdout.split("\n")) {
    if (line.trim() === "") continue;
    let event: any;
    try {
      event = JSON.parse(line);
    } catch {
      continue;
    }
    if (event.type === "session" && typeof event.id === "string") output.sessionId = event.id;
    if (event.type === "tool_execution_end" && event.toolName === VERDICT_TOOL && !event.isError) {
      const findings = parseFindings(event.result?.details?.findings);
      if (typeof findings !== "string") output.findings = findings;
    }
    if (event.type === "message_end" && event.message?.role === "assistant" && event.message.usage) {
      const usage = event.message.usage;
      output.usage.input += usage.input ?? 0;
      output.usage.output += usage.output ?? 0;
      output.usage.cacheRead += usage.cacheRead ?? 0;
      output.usage.cacheWrite += usage.cacheWrite ?? 0;
      output.usage.cost += usage.cost?.total ?? 0;
    }
  }
  return output;
}

export interface ReviewInput {
  exec: Exec;
  root: string;
  round: number;
  task: Task;
  /** The user's messages to the task after its request (`followUpMessages`): part of the task. */
  followUps: string[];
  diff: TaskDiff;
  /** `provider/model` for the child, null for its own default. */
  model: string | null;
  rolePrompt: string | undefined;
  /** Holds this task's reviewer sessions and patch files. */
  reviewsDir: string;
  cancelled: () => boolean;
}

export async function runReview(input: ReviewInput): Promise<ReviewData> {
  const { exec, root, round, task, model, reviewsDir } = input;
  const result = (status: ReviewStatus, output: ReviewerOutput | null, sessionFile: string | null, wallMs: number): ReviewData => {
    const findings = (status === "ok" && output?.findings) || [];
    return {
      round,
      status,
      approved: approvedFrom(status, findings),
      findings,
      model,
      sessionFile,
      usage: output?.usage ?? { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, cost: 0 },
      wallMs,
    };
  };
  if (input.diff.patch.trim() === "") return result("no_diff", null, null, 0);

  await mkdir(reviewsDir, { recursive: true });
  const patchFile = join(reviewsDir, `round-${round}.patch`);
  await writeFile(patchFile, reviewPatch(input.diff, PATCH_MAX_CHARS));
  // The request goes into a file: as an argument a long one would exceed the argv limit.
  const requestFile = join(reviewsDir, `round-${round}.request.md`);
  await writeFile(requestFile, reviewRequest(task, input.followUps));
  const prompt =
    `Review the change of a task. The task is in ${requestFile}, its diff is the patch file ${patchFile}: read both with the read tool, ` +
    `look at the surrounding code where the diff alone is not enough, then finish with ${VERDICT_TOOL}. ` +
    `Follow-up messages from the user in the task file are part of the task: a change one of them asks for is on-request.`;
  const argv = ["pi", "--mode", "json", "-p", "--tools", `read,grep,find,ls,${VERDICT_TOOL}`, "-ne", "-e", REVIEWER_TOOL_EXTENSION];
  argv.push("--no-skills", "--no-prompt-templates", "--no-themes", "--session-dir", reviewsDir);
  if (model) argv.push("--model", model);
  if (input.rolePrompt) argv.push("--append-system-prompt", input.rolePrompt);
  argv.push(prompt);

  const child = await exec(argv, { cwd: root, timeoutMs: REVIEW_TIMEOUT_MS, maxStdoutChars: REVIEWER_STDOUT_MAX_CHARS });
  const output = readReviewerOutput(child.stdout);
  const sessionName = output.sessionId ? (await readdir(reviewsDir)).find((name) => name.endsWith(`${output.sessionId}.jsonl`)) : undefined;
  const sessionFile = sessionName ? join(reviewsDir, sessionName) : null;
  // A killed or failed run never yields a verdict, even when a verdict call is already in the stream.
  let status: ReviewStatus = "ok";
  if (child.timedOut) status = "timeout";
  else if (child.killed || input.cancelled()) status = "aborted";
  else if (child.exitCode !== 0) status = "failed";
  else if (output.findings === null) status = "malformed";
  return result(status, output, sessionFile, child.durationMs);
}
