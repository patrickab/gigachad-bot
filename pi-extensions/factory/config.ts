/**
 * The repo's side of a factory task.
 *
 * - Consumes `factory.yaml` (the keys the pipeline runs: `hooks.after_code`, `hooks.after_merge`, `gates`, `protected`)
 *   and the role prompts `.factory/agents/<role>.md`, all as the task's base commit holds them. The working tree is
 *   the coder's to edit, so it is never the source of the rules the coder is held to
 * - Provides `TaskFiles` (`loadTaskFiles`), `FactoryConfig`, the protected-path match and the role prompt text
 * - `hooks.after_create` and `preview` belong to the backend and are not read here
 */
import { git } from "./git.ts";
import type { Exec } from "./run.ts";

export interface FactoryConfig {
  hooks: { after_code: string[][]; after_merge: string[][] };
  gates: string[][];
  protected: string[];
}

export const DEFAULT_PROTECTED = ["factory.yaml", ".factory/**"];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function argvList(value: unknown, key: string): string[][] {
  if (value === undefined || value === null) return [];
  const valid =
    Array.isArray(value) &&
    value.every((argv) => Array.isArray(argv) && argv.length > 0 && argv.every((part) => typeof part === "string" || typeof part === "number"));
  if (!valid) throw new Error(`factory.yaml: ${key} must be a list of argv lists`);
  return (value as (string | number)[][]).map((argv) => argv.map(String));
}

/** `raw` is the parsed YAML document. A missing or empty file (`undefined`, `null`) means no hooks and no gates. */
export function parseFactoryConfig(raw: unknown): FactoryConfig {
  if (raw === undefined || raw === null) return { hooks: { after_code: [], after_merge: [] }, gates: [], protected: DEFAULT_PROTECTED };
  if (!isRecord(raw)) throw new Error("factory.yaml: the document must be a mapping");
  const hooks = raw.hooks ?? {};
  if (!isRecord(hooks)) throw new Error("factory.yaml: hooks must be a mapping");
  const globs = raw.protected ?? DEFAULT_PROTECTED;
  if (!Array.isArray(globs) || !globs.every((glob) => typeof glob === "string" && glob !== "")) {
    throw new Error("factory.yaml: protected must be a list of globs");
  }
  return {
    hooks: { after_code: argvList(hooks.after_code, "hooks.after_code"), after_merge: argvList(hooks.after_merge, "hooks.after_merge") },
    gates: argvList(raw.gates, "gates"),
    protected: globs,
  };
}

/**
 * A glob over repo-relative paths with git's `glob` pathspec rules: `*` and `?` stay inside one path segment,
 * a `**` segment spans any number of them.
 */
export function globToRegExp(glob: string): RegExp {
  let source = "";
  for (let index = 0; index < glob.length; index += 1) {
    const char = glob[index];
    if (glob.startsWith("**/", index) && (index === 0 || glob[index - 1] === "/")) {
      source += "(?:.*/)?";
      index += 2;
    } else if (glob.startsWith("**", index)) {
      source += ".*";
      index += 1;
    } else if (char === "*") {
      source += "[^/]*";
    } else if (char === "?") {
      source += "[^/]";
    } else {
      source += char.replace(/[.+^${}()|[\]\\]/g, "\\$&");
    }
  }
  return new RegExp(`^${source}$`);
}

/** The paths of `paths` that a protected glob matches. */
export function protectedPaths(paths: string[], globs: string[]): string[] {
  const patterns = globs.map(globToRegExp);
  return paths.filter((path) => patterns.some((pattern) => pattern.test(path)));
}

/** A role file's prompt: its text without a leading YAML frontmatter block. */
export function rolePromptText(content: string): string {
  return content.replace(/^---\n[\s\S]*?\n---\n/, "").trim();
}

export interface TaskFiles {
  config: FactoryConfig;
  /** Role prompts, undefined for a role the repo has no file for. */
  prompts: { coder: string | undefined; reviewer: string | undefined };
}

const CONFIG_PATH = "factory.yaml";
const ROLE_PATHS = { coder: ".factory/agents/coder.md", reviewer: ".factory/agents/reviewer.md" };

/** `factory.yaml` and the role prompts of commit `base` in the repository of `root`. Throws on a `factory.yaml` that does not parse. */
export async function loadTaskFiles(exec: Exec, root: string, base: string): Promise<TaskFiles> {
  const paths = [CONFIG_PATH, ROLE_PATHS.coder, ROLE_PATHS.reviewer];
  const present = new Set((await git(exec, root, ["ls-tree", "-r", "--name-only", base, "--", ...paths])).split("\n"));
  const show = (path: string) => (present.has(path) ? git(exec, root, ["show", `${base}:${path}`]) : undefined);
  const yaml = await show(CONFIG_PATH);
  let raw: unknown;
  try {
    raw = yaml === undefined ? undefined : Bun.YAML.parse(yaml);
  } catch (error) {
    throw new Error(`factory.yaml: ${error instanceof Error ? error.message : String(error)}`);
  }
  const prompt = async (path: string) => {
    const content = await show(path);
    return content === undefined ? undefined : rolePromptText(content) || undefined;
  };
  return { config: parseFactoryConfig(raw), prompts: { coder: await prompt(ROLE_PATHS.coder), reviewer: await prompt(ROLE_PATHS.reviewer) } };
}
