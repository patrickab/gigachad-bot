import { SLASH_SOURCE_ORDER, slashMatchRank } from "@/components/ChatInput-slash-commands";
import type { SlashCommandInfo } from "@/hooks/useAgentSession";
import { buildAtInsertText, buildEntriesFromFiles, extractAtQuery, filterFileEntries, type FileIndexEntry } from "@/lib/file-fuzzy";
import type { Completion } from "@gigachad/chat-ui/protocol";

const COMMAND_LIMIT = 50;
/** `slashMatchRank` for a command that matches neither name nor description. */
const NO_MATCH_RANK = 4;

/** Slash query the way upstream's menu opens it: the whole draft is `/word`, no whitespace yet. */
export function slashQueryOf(text: string): string | null {
  return text.startsWith("/") && !/\s/.test(text.slice(1)) ? text.slice(1).toLowerCase() : null;
}

/** omp's extension, prompt and skill commands matching `query`, ranked like upstream's menu. omp builtins are not in `commands`. */
export function commandCompletions(commands: SlashCommandInfo[], text: string, query: string): Completion[] {
  return commands
    .filter((command) => slashMatchRank(command, query) < NO_MATCH_RANK)
    .sort((a, b) => slashMatchRank(a, query) - slashMatchRank(b, query)
      || SLASH_SOURCE_ORDER[a.source] - SLASH_SOURCE_ORDER[b.source]
      || a.name.localeCompare(b.name))
    .slice(0, COMMAND_LIMIT)
    .map((command) => {
      const insert = `/${command.name} `;
      return { kind: command.source, label: `/${command.name}`, detail: command.description, from: 0, to: text.length, insert, caretOffset: insert.length };
    });
}

/** Files for an `@` query, ranked by omp-web's file index like upstream's menu: server-side with a query, the shallow listing without. */
async function fileEntries(cwd: string, query: string): Promise<FileIndexEntry[]> {
  const params = new URLSearchParams({ cwd });
  if (query) params.set("q", query);
  const response = await fetch(`/api/file-index?${params}`);
  if (!response.ok) throw new Error(`file index failed: ${response.status}`);
  const body = (await response.json()) as { files?: string[]; matches?: FileIndexEntry[] };
  return query ? body.matches ?? [] : filterFileEntries(buildEntriesFromFiles(body.files ?? []), "");
}

/**
 * `@` completions for the token before `caret`, null without one. Inserts like upstream's `applyAtCompletion`.
 *
 * - Files close the token (`@path `), directories keep it open for drill-down (`@dir/`)
 * - Inside a quoted token the closing quote after the caret is replaced, the insert carries its own
 */
export async function fileCompletions(cwd: string, text: string, caret: number): Promise<Completion[] | null> {
  const token = extractAtQuery(text.slice(0, caret));
  if (!token) return null;
  const to = token.quoted && text.startsWith('"', caret) ? caret + 1 : caret;
  return (await fileEntries(cwd, token.query)).map((entry) => {
    const insertion = buildAtInsertText(entry.path, entry.isDir, token.quoted);
    return {
      kind: entry.isDir ? "directory" : "file",
      label: entry.isDir ? `${entry.path}/` : entry.path,
      from: token.start,
      to,
      insert: insertion.text,
      caretOffset: insertion.cursorOffset,
    };
  });
}
