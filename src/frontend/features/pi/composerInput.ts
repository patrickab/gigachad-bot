import { MessageSquareText, Puzzle, Shrink, Sparkles, type LucideIcon } from "lucide-react"
import type { CompleteDraft, ComposerCompletion } from "@/features/chat/composer/CompletionMenu"
import { fuzzyPathScore, fuzzyScore, rankByScore } from "@/lib/fuzzy"
import type { PiCommand } from "./types"

/**
 * What a Pi composer send means, and the composer's `/` and `@` suggestions.
 *
 * - `!cmd` runs in the session's shell, `!!cmd` keeps its output out of the model's context (F7). Composer chips never join a command
 * - `/compact [instructions]` compacts the context, steered by the instructions (F8)
 * - Anything else is a prompt, chips included
 */
export type ComposerAction =
  | { kind: "prompt"; text: string }
  | { kind: "bash"; command: string; excludeFromContext: boolean }
  | { kind: "compact"; instructions?: string }

/** Commands the composer runs itself, listed with Pi's own in `/` suggestions. */
const BUILTIN_COMMANDS: { name: string; description: string; icon: LucideIcon }[] = [
  { name: "compact", description: "Compact the context, optionally steered by instructions", icon: Shrink },
]
const COMMAND_ICONS: Record<PiCommand["source"], LucideIcon> = { extension: Puzzle, prompt: MessageSquareText, skill: Sparkles }
const MAX_FILE_COMPLETIONS = 50

/** The typed draft of `sent` (what `ChatInput` sends: the draft, then each chip's text, blank-line separated). */
export function draftOf(sent: string, chipTexts: readonly string[]): string {
  const suffix = chipTexts.filter((text) => text.length > 0).join("\n\n")
  if (!suffix) return sent
  if (sent === suffix) return ""
  return sent.endsWith(`\n\n${suffix}`) ? sent.slice(0, sent.length - suffix.length - 2) : sent
}

/** The action of a send of `sent` with `chipTexts` staged. */
export function composerAction(sent: string, chipTexts: readonly string[]): ComposerAction {
  const bang = /^!(!?)([\s\S]*)$/.exec(draftOf(sent, chipTexts))
  const command = bang?.[2].trim()
  if (bang && command) return { kind: "bash", command, excludeFromContext: bang[1] === "!" }
  const compact = /^\/compact(?:\s+([\s\S]*))?$/.exec(sent)
  if (compact) {
    const instructions = compact[1]?.trim()
    return instructions ? { kind: "compact", instructions } : { kind: "compact" }
  }
  // Pi splits `/skill:<name> <args>` at the first space only: a newline or tab after the name would lose the args
  return { kind: "prompt", text: sent.replace(/^(\/skill:\S+)\s+/, "$1 ") }
}

function commandCompletion(name: string, detail: string | undefined, icon: LucideIcon, text: string): ComposerCompletion {
  const insert = `/${name} `
  return { label: `/${name}`, detail, icon, from: 0, to: text.length, insert, caretOffset: insert.length }
}

/** `/` suggestions while the draft is one word starting with `/`, ranked by fuzzy score. */
function slashCompletions(text: string, commands: readonly PiCommand[]): ComposerCompletion[] {
  const query = text.slice(1).toLowerCase()
  const entries = [
    ...BUILTIN_COMMANDS.map((command) => ({ name: command.name, completion: commandCompletion(command.name, command.description, command.icon, text) })),
    ...commands.map((command) => ({ name: command.name, completion: commandCompletion(command.name, command.description, COMMAND_ICONS[command.source], text) })),
  ]
  return entries
    .map((entry) => ({ entry, score: query ? fuzzyScore(entry.name, query) : 0 }))
    .filter((ranked): ranked is { entry: (typeof entries)[number]; score: number } => ranked.score !== null)
    .sort((left, right) => right.score - left.score || left.entry.name.localeCompare(right.entry.name))
    .map(({ entry }) => entry.completion)
}

/** `@path` suggestions for the `@` word ending at `caret`, `files` (repo-relative paths) ranked by fuzzy path score (M7, K6). */
export function fileCompletions(text: string, caret: number, files: readonly string[]): ComposerCompletion[] {
  const match = /(?:^|\s)@([^\s@]*)$/.exec(text.slice(0, caret))
  if (!match) return []
  const query = match[1]
  const from = caret - query.length - 1
  const ranked = query ? rankByScore(files, (path) => fuzzyPathScore(path, query)) : files
  return ranked.slice(0, MAX_FILE_COMPLETIONS).map((path) => {
    const insert = `@${path} `
    return { label: path, mono: true, from, to: caret, insert, caretOffset: insert.length }
  })
}

/** The composer's suggestions: Pi's commands for a lone `/word`, else the repo's files after `@`. */
export function piCompletions(commands: readonly PiCommand[], files: readonly string[]): CompleteDraft {
  return async (text, caret) => {
    if (text.startsWith("/") && !/\s/.test(text)) return slashCompletions(text, commands)
    return fileCompletions(text, caret, files)
  }
}
