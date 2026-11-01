import { fenceCode } from "@gigachad/chat-ui"
import type { NeovimSelection } from "@/features/editor/NeovimEditor"
import { baseName } from "@/lib/paths"

/** The composer draft for `selection` of the file at `path`: `name:start-end` over the lines fenced as `language`, then a blank line to type below. */
export function quoteSelection(path: string, selection: NeovimSelection, language: string): string {
  return `${baseName(path)}:${selection.start_line}-${selection.end_line}\n${fenceCode(language, selection.text)}\n\n`
}
