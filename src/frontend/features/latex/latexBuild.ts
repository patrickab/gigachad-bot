import { fenceCode } from "@gigachad/chat-ui"
import type { NeovimSelection } from "@/features/editor/NeovimEditor"
import { fileViewerRawUrl } from "@/lib/backend/urls"
import { baseName } from "@/lib/paths"

/**
 * Run `build` one at a time. The returned trigger resolves once a build that started after the call has finished.
 *
 * - A trigger during a build asks for exactly one more build after it, however many arrive
 * - `build` must not throw
 */
export function createBuildQueue(build: () => Promise<void>): () => Promise<void> {
  let running: Promise<void> | null = null
  let followUp = false
  const drain = async () => {
    try {
      do {
        followUp = false
        await build()
      } while (followUp)
    } finally {
      running = null
    }
  }
  return () => {
    if (running) {
      followUp = true
      return running
    }
    running = drain()
    return running
  }
}

/** The composer draft for `selection` of the file at `path`: `name:start-end` over the quoted lines, then a blank line to type below. */
export function quoteSelection(path: string, selection: NeovimSelection): string {
  return `${baseName(path)}:${selection.start_line}-${selection.end_line}\n${fenceCode("latex", selection.text)}\n\n`
}

/** Raw URL of the built PDF, `revision` makes the browser and the viewer load the new bytes. */
export function pdfUrlOf(pdfPath: string, revision: number): string {
  return `${fileViewerRawUrl(pdfPath)}&v=${revision}`
}

/** Whether `path` names a `.tex` file, the files the LaTeX workspace opens. */
export function isLatexFile(path: string): boolean {
  return path.toLowerCase().endsWith(".tex")
}
