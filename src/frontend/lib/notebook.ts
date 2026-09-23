/**
 * Percent-format notebook parsing for chat-scoped Python notebooks.
 *
 * A notebook is plain text: `# %%` opens a code cell, `# %% [markdown]` opens
 * a Markdown cell whose body lines are `# `-prefixed.
 */

export interface NotebookCell {
  kind: "code" | "markdown"
  source: string
}

const CODE_MARKER = "# %%"
const MARKDOWN_MARKER = "# %% [markdown]"

// Recognise a cell-opening marker; anything else belongs to the current cell.
function markerKind(line: string): NotebookCell["kind"] | null {
  const trimmed = line.trim()
  if (trimmed === CODE_MARKER) return "code"
  if (trimmed === MARKDOWN_MARKER) return "markdown"
  return null
}

// Drop trailing blank lines so cells never carry stray padding.
function trimTrailingBlanks(lines: string[]): string[] {
  const copy = [...lines]
  while (copy.length > 0 && copy[copy.length - 1].trim() === "") copy.pop()
  return copy
}

export function parseNotebook(text: string): NotebookCell[] {
  const lines = text.split(/\r?\n/)
  const starts: { line: number; kind: NotebookCell["kind"] }[] = []
  lines.forEach((line, index) => {
    const kind = markerKind(line)
    if (kind) starts.push({ line: index, kind })
  })

  // No markers: the whole file is a single implicit code cell.
  if (starts.length === 0) {
    return [{ kind: "code", source: trimTrailingBlanks(lines).join("\n") }]
  }

  const cells: NotebookCell[] = []

  // Non-blank content before the first marker is an implicit leading code cell.
  const head = trimTrailingBlanks(lines.slice(0, starts[0].line))
  if (head.length > 0) cells.push({ kind: "code", source: head.join("\n") })

  starts.forEach((start, index) => {
    const end = index + 1 < starts.length ? starts[index + 1].line : lines.length
    let body = lines.slice(start.line + 1, end)
    // Markdown bodies are `# `-prefixed; a bare `#` encodes an empty line.
    if (start.kind === "markdown")
      // Normalize on parse: a bare `#` encodes an empty line, `# ` strips the
      // prefix, and a non-conforming line gains the prefix.
      body = body.map((line) => (line === "" || line === "#" ? "" : line.startsWith("# ") ? line.slice(2) : `# ${line}`))
    const source = trimTrailingBlanks(body).join("\n")
    cells.push({ kind: start.kind, source })
  })

  return cells
}

export function serializeNotebook(cells: NotebookCell[]): string {
  if (cells.length === 0) return ""
  const blocks = cells.map((cell) => {
    const marker = cell.kind === "markdown" ? MARKDOWN_MARKER : CODE_MARKER
    // Markdown body lines regain their `# ` prefix; empty lines become bare `#`.
    const body =
      cell.source === ""
        ? []
        : cell.source.split("\n").map((line) => (cell.kind === "markdown" ? (line === "" ? "#" : `# ${line}`) : line))
    return [marker, ...body].join("\n")
  })
  // Blank line between cells, single trailing newline at EOF.
  return `${blocks.join("\n\n")}\n`
}

function unmatchedCells(before: NotebookCell[], after: NotebookCell[]): [NotebookCell[], NotebookCell[]] {
  const pendingBefore = [...before]
  const pendingAfter = after.filter((cell) => {
    const index = pendingBefore.findIndex((candidate) => candidate.kind === cell.kind && candidate.source === cell.source)
    if (index === -1) return true
    pendingBefore.splice(index, 1)
    return false
  })
  return [pendingBefore, pendingAfter]
}

// Pair leftover before/after cells by kind in order; each pair is one edit, not a remove+add.
function pairEdits(pendingBefore: NotebookCell[], pendingAfter: NotebookCell[]): number {
  const taken = new Set<number>()
  let modified = 0
  pendingAfter.forEach((cell) => {
    const match = pendingBefore.findIndex((candidate, index) => !taken.has(index) && candidate.kind === cell.kind)
    if (match !== -1) {
      taken.add(match)
      modified += 1
    }
  })
  return modified
}

export function notebookDiff(
  beforeText: string,
  afterText: string,
): { added: number; modified: number; removed: number } {
  const [pendingBefore, pendingAfter] = unmatchedCells(parseNotebook(beforeText), parseNotebook(afterText))
  const modified = pairEdits(pendingBefore, pendingAfter)
  return {
    added: pendingAfter.length - modified,
    modified,
    removed: pendingBefore.length - modified,
  }
}