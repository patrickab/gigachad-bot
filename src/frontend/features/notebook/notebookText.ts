/**
 * The notebook as one plain text in the percent format (`# %%` markers), what Neovim edits, and the way back to the model.
 *
 * - Every cell starts with a marker line, cells are joined by one blank line, the text ends with `\n`. An empty notebook is `""`
 * - A code cell's body is its source, a markdown or raw cell's body is its source with every line commented (`# `, an empty line `#`,
 *   a line starting with `%%` `#%%…` so it never reads as a marker)
 * - The text has no outputs: `applyNotebookText` matches the parsed cells to the model's, so a cell that stays keeps its outputs
 */
import { applyNotebookOps, diffNotebook, notebookOf, type NotebookCellContent, type NotebookContent, type SharedNotebookLike } from "./notebookSync"

/** One cell of the text. */
export interface TextCell {
  cell_type: "code" | "markdown" | "raw"
  source: string
}

/** A marker line, also the Lua side's (`neovim_rpc.py`). */
const MARKER = /^# %%(\s.*)?$/

const sourceOf = (cell: NotebookCellContent) => (Array.isArray(cell.source) ? cell.source.join("") : cell.source)

function markerOf(cellType: string): string {
  if (cellType === "markdown") return "# %% [markdown]"
  if (cellType === "raw") return "# %% [raw]"
  return "# %%"
}

function bodyOf(cell: NotebookCellContent): string {
  const source = sourceOf(cell)
  if (cell.cell_type === "code" || !source) return source
  return source.split("\n").map((line) => (!line ? "#" : MARKER.test(`# ${line}`) ? `#${line}` : `# ${line}`)).join("\n")
}

/** The percent text of `content`, its cells without outputs. */
export function notebookToText(content: NotebookContent): string {
  if (content.cells.length === 0) return ""
  const blocks = content.cells.map((cell) => {
    const body = bodyOf(cell)
    return body ? `${markerOf(cell.cell_type)}\n${body}` : markerOf(cell.cell_type)
  })
  return `${blocks.join("\n\n")}\n`
}

function kindOf(marker: string): TextCell["cell_type"] {
  const rest = marker.slice(4).trim()
  if (rest.startsWith("[markdown]") || rest.startsWith("[md]")) return "markdown"
  if (rest.startsWith("[raw]")) return "raw"
  return "code"
}

const uncomment = (line: string) => (line.startsWith("# ") ? line.slice(2) : line.startsWith("#") ? line.slice(1) : line)

/** The cells of `text` with the 1-based line each starts on. */
function scan(text: string): (TextCell & { line: number })[] {
  const lines = text.split("\n")
  if (text.endsWith("\n")) lines.pop()
  const cells: (TextCell & { line: number; body: string[] })[] = []
  const preamble: string[] = []
  lines.forEach((line, index) => {
    // ponytail: a code line that looks like a marker starts a new cell, as in jupytext
    if (MARKER.test(line)) cells.push({ cell_type: kindOf(line), source: "", line: index + 1, body: [] })
    else (cells.at(-1)?.body ?? preamble).push(line)
  })
  if (preamble.some((line) => line.trim())) cells.unshift({ cell_type: "code", source: "", line: 1, body: preamble })
  return cells.map(({ cell_type, line, body }) => {
    let end = body.length
    while (end > 0 && !body[end - 1].trim()) end--
    const kept = body.slice(0, end)
    return { cell_type, line, source: (cell_type === "code" ? kept : kept.map(uncomment)).join("\n") }
  })
}

/** The cells of a percent text. Text before the first marker is a code cell when it is not blank. */
export function parseNotebookText(text: string): TextCell[] {
  return scan(text).map(({ cell_type, source }) => ({ cell_type, source }))
}

/** Index of the cell of `text` that holds the 1-based `line` (its marker or body), 0 before the first marker, null when the text has no cells. */
export function cellAtLine(text: string, line: number): number | null {
  const cells = scan(text)
  if (cells.length === 0) return null
  let index = 0
  cells.forEach((cell, at) => { if (cell.line <= line) index = at })
  return index
}

// ---------------------------------------------------------------------------------------------------------------------
// Matching the text's cells to the model's
// ---------------------------------------------------------------------------------------------------------------------

type Pair = [number, number]

/** Longest common subsequence of `a[aFrom, aTo)` and `b[bFrom, bTo)` under `equal`, as index pairs in order. */
function lcs<A, B>(a: A[], b: B[], [aFrom, aTo, bFrom, bTo]: number[], equal: (x: A, y: B) => boolean): Pair[] {
  const [n, m] = [aTo - aFrom, bTo - bFrom]
  const lengths = Array.from({ length: n + 1 }, () => new Array<number>(m + 1).fill(0))
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      lengths[i][j] = equal(a[aFrom + i], b[bFrom + j]) ? lengths[i + 1][j + 1] + 1 : Math.max(lengths[i + 1][j], lengths[i][j + 1])
    }
  }
  const pairs: Pair[] = []
  for (let i = 0, j = 0; i < n && j < m;) {
    if (equal(a[aFrom + i], b[bFrom + j])) pairs.push([aFrom + i++, bFrom + j++])
    else if (lengths[i + 1][j] >= lengths[i][j + 1]) i++
    else j++
  }
  return pairs
}

/** The ranges `[aFrom, aTo, bFrom, bTo]` between the `pairs` inside `range`, empty ones left out. */
function gapsOf(pairs: Pair[], [aFrom, aTo, bFrom, bTo]: number[]): number[][] {
  const gaps: number[][] = []
  let [a, b] = [aFrom, bFrom]
  for (const [i, j] of [...pairs, [aTo, bTo] as Pair]) {
    if (i > a || j > b) gaps.push([a, i, b, j])
    ;[a, b] = [i + 1, j + 1]
  }
  return gaps
}

/** `2 · common lines / all lines` of two sources. */
function similarity(a: string, b: string): number {
  const [left, right] = [a.split("\n"), b.split("\n")]
  const counts = new Map<string, number>()
  for (const line of left) counts.set(line, (counts.get(line) ?? 0) + 1)
  let common = 0
  for (const line of right) {
    const count = counts.get(line) ?? 0
    if (count > 0) {
      common++
      counts.set(line, count - 1)
    }
  }
  return (2 * common) / (left.length + right.length)
}

/** Pairs `[old index, new index]` of the cells that stay: exact matches, then similar ones of the same type, then same-type cells at the same place in a gap of equal size. */
function matchCells(old: TextCell[], next: TextCell[]): Pair[] {
  const all = [0, old.length, 0, next.length]
  const exact = lcs(old, next, all, (a, b) => a.cell_type === b.cell_type && a.source === b.source)
  const pairs = [...exact]
  for (const gap of gapsOf(exact, all)) {
    const similar = lcs(old, next, gap, (a, b) => a.cell_type === b.cell_type && similarity(a.source, b.source) >= 0.5)
    pairs.push(...similar)
    for (const [aFrom, aTo, bFrom, bTo] of gapsOf(similar, gap)) {
      if (aTo - aFrom !== bTo - bFrom) continue
      for (let k = 0; k < aTo - aFrom; k++) if (old[aFrom + k].cell_type === next[bFrom + k].cell_type) pairs.push([aFrom + k, bFrom + k])
    }
  }
  return pairs
}

function newCell({ cell_type, source }: TextCell): NotebookCellContent {
  const cell: NotebookCellContent = { id: crypto.randomUUID(), cell_type, source, metadata: {} }
  if (cell_type === "code") Object.assign(cell, { outputs: [], execution_count: null })
  return cell
}

/**
 * Make `model` hold the cells of the percent `text`: matched cells keep their id, metadata, outputs and execution count and take the new source,
 * the others are deleted or inserted, in the text's order. A matched cell whose source the text did not change (it equals the cell's own
 * round trip through the text, which drops trailing blank lines) keeps its source byte for byte.
 */
export function applyNotebookText(model: SharedNotebookLike, text: string): void {
  const notebook = notebookOf(model)
  // Keys of our own for the model's cells: `diffNotebook` matches by id, and below nbformat 4.5 `toJSON` has none. A matched cell is never inserted, so a key never reaches the model
  const current: NotebookContent = { ...notebook, cells: notebook.cells.map((cell, index) => ({ ...cell, id: `cell-${index}` })) }
  const next = parseNotebookText(text)
  // A cell as the text shows it, so a cell the user did not touch matches exactly
  const old = current.cells.map((cell) => ({ cell_type: cell.cell_type as TextCell["cell_type"], source: parseNotebookText(notebookToText({ ...current, cells: [cell] }))[0]?.source ?? "" }))
  const matched = new Map(matchCells(old, next).map(([i, j]) => [j, i]))
  const cells = next.map((cell, index) => {
    const at = matched.get(index)
    if (at === undefined) return newCell(cell)
    return cell.source === old[at].source ? current.cells[at] : { ...current.cells[at], source: cell.source }
  })
  applyNotebookOps(model, diffNotebook(current, current, { ...current, cells }))
}
