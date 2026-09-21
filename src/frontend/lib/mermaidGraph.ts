// Converts between the canonical Architecture Graph model and a deliberately
// narrow Mermaid flowchart subset: one `flowchart <direction>` header, node
// declarations using `[]`/`()`/`(())`/`{}` shapes, and `-->`/`<-->` edges with
// optional `|label|` and an optional Mermaid v11 `id@` explicit edge id.
// Anything outside that subset is rejected rather than guessed at.
//
// ponytail: this is a small hand-written parser, not the real Mermaid
// grammar (no subgraphs, styling, links, or the full node-shape set).
// Upgrade path if broader compatibility is needed: parse through Mermaid's
// own `mermaidAPI`/`FlowDB` instead of this regex-based reader.
import {
  ARCHITECTURE_GRAPH_VERSION,
  nextArchitectureGraphId,
  validateArchitectureGraph,
  type ArchitectureGraph,
  type ArchitectureGraphEdge,
  type ArchitectureGraphNode,
  type ArchitectureGraphNodeShape,
} from "./architectureGraph"

function escapeMermaidLabel(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/[[\]{}|]/g, (char) => `#${char.charCodeAt(0)};`)
}

function unescapeMermaidLabel(text: string): string {
  return text.replace(/#(9[13]|12[345]);/g, (_, code: string) => String.fromCharCode(Number(code))).replace(/&quot;/g, "\"").replace(/&amp;/g, "&")
}

// Splits a Mermaid label into a title and bullet list using the same "line
// per entry" convention the node card editor already writes with `<br/>`.
function labelToTitleAndBullets(label: string): { title: string, bullets: string[] } {
  const [title = "Untitled", ...rest] = unescapeMermaidLabel(label).split(/<br\s*\/?>/i).map((line) => line.trim()).filter(Boolean)
  return { title, bullets: rest.map((line) => line.replace(/^[•-]\s*/, "")) }
}

function titleAndBulletsToLabel(node: ArchitectureGraphNode): string {
  return escapeMermaidLabel([node.title, ...node.bullets.map((bullet) => `• ${bullet}`)].join("<br/>"))
}

function shapeBrackets(shape: ArchitectureGraphNodeShape | undefined, label: string): string {
  if (shape === "ellipse") return `((${label}))`
  if (shape === "diamond") return `{${label}}`
  return `[${label}]`
}

interface ParsedToken {
  id: string
  shape?: ArchitectureGraphNodeShape
  label?: string
}

const NODE_TOKEN = /^([A-Za-z0-9_-]+)(?:\(\((.*)\)\)|\[(.*)\]|\{(.*)\}|\((.*)\))?$/

function parseNodeToken(token: string): ParsedToken | null {
  const match = token.trim().match(NODE_TOKEN)
  if (!match) return null
  const [, id, circle, rect, diamond, rounded] = match
  if (circle !== undefined) return { id, shape: "ellipse", label: circle }
  if (diamond !== undefined) return { id, shape: "diamond", label: diamond }
  if (rect !== undefined) return { id, shape: "rectangle", label: rect }
  if (rounded !== undefined) return { id, shape: "rectangle", label: rounded }
  return { id }
}

interface ParsedEdgeLine {
  left: string
  right: string
  edgeId?: string
  bidirectional: boolean
  label?: string
}

const ARROW = /(?:([A-Za-z0-9_-]+)@)?(-->|<-->)/

function parseEdgeLine(line: string): ParsedEdgeLine | null {
  const arrow = line.match(ARROW)
  if (!arrow || arrow.index === undefined) return null
  const left = line.slice(0, arrow.index).trim()
  let rest = line.slice(arrow.index + arrow[0].length).trim()
  let label: string | undefined
  const labelled = rest.match(/^\|(.*?)\|\s*(.*)$/)
  if (labelled) { label = labelled[1]; rest = labelled[2] }
  const right = rest.trim()
  if (!left || !right) return null
  return { left, right, edgeId: arrow[1], bidirectional: arrow[2] === "<-->", label }
}

const HEADER = /^(?:flowchart|graph)\s+(TB|TD|BT|LR|RL)$/i

/** Parses the supported Mermaid flowchart subset into a validated Architecture Graph. */
export function parseMermaidFlowchart(source: string, title = "Imported flowchart"): ArchitectureGraph {
  const lines = source.split("\n").map((line) => line.replace(/^\s*%%.*/, "").trim()).filter(Boolean)
  const header = lines[0]?.match(HEADER)
  if (!header) throw new Error("Only a 'flowchart <direction>' or 'graph <direction>' header is supported")

  const nodes = new Map<string, ArchitectureGraphNode>()
  const edges: ArchitectureGraphEdge[] = []
  const edgeIds = new Set<string>()

  const ensureNode = (token: ParsedToken) => {
    const existing = nodes.get(token.id)
    const { title: nodeTitle, bullets } = token.label !== undefined ? labelToTitleAndBullets(token.label) : { title: existing?.title ?? token.id, bullets: existing?.bullets ?? [] }
    const shape = (token.shape === "rectangle" ? undefined : token.shape) ?? existing?.shape
    nodes.set(token.id, { id: token.id, title: nodeTitle, bullets, position: { x: 0, y: 0 }, ...(shape ? { shape } : {}) })
  }

  for (const line of lines.slice(1)) {
    // Try a node declaration first: a label may itself contain "-->" or "|",
    // and NODE_TOKEN never matches a real edge line (it requires the whole
    // line to be one id plus at most one shape/bracket pair).
    const declared = parseNodeToken(line)
    if (declared) { ensureNode(declared); continue }
    const edgeLine = parseEdgeLine(line)
    if (edgeLine) {
      const source = parseNodeToken(edgeLine.left)
      const target = parseNodeToken(edgeLine.right)
      if (!source || !target) throw new Error(`Unsupported Mermaid syntax: ${line}`)
      ensureNode(source)
      ensureNode(target)
      const id = edgeLine.edgeId ?? nextArchitectureGraphId("edge", edgeIds)
      if (edgeIds.has(id)) throw new Error(`Duplicate edge id: ${id}`)
      edgeIds.add(id)
      edges.push({
        id, source: source.id, target: target.id,
        direction: edgeLine.bidirectional ? "bidirectional" : "one-way",
        ...(edgeLine.label?.trim() ? { label: unescapeMermaidLabel(edgeLine.label.trim()) } : {}),
      })
      continue
    }
    throw new Error(`Unsupported Mermaid syntax: ${line}`)
  }

  // Mermaid carries no position data, so nodes land on a deterministic grid.
  const columns = Math.max(1, Math.ceil(Math.sqrt(nodes.size)))
  const positioned = Array.from(nodes.values()).map((node, index) => ({
    ...node,
    position: { x: (index % columns) * 280, y: Math.floor(index / columns) * 200 },
  }))

  return validateArchitectureGraph({ version: ARCHITECTURE_GRAPH_VERSION, title, nodes: positioned, edges })
}

/** Serializes an Architecture Graph as a Mermaid flowchart, preserving ids, shapes, and labels. */
export function graphToMermaidFlowchart(graph: ArchitectureGraph): string {
  const lines = ["flowchart TB"]
  for (const node of graph.nodes) lines.push(`    ${node.id}${shapeBrackets(node.shape, titleAndBulletsToLabel(node))}`)
  for (const edge of graph.edges) {
    const arrow = edge.direction === "bidirectional" ? "<-->" : "-->"
    const label = edge.label ? `|${escapeMermaidLabel(edge.label)}|` : ""
    lines.push(`    ${edge.source} ${edge.id}@${arrow}${label} ${edge.target}`)
  }
  return lines.join("\n")
}
