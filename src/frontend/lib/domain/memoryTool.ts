// The text of a memory tool call's row (`list_memories`, `read_memories`, `apply_memory_ops`), shared by the chat's
// tool element and the Pi tab's row: one summary line and one line per op.

import { isRecord } from "@/lib/utils"

/** The memory tools' names; the backend expands the chat's `memory` toggle into these. */
export const MEMORY_TOOL_NAMES = ["list_memories", "read_memories", "apply_memory_ops"] as const

/** How Pi names the memory MCP server's tools: `mcp__<server>__<tool>`. */
export const MEMORY_MCP_PREFIX = "mcp__memory__"

export type MemoryToolKind = "list" | "read" | "ops"

/** One op of an `apply_memory_ops` call: `marker` (`+`, `~`, `−`, `write_doc`, `describe`), `ref` the id or path it names. */
export interface MemoryOpLine {
  marker: string
  ref: string
  text: string
}

export interface MemoryToolView {
  kind: MemoryToolKind
  summary: string
  ops: MemoryOpLine[]
}

const KINDS: Record<string, MemoryToolKind> = { list_memories: "list", read_memories: "read", apply_memory_ops: "ops" }

/** The kind of a memory tool, by its bare or `mcp__memory__` name; null for any other tool. */
export function memoryToolKind(name: string): MemoryToolKind | null {
  const bare = name.startsWith(MEMORY_MCP_PREFIX) ? name.slice(MEMORY_MCP_PREFIX.length) : name
  return KINDS[bare] ?? null
}

const optionalString = (value: unknown): value is string | undefined => value === undefined || typeof value === "string"
const str = (value: unknown): string => (typeof value === "string" ? value : "")

function opLine(op: Record<string, unknown>): MemoryOpLine {
  const path = str(op.path)
  switch (op.op) {
    case "add": return { marker: "+", ref: "", text: str(op.text) }
    case "update": return { marker: "~", ref: str(op.id), text: str(op.text) }
    case "retire": return { marker: "−", ref: str(op.id), text: "retired" }
    case "write_doc": return { marker: "write_doc", ref: path, text: "" }
    case "describe": return { marker: "describe", ref: path, text: str(op.description) }
    default: return { marker: str(op.op), ref: path, text: "" }
  }
}

/** The summary and op lines of a memory tool call; null for another tool or arguments that cannot be read. */
export function memoryToolView(name: string, args: unknown): MemoryToolView | null {
  const kind = memoryToolKind(name)
  if (!kind || !isRecord(args) || !optionalString(args.memory_store)) return null
  const store = args.memory_store ? [args.memory_store] : []
  if (kind === "list") {
    if (!optionalString(args.path)) return null
    return { kind, summary: ["list", ...store, ...(args.path ? [args.path] : [])].join(" · "), ops: [] }
  }
  if (kind === "read") {
    const paths = args.paths
    if (!Array.isArray(paths) || paths.length === 0 || !paths.every((path) => typeof path === "string")) return null
    return { kind, summary: ["read", ...store, paths.join(", ")].join(" · "), ops: [] }
  }
  const ops = args.ops
  if (!Array.isArray(ops) || ops.length === 0 || !ops.every((op) => isRecord(op) && typeof op.op === "string")) return null
  const records = ops as Record<string, unknown>[]
  const paths = [...new Set(records.map((op) => str(op.path)).filter(Boolean))]
  const count = `${records.length} change${records.length === 1 ? "" : "s"}`
  return { kind, summary: [count, ...(paths.length ? [paths.join(", ")] : [])].join(" · "), ops: records.map(opLine) }
}
