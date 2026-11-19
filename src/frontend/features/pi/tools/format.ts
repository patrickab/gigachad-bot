// Ported from kahme247/ompweb components/MessageView-tool-format.ts (MIT, Copyright (c) 2026 agegr).
import { isRecord } from "@/lib/utils"
import type { ToolCall } from "../types"

export type ToolCategory = "read" | "search" | "edit" | "terminal" | "todo" | "task" | "code" | "web" | "other"

export interface TodoSummary { op: string; action: string; label: string; task?: string }

const PREVIEW_LIMIT = 120

/** Icon and wording group of a tool, from Pi's built-in names and omp's conventions. */
export function getToolCategory(toolName: string): ToolCategory {
  const name = toolName.toLowerCase()
  if (name === "read" || name === "ls" || name.endsWith(".read") || name.endsWith("_read")) return "read"
  if (name === "grep" || name === "glob" || name === "find" || name.includes("search") || name.endsWith(".grep") || name.endsWith(".glob")) return "search"
  if (name === "edit" || name === "write" || name === "ast_edit" || name.endsWith(".edit") || name.endsWith(".write")) return "edit"
  if (name === "bash" || name === "terminal" || name === "exec" || name === "shell" || name === "cmd") return "terminal"
  if (name === "todo" || name.endsWith(".todo")) return "todo"
  if (name === "task" || name === "agent" || name.endsWith(".task") || name.endsWith(".agent")) return "task"
  if (name === "eval" || name.endsWith(".eval")) return "code"
  if (name === "web_search" || name === "fetch" || name.includes("browser")) return "web"
  return "other"
}

/** Human summary of a todo tool call (`op` plus task), null when the arguments carry no `op`. */
export function getTodoSummary(input: unknown): TodoSummary | null {
  if (!isRecord(input) || typeof input.op !== "string") return null
  const op = input.op
  const task = typeof input.task === "string" ? input.task : undefined
  const phase = typeof input.phase === "string" ? input.phase : undefined
  const items = Array.isArray(input.items) ? input.items.filter((item): item is string => typeof item === "string") : undefined
  const named = (action: string, fallback: string): TodoSummary => ({ op, action, label: task ? `${action} "${task}"` : fallback, task })

  switch (op) {
    case "done": return named("Completed", "Completed task")
    case "start": return named("Started", "Started task")
    case "block": return named("Blocked", "Blocked task")
    case "unblock": return named("Unblocked", "Unblocked task")
    case "drop": return named("Dropped", "Dropped task")
    case "append": {
      const label = task ? `Added "${task}"` : items?.length ? `Added ${items.length} task${items.length > 1 ? "s" : ""}` : "Added task"
      return { op, action: "Added", label, task: task ?? items?.[0] }
    }
    case "init": {
      const count = Array.isArray(input.list)
        ? input.list.reduce<number>((total, phaseItem) => total + (isRecord(phaseItem) && Array.isArray(phaseItem.items) ? phaseItem.items.length : 0), 0)
        : items?.length ?? 0
      return { op, action: "Initialized", label: `Initialized plan (${count} tasks)`, task: phase }
    }
    default: return { op, action: op, label: `Task: ${op}${task ? ` "${task}"` : ""}`, task }
  }
}

/** One-line rendering of an argument value. */
function formatPreviewValue(value: unknown): string {
  if (typeof value === "string") return value
  if (value === null || value === undefined) return ""
  if (typeof value === "number" || typeof value === "boolean" || typeof value === "bigint") return String(value)
  return safeJson(value)
}

/** JSON text of a value, empty when it cannot be serialized. */
export function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2) ?? ""
  } catch {
    return ""
  }
}

/** Short row preview: the command, path, pattern or query the call acts on. */
export function getToolPreview(call: ToolCall): string {
  const input = call.arguments
  if (!isRecord(input)) return ""
  if (getToolCategory(call.name) === "todo") {
    const todo = getTodoSummary(input)
    if (todo) return todo.label
  }
  const key = ["command", "path", "file_path", "pattern", "query"].find((name) => name in input) ?? Object.keys(input)[0]
  return key === undefined ? "" : formatPreviewValue(input[key]).slice(0, PREVIEW_LIMIT)
}

/** The `$ …` line of the expanded row. */
export function formatToolCommand(call: ToolCall): string {
  const input = call.arguments
  if (!isRecord(input)) return call.name
  if (typeof input.command === "string") return input.command
  for (const key of ["path", "file_path", "pattern", "query"]) {
    if (typeof input[key] === "string") return `${call.name} ${input[key]}`
  }
  const json = safeJson(input).replace(/\s*\n\s*/g, " ")
  return json ? `${call.name} ${json}` : call.name
}

/** Path a read/edit/write call targets, null for other tools or a missing path. */
export function getToolFilePath(call: ToolCall): string | null {
  const category = getToolCategory(call.name)
  if ((category !== "read" && category !== "edit") || call.name.toLowerCase() === "ls") return null
  const input = call.arguments
  if (!isRecord(input)) return null
  const path = typeof input.path === "string" ? input.path : typeof input.file_path === "string" ? input.file_path : null
  return path?.trim() || null
}

/** Whether a path carries a URL scheme (`https://`, `local://`, ...) rather than naming a file. */
export function hasUrlScheme(path: string): boolean {
  return /^[a-z][a-z0-9+.-]*:\/\//i.test(path)
}

/** The page a `read` of an http(s) URL fetched, without read selectors such as `:raw` or `:10-20`. */
export function readTargetWebUrl(target: string): string | null {
  if (!/^https?:\/\//i.test(target)) return null
  try {
    const url = new URL(target)
    url.pathname = url.pathname.replace(/(?::(?:raw|-?\d[\d,+-]*))+$/, "")
    return url.href
  } catch {
    return null
  }
}

/** Seconds label for a run duration (`0.4s`, `12s`). */
export function formatDuration(durationMs: number): string {
  const seconds = durationMs / 1000
  return seconds < 10 ? `${seconds.toFixed(1)}s` : `${Math.round(seconds)}s`
}
