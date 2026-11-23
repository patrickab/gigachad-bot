// Ported from kahme247/ompweb components/MessageView.tsx (MIT, Copyright (c) 2026 agegr).
"use client"

import { memo, useId, useState, type JSX, type MouseEvent } from "react"
import { Bot, Check, CheckSquare, ChevronDown, CircleAlert, CircleSlash, Code2, FileEdit, FileText, Globe, LoaderCircle, Search, Terminal, Wrench, type LucideIcon } from "lucide-react"
import type { ImageContent, TextContent, ToolCall, ToolRun } from "../types"
import { getResultDiff, PairedDiffResult, PairedResult } from "./DiffView"
import {
  formatDuration,
  formatToolCommand,
  getTodoSummary,
  getToolCategory,
  getToolFilePath,
  getToolPreview,
  hasUrlScheme,
  readTargetWebUrl,
  safeJson,
  type ToolCategory,
} from "./format"
import "./tools.css"

interface ToolCallRowProps {
  call: ToolCall
  run?: ToolRun
  streaming?: boolean
  defaultCollapsed?: boolean
  onOpenFile?: (path: string) => void
}

const CATEGORY_ICONS: Record<ToolCategory, LucideIcon> = {
  read: FileText,
  search: Search,
  edit: FileEdit,
  terminal: Terminal,
  todo: CheckSquare,
  task: Bot,
  code: Code2,
  web: Globe,
  other: Wrench,
}

/** What the preview text does when activated: open a workspace file, open a web page, or nothing. */
type PreviewTarget = { kind: "file"; path: string } | { kind: "web"; url: string } | null

/** Clickable target of a read/edit/write preview; other URL schemes (`local://`, ...) are not openable. */
function getPreviewTarget(call: ToolCall, canOpenFile: boolean): PreviewTarget {
  const rawPath = getToolFilePath(call)
  if (!rawPath) return null
  if (hasUrlScheme(rawPath)) {
    const url = readTargetWebUrl(rawPath)
    return url ? { kind: "web", url } : null
  }
  // Strip only a trailing `:<line>` suffix, a drive-letter colon is never one.
  return canOpenFile ? { kind: "file", path: rawPath.replace(/:\d+$/, "") } : null
}

/** One collapsible activity row for a Pi tool call: status, tool, preview, duration; expanded shows input and output. */
export function ToolCallRow(props: ToolCallRowProps): JSX.Element {
  return <MemoToolCallRow {...props} />
}

// Streaming re-parses tool call arguments every frame, so compare the small argument objects shallowly.
const MemoToolCallRow = memo(ToolCallRowView, (prev, next) => (
  prev.call.id === next.call.id
  && prev.call.name === next.call.name
  && argumentsShallowEqual(prev.call.arguments, next.call.arguments)
  && prev.run === next.run
  && prev.streaming === next.streaming
  && prev.defaultCollapsed === next.defaultCollapsed
  && prev.onOpenFile === next.onOpenFile
))

/** Whether two argument objects hold the same values by identity. */
function argumentsShallowEqual(a: Record<string, unknown>, b: Record<string, unknown>): boolean {
  if (a === b) return true
  const keysA = Object.keys(a)
  return keysA.length === Object.keys(b).length && keysA.every((key) => a[key] === b[key])
}

function ToolCallRowView({ call, run, streaming = false, defaultCollapsed = true, onOpenFile }: ToolCallRowProps) {
  const isRunning = run?.running === true
  const isError = run?.isError === true
  const isDone = run !== undefined && !isRunning
  // A running row opens when tool calls are not kept collapsed, so its output is watchable live. It never auto-collapses.
  const [expanded, setExpanded] = useState((streaming || isRunning) && !defaultCollapsed)
  const [wasRunning, setWasRunning] = useState(isRunning)
  if (wasRunning !== isRunning) {
    setWasRunning(isRunning)
    if (isRunning && !defaultCollapsed) setExpanded(true)
  }
  const [inputExpanded, setInputExpanded] = useState(false)
  const detailsId = useId()
  const inputId = useId()

  const content = run?.result?.content ?? []
  const resultText = run?.result ? content.filter((block): block is TextContent => block.type === "text").map((block) => block.text).join("\n") : null
  const resultImages = content.filter((block): block is ImageContent => block.type === "image")
  const lastLine = isRunning ? (resultText ?? "").trimEnd().split("\n").at(-1)?.trim() : undefined
  const resultIsEmpty = resultText !== null && (resultText.trim() === "" || resultText.trim() === "(no output)")
  const resultDiff = expanded && isDone && !isError ? getResultDiff(run.result?.details) : null

  const category = getToolCategory(call.name)
  const CategoryIcon = CATEGORY_ICONS[category]
  const todoSummary = category === "todo" ? getTodoSummary(call.arguments) : null
  const preview = getToolPreview(call)
  const previewTarget = getPreviewTarget(call, onOpenFile !== undefined)
  const status = isError ? "failed" : isDone ? "done" : isRunning || streaming ? "running" : "not run"

  const openPreviewTarget = (event: MouseEvent) => {
    event.stopPropagation()
    if (previewTarget?.kind === "file") onOpenFile?.(previewTarget.path)
    else if (previewTarget?.kind === "web") window.open(previewTarget.url, "_blank", "noopener,noreferrer")
  }

  return (
    <div className="pi-activity-row" data-activity-operation="true">
      {/* Clicks anywhere on the header toggle; the trigger button's own click (mouse or keyboard) bubbles here. */}
      <div className="pi-activity-row-header" onClick={() => setExpanded((value) => !value)}>
        <button
          type="button"
          className="pi-activity-row-trigger"
          data-fill={previewTarget ? undefined : "true"}
          aria-expanded={expanded}
          aria-controls={detailsId}
        >
          <span className="pi-activity-row-indicator" data-error={isError || undefined} aria-hidden>
            {isError ? (
              <CircleAlert size={12} strokeWidth={1.8} />
            ) : isDone ? (
              <Check size={12} strokeWidth={2} />
            ) : isRunning || streaming ? (
              <LoaderCircle size={12} strokeWidth={1.8} className="pi-activity-row-spinner" />
            ) : (
              <CircleSlash size={12} strokeWidth={1.8} className="pi-activity-row-idle" />
            )}
          </span>
          <span className="sr-only">{status}</span>
          <span className="pi-activity-tool-icon" data-category={category} aria-hidden>
            <CategoryIcon size={12} strokeWidth={1.8} />
          </span>
          <span className="pi-activity-row-tool" data-error={isError || undefined}>{call.name}</span>
          {!previewTarget && <span className="pi-activity-row-preview">{preview}</span>}
        </button>
        {previewTarget && (
          <button type="button" className="pi-activity-row-preview pi-activity-file-link" onClick={openPreviewTarget}>
            {preview}
          </button>
        )}
        {isDone && run.durationMs !== undefined && (
          <span className="pi-activity-row-duration">{formatDuration(run.durationMs)}</span>
        )}
        <ChevronDown size={12} strokeWidth={1.8} aria-hidden className="pi-activity-row-chevron" data-expanded={expanded || undefined} />
      </div>
      {!expanded && isRunning && lastLine && <div className="pi-tool-call-live-tail" data-tool-live-tail="true">{lastLine}</div>}
      {expanded && (
        <div id={detailsId} className="pi-tool-call-details" data-error={isError || undefined}>
          <div className="pi-tool-call-command">
            <span className="pi-tool-call-command-prompt" aria-hidden>$</span>
            <code>{formatToolCommand(call)}</code>
            <button
              type="button"
              className="pi-tool-call-input-toggle"
              aria-expanded={inputExpanded}
              aria-controls={inputId}
              onClick={() => setInputExpanded((value) => !value)}
            >
              {inputExpanded ? "Collapse input" : "Show full input"}
            </button>
          </div>
          <div id={inputId} hidden={!inputExpanded} className="pi-tool-call-input">
            {inputExpanded && (Object.keys(call.arguments).length > 0 ? (
              <dl>
                {Object.entries(call.arguments).map(([key, value]) => (
                  <div key={key}>
                    <dt>{key}</dt>
                    <dd><pre>{typeof value === "string" ? value : safeJson(value)}</pre></dd>
                  </div>
                ))}
              </dl>
            ) : <pre>{safeJson(call.arguments)}</pre>)}
          </div>
          {todoSummary && (
            <div className="pi-tool-call-todo-badge">
              <span className="pi-todo-op-tag" data-op={todoSummary.op}>{todoSummary.action}</span>
              <span className="pi-todo-task-name">{todoSummary.task ?? todoSummary.label}</span>
            </div>
          )}
          {isRunning && (resultText ?? "").trim() === "" ? (
            // No output yet: say so instead of "(no output)", which would claim the tool finished with nothing.
            <div className="pi-tool-call-running" data-tool-running="true">Running…</div>
          ) : run?.result ? (
            resultDiff ? (
              <PairedDiffResult text={resultDiff} />
            ) : (
              <>
                {resultImages.length > 0 && (
                  <div className="pi-tool-call-images">
                    {resultImages.map((image, index) => (
                      // eslint-disable-next-line @next/next/no-img-element -- inline base64 tool output, nothing for next/image to optimize
                      <img key={index} src={`data:${image.mimeType};base64,${image.data}`} alt="" className="pi-tool-call-image" />
                    ))}
                  </div>
                )}
                {!(resultIsEmpty && resultImages.length > 0) && (
                  <PairedResult text={resultText ?? ""} isEmpty={resultIsEmpty} isError={isError} />
                )}
              </>
            )
          ) : null}
        </div>
      )}
    </div>
  )
}
