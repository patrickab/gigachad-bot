"use client"

import { CodeBlock } from "@gigachad/chat-ui"
import { memo, useState, type ReactNode } from "react"
import { ChevronDown } from "lucide-react"
import { cn } from "@/lib/utils"
import { renderToolCall } from "@/features/conversation/tools/renderers"
import { ToolCallRow } from "@/features/conversation/tools/ToolCallRow"
import { toolPayload } from "@/features/chat/toolCalls"
import { toolDisplay } from "@/features/chat/tools"
import { LearnSessionLink } from "@/features/learn/LearnSessionLink"
import type { ToolCallRecord } from "@/lib/domain/types"
import { useThemeName } from "@/hooks/useThemeName"
import { PlotElement } from "@/components/content/PlotElement"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"
import { SandboxOutputElement } from "./messages/SandboxOutputElement"
import { chatToolRun } from "./toolRun"
import "@gigachad/chat-ui/code.css"

/** A text-button toggle that reveals `children` below it. */
function Disclosure({ label, children }: { label: string; children: ReactNode }) {
  const [open, setOpen] = useState(false)
  return <>
    <button
      type="button"
      onClick={() => setOpen((isOpen) => !isOpen)}
      aria-expanded={open}
      className="flex items-center gap-1 text-xs font-medium text-ink-subtle"
    >
      <ChevronDown className={cn("h-4 w-4 transition-transform", open && "rotate-180")} aria-hidden="true" />
      <span>{label}</span>
    </button>
    {open && children}
  </>
}

/** The call's script, highlighted in the shell's theme. */
function ScriptBlock({ script }: { script: string }) {
  return <CodeBlock codeString={script} language="python" theme={useThemeName()} />
}

/** What a family shows under its row, visible without opening it; null when there is nothing (yet). */
function chatToolBody(record: ToolCallRecord): ReactNode {
  const shown = toolPayload(record)
  switch (shown.family) {
    case "mindmap":
    case "diagram": {
      const markdown = shown.family === "mindmap" ? shown.content : shown.mermaid ? `\`\`\`mermaid\n${shown.mermaid}\n\`\`\`` : null
      return markdown ? <LaTeXMarkdown content={markdown} /> : record.error ? <p className="text-xs text-danger">{record.error}</p> : null
    }
    case "plot":
      return <>
        {shown.brief && (
          <Disclosure label="Briefing">
            <p className="mt-2 whitespace-pre-line text-sm text-ink-muted">{shown.brief}</p>
          </Disclosure>
        )}
        {shown.figure && <PlotElement figure={shown.figure} />}
        {shown.script && (
          <Disclosure label="Code">
            <div className="mt-2"><ScriptBlock script={shown.script} /></div>
          </Disclosure>
        )}
        {record.error && <p className="text-xs text-danger">{record.error}</p>}
      </>
    case "workspace":
      // Outputs describe a finished run; while running the row's spinner is the whole story.
      if (record.status === "running") return null
      return shown.sandbox && shown.sandbox.outputs.length > 0 ? (
        shown.sandbox.outputs.map((output, i) => <SandboxOutputElement key={output.display_id ?? i} output={output} />)
      ) : shown.sandbox?.status === "failed" || record.status === "error" ? (
        <p className="text-xs text-danger">{record.error ?? "The workspace agent run failed."}</p>
      ) : (
        <p className="text-xs text-ink-faint">Workspace updated.</p>
      )
    case "learn":
      return shown.session && <LearnSessionLink sessionId={shown.session.sessionId} title={shown.session.title} projectSlug={shown.session.projectSlug} />
    case "sources":
      if (shown.sources.length === 0 && shown.costs === null) return null
      return <>
        {shown.costs !== null && <p className="text-[10px] tabular-nums text-ink-faint">cost ${shown.costs.toFixed(4)}</p>}
        {shown.sources.length > 0 && (
          <ul className="space-y-1">
            {shown.sources.map((source) => (
              <li key={source.url} className="flex min-w-0 items-baseline gap-2 text-[10px]">
                {source.label && <span className="shrink-0 font-medium text-ink-subtle">[{source.label}]</span>}
                <a
                  href={source.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="min-w-0 truncate text-ink-muted underline decoration-divider-strong hover:text-ink"
                >
                  {source.title || source.url}
                </a>
              </li>
            ))}
          </ul>
        )}
      </>
    case "generic":
      return null
  }
}

/** One chat tool call as Pi's tool row, through the registered renderers, then the family's result under it. Re-renders only when its record is replaced. */
export const ChatToolCall = memo(function ChatToolCall({ record }: { record: ToolCallRecord }) {
  const { call, run, stages } = chatToolRun(record)
  const body = chatToolBody(record)
  return <>
    {renderToolCall(call, run, () => <ToolCallRow call={call} run={run} display={toolDisplay(call.name)} stages={stages} />)}
    {body ? <div className="space-y-3 py-2 pl-7">{body}</div> : null}
  </>
})
