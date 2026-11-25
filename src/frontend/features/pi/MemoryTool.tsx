"use client"

import { Brain } from "lucide-react"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"
import { MEMORY_MCP_PREFIX, MEMORY_TOOL_NAMES, memoryToolView, type MemoryOpLine } from "@/lib/domain/memoryTool"
import { ActivityRow, ResultDisclosure } from "./FactoryTool"
import { registerToolRenderer } from "./renderers"

/** An `apply_memory_ops` call's ops, one line each: marker, id or path, text. */
export function MemoryOpLines({ ops }: { ops: MemoryOpLine[] }) {
  return (
    <ul className="m-0 grid list-none grid-cols-[auto_auto_minmax(0,1fr)] gap-x-2 p-0 text-ink-muted">
      {ops.map((op, index) => (
        <li key={index} className="contents" data-memory-op={op.marker}>
          <span className="text-ink">{op.marker}</span>
          <span className="text-ink-subtle">{op.ref}</span>
          <span className="break-words">{op.text}</span>
        </li>
      ))}
    </ul>
  )
}

for (const name of MEMORY_TOOL_NAMES) {
  for (const registered of [name, `${MEMORY_MCP_PREFIX}${name}`]) {
    registerToolRenderer(registered, (call, run, next) => {
      const view = memoryToolView(call.name, call.arguments)
      if (!view) return next()
      return (
        <ActivityRow run={run} icon={Brain} tool="memory" summary={view.summary}>
          {view.ops.length > 0 && <MemoryOpLines ops={view.ops} />}
          <ResultDisclosure
            run={run}
            render={view.kind === "read" ? (text) => <div className="pl-5"><LaTeXMarkdown content={text} /></div> : undefined}
          />
        </ActivityRow>
      )
    })
  }
}
