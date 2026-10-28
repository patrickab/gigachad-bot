"use client"

import { useMemo } from "react"
import { cn } from "@/lib/utils"
import { useOmp } from "./OmpProvider"
import { sessionBranches } from "./branches"

/** In-session branches of the session shown in the frame, styled like a chat's branch preview. Selecting one switches the frame to it. */
export function OmpBranchTree({ sessionId }: { sessionId: string }) {
  const { branchTree, selectLeaf } = useOmp()
  const tree = branchTree?.sessionId === sessionId ? branchTree : null
  const branches = useMemo(() => (tree ? sessionBranches(tree.tree) : []), [tree])
  if (!tree || branches.length < 2) return null

  return (
    <div role="group" aria-label="Session branches" className="relative">
      <div className="pointer-events-none absolute bottom-0 left-[11px] top-0 w-px bg-divider" />
      {branches.map((branch) => {
        const active = branch.leafId === tree.activeLeafId
        return (
          <div key={branch.leafId} className="relative">
            <div className="pointer-events-none absolute left-[11px] top-2 h-px w-[10px] bg-divider" />
            <button
              type="button"
              aria-current={active ? "true" : undefined}
              onClick={() => selectLeaf(sessionId, branch.leafId)}
              className={cn("relative z-[1] ml-6 flex w-[calc(100%-1.5rem)] items-center gap-1.5 rounded-sm py-0.5 pr-1 text-left text-[11px] transition-colors",
                active ? "bg-surface-elevated font-semibold text-ink" : "text-ink-muted hover:bg-surface-elevated/50 hover:text-ink")}
            >
              <span className="min-w-0 flex-1 truncate">{branch.label}</span>
            </button>
          </div>
        )
      })}
    </div>
  )
}
