"use client"

import { Brain, Check } from "lucide-react"
import { useCallback, useEffect, useRef, useState } from "react"
import { useClickOutside } from "@/hooks/useClickOutside"
import { fetchReasoningLevels } from "@/lib/api"
import { cn } from "@/lib/utils"

/** Shared across every instance so switching tabs/models reuses a lookup
 *  already made this session instead of refetching. */
const levelsCache = new Map<string, string[]>()

interface ReasoningSelectorProps {
  model: string
  reasoningEffort: string
  onReasoningChange: (v: string) => void
  /** Side of the trigger the menu opens on. */
  placement?: "above" | "below"
}

function levelLabel(level: string): string {
  return level === "none" ? "off" : level
}

/** Pie glyph filling with the effort level, slashed circle for off. */
function ReasoningGlyph({ level }: { level: string }) {
  return <span className="reasoning-glyph" data-level={levelLabel(level)} aria-hidden="true" />
}

export function ReasoningSelector({ model, reasoningEffort, onReasoningChange, placement = "above" }: ReasoningSelectorProps) {
  const [levels, setLevels] = useState<string[] | undefined>(() => levelsCache.get(model))

  useEffect(() => {
    const cached = levelsCache.get(model)
    if (cached !== undefined) {
      setLevels(cached)
      return
    }
    setLevels(undefined)
    let cancelled = false
    fetchReasoningLevels(model)
      .then(({ levels: fetched }) => {
        levelsCache.set(model, fetched)
        if (!cancelled) setLevels(fetched)
      })
      .catch(() => {
        if (!cancelled) setLevels([])
      })
    return () => {
      cancelled = true
    }
  }, [model])

  // A level the new model lacks (e.g. `xhigh` after switching models) falls back to
  // `medium`, which every reasoning model offers. A model that can't reason forces `none`, and
  // the level it replaced is restored once a reasoning model is selected again.
  // The ref keeps an unstable `onReasoningChange` from re-running the effect.
  const onReasoningChangeRef = useRef(onReasoningChange)
  onReasoningChangeRef.current = onReasoningChange
  const effortBeforeForcedOffRef = useRef<string | null>(null)
  useEffect(() => {
    if (levels === undefined) return
    if (levels.length === 0) {
      if (reasoningEffort === "none") return
      effortBeforeForcedOffRef.current = reasoningEffort
      onReasoningChangeRef.current("none")
      return
    }
    const wantedEffort = effortBeforeForcedOffRef.current ?? reasoningEffort
    effortBeforeForcedOffRef.current = null
    const supportedEffort = levels.includes(wantedEffort) ? wantedEffort : "medium"
    if (supportedEffort !== reasoningEffort) onReasoningChangeRef.current(supportedEffort)
  }, [levels, reasoningEffort])

  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  const close = useCallback(() => setOpen(false), [])
  useClickOutside(ref, close)

  if (!levels?.length) return null

  return (
    <div ref={ref} className="relative min-w-0">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-label={`Reasoning: ${levelLabel(reasoningEffort)}`}
        aria-haspopup="menu"
        aria-expanded={open}
        className={cn(
          "flex h-7 items-center gap-1 rounded-full px-2.5 text-xs text-ink-muted transition-colors hover:bg-hover hover:text-ink",
          open && "text-ink",
        )}
      >
        <ReasoningGlyph level={reasoningEffort} />
      </button>
      {open && (
        <div role="menu" aria-label="Reasoning" className={cn("absolute left-0 z-50 w-44 overflow-hidden rounded-xl border border-divider bg-paper shadow-[var(--shadow-xl)]", placement === "below" ? "top-full mt-2" : "bottom-full mb-2")}>
          <div className="flex h-8 items-center gap-2 border-b border-divider px-2.5">
            <Brain className="h-3 w-3 text-ink-muted" />
            <span className="text-[11px] font-semibold text-ink">Reasoning</span>
            <span className="ml-auto font-mono text-[10px] tabular-nums text-ink-subtle">{levels.length}</span>
          </div>
          <div className="flex flex-col p-1">
            {levels.map((level) => {
              const active = level === reasoningEffort
              return (
                <button
                  key={level}
                  type="button"
                  role="menuitemradio"
                  aria-checked={active}
                  onClick={() => { onReasoningChange(level); close() }}
                  className={cn(
                    "flex h-8 w-full items-center gap-2 rounded-lg px-2 text-left text-xs capitalize transition-colors",
                    active ? "bg-surface-elevated font-semibold text-ink" : "text-ink-muted hover:bg-hover hover:text-ink",
                  )}
                >
                  <span className="flex w-3.5 shrink-0 justify-center">{active && <Check className="h-3 w-3" />}</span>
                  <ReasoningGlyph level={level} />
                  <span className="truncate">{levelLabel(level)}</span>
                </button>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
