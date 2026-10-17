"use client"

import { Check } from "lucide-react"
import { useCallback, useEffect, useRef, useState } from "react"
import { useClickOutside } from "@/hooks/useClickOutside"
import { fetchReasoningLevels } from "@/lib/api"
import { cn } from "@/lib/utils"

/** Shared across every instance so switching tabs/models reuses a lookup
 *  already made this session instead of refetching. */
const levelsCache = new Map<string, string[]>()

interface ReasoningSelectorProps {
  /** Levels from `useReasoningLevels`, undefined while loading. */
  levels: string[] | undefined
  /** Effective effort from `effectiveReasoningEffort`. */
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

/** Reasoning levels `model` accepts, `[]` when it can't reason, undefined while loading. */
export function useReasoningLevels(model: string): string[] | undefined {
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

  return levels
}

/** Effort actually used for the model, derived so the stored choice survives model switches.
 *
 *  - Model can't reason -> `none`, stored effort untouched for the next reasoning model
 *  - Level the model lacks (e.g. `xhigh`) -> `medium`, which every reasoning model offers
 *  - Levels still loading -> stored effort, the backend drops unsupported params
 */
export function effectiveReasoningEffort(levels: string[] | undefined, storedEffort: string): string {
  if (levels === undefined) return storedEffort
  if (levels.length === 0) return "none"
  return levels.includes(storedEffort) ? storedEffort : "medium"
}

export function ReasoningSelector({ levels, reasoningEffort, onReasoningChange, placement = "above" }: ReasoningSelectorProps) {
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
        <div role="menu" aria-label="Reasoning" className={cn("absolute left-0 z-50 w-40 rounded-xl border border-divider bg-surface p-1.5 shadow-[var(--shadow-lg)]", placement === "below" ? "top-full mt-2" : "bottom-full mb-2")}>
          <div className="flex flex-col gap-0.5">
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
                    "flex w-full items-center gap-2 rounded-md px-2 py-2 text-left text-sm capitalize text-ink transition-colors",
                    active ? "bg-surface-elevated" : "hover:bg-surface-elevated",
                  )}
                >
                  <ReasoningGlyph level={level} />
                  <span className="flex-1 truncate">{levelLabel(level)}</span>
                  {active && <Check className="h-3.5 w-3.5 shrink-0" />}
                </button>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
