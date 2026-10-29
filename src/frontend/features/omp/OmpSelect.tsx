"use client"

import { useEffect, useId, useMemo, useRef, useState, type KeyboardEvent } from "react"
import { Search } from "lucide-react"
import { FloatingWindow } from "@/components/ui/FloatingWindow"
import { useResetOnChange } from "@/hooks/useResetOnChange"
import { fuzzyScore } from "@/lib/fuzzy"
import { cn } from "@/lib/utils"
import type { SelectRequest } from "@gigachad/chat-ui/protocol"

/** A label's last path segment and what leads to it: `./scripts/a.sh` -> `a.sh`, `./scripts`. A plain label is all name. */
function splitLabel(label: string): { name: string; lead: string } {
  const slash = label.lastIndexOf("/")
  return slash === -1 ? { name: label, lead: "" } : { name: label.slice(slash + 1), lead: label.slice(0, slash) }
}

/** Options matching `query`, best first, a name match outranks the same match in its lead. No query keeps the extension's order. */
function rankOptions(options: string[], query: string): string[] {
  const trimmedQuery = query.trim()
  if (!trimmedQuery) return options
  return options
    .map((option) => {
      const byName = fuzzyScore(splitLabel(option).name, trimmedQuery)
      const byLabel = fuzzyScore(option, trimmedQuery)
      return { option, score: Math.max(byName === null ? -Infinity : byName + 1, byLabel ?? -Infinity) }
    })
    .filter(({ score }) => score > -Infinity)
    .sort((left, right) => right.score - left.score)
    .map(({ option }) => option)
}

/** An omp extension's `select` (e.g. `/files`) as a fuzzy finder: type to filter, arrows move, Enter picks, Escape cancels. */
export function OmpSelect({ request, onAnswer }: { request: SelectRequest; onAnswer: (value: string | null) => void }) {
  const [query, setQuery] = useState("")
  const [activeIndex, setActiveIndex] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLDivElement>(null)
  const listId = useId()
  const matches = useMemo(() => rankOptions(request.options, query), [request.options, query])
  useResetOnChange(query, () => setActiveIndex(0))

  useEffect(() => { inputRef.current?.focus() }, [])

  useEffect(() => {
    listRef.current?.querySelector(`[data-index="${activeIndex}"]`)?.scrollIntoView({ block: "nearest" })
  }, [activeIndex])

  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.nativeEvent.isComposing) return
    if (event.key === "ArrowDown") {
      event.preventDefault()
      setActiveIndex((index) => Math.min(index + 1, matches.length - 1))
    } else if (event.key === "ArrowUp") {
      event.preventDefault()
      setActiveIndex((index) => Math.max(index - 1, 0))
    } else if (event.key === "Enter") {
      event.preventDefault()
      const picked = matches[activeIndex]
      if (picked !== undefined) onAnswer(picked)
    }
  }

  return (
    // Top-anchored (self-start overrides the backdrop's centering), so filtering shrinks the list without moving the input
    <FloatingWindow label={request.title} onClose={() => onAnswer(null)} panelClassName="mt-[15vh] h-auto max-h-[min(60vh,32rem)] max-w-xl self-start rounded-2xl">
      <div className="flex items-center gap-2.5 border-b border-divider/30 px-4 py-3">
        <Search className="h-4 w-4 shrink-0 text-ink-faint" aria-hidden />
        <input
          ref={inputRef}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={onKeyDown}
          placeholder={request.title}
          aria-label={request.title}
          role="combobox"
          aria-expanded
          aria-controls={listId}
          aria-activedescendant={matches.length > 0 ? `${listId}-${activeIndex}` : undefined}
          spellCheck={false}
          autoComplete="off"
          className="min-w-0 flex-1 bg-transparent text-sm text-ink outline-none placeholder:text-ink-faint"
        />
      </div>
      <div ref={listRef} id={listId} role="listbox" aria-label={request.title} className="min-h-0 flex-1 overflow-y-auto py-1.5">
        {matches.length === 0 && <div className="px-4 py-6 text-center text-xs text-ink-faint">No matches</div>}
        {matches.map((option, index) => {
          const { name, lead } = splitLabel(option)
          const active = index === activeIndex
          return (
            <div
              key={option}
              id={`${listId}-${index}`}
              role="option"
              aria-selected={active}
              data-index={index}
              // Move, not enter: a list scrolled under a resting pointer must not steal the arrow-key selection
              onMouseMove={() => setActiveIndex(index)}
              onClick={() => onAnswer(option)}
              className={cn(
                "mx-1.5 flex cursor-pointer items-baseline gap-3 rounded-lg px-2.5 py-1.5 text-xs transition-colors",
                active ? "bg-surface-elevated text-ink" : "text-ink-muted",
              )}
            >
              <span className="max-w-[60%] shrink-0 truncate">{name}</span>
              {lead && <span className="min-w-0 flex-1 truncate text-right text-[10px] text-ink-faint">{lead}</span>}
            </div>
          )
        })}
      </div>
    </FloatingWindow>
  )
}
