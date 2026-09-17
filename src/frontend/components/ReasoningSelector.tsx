"use client"

import { useEffect, useRef, useState } from "react"
import { fetchReasoningSupport } from "@/lib/api"
import { REASONING_LEVELS } from "@/lib/config"

/** Shared across every instance so switching tabs/models reuses a lookup
 *  already made this session instead of refetching. */
const supportCache = new Map<string, boolean>()

interface ReasoningSelectorProps {
  model: string
  reasoningEffort: string
  onReasoningChange: (v: string) => void
}

export function ReasoningSelector({ model, reasoningEffort, onReasoningChange }: ReasoningSelectorProps) {
  const [supported, setSupported] = useState<boolean | undefined>(() => supportCache.get(model))

  useEffect(() => {
    const cached = supportCache.get(model)
    if (cached !== undefined) {
      setSupported(cached)
      return
    }
    setSupported(undefined)
    let cancelled = false
    fetchReasoningSupport(model)
      .then(({ supports_reasoning }) => {
        supportCache.set(model, supports_reasoning)
        if (!cancelled) setSupported(supports_reasoning)
      })
      .catch(() => {
        if (!cancelled) setSupported(false)
      })
    return () => {
      cancelled = true
    }
  }, [model])

  const onReasoningChangeRef = useRef(onReasoningChange)
  onReasoningChangeRef.current = onReasoningChange
  useEffect(() => {
    if (supported === false && reasoningEffort !== "none") onReasoningChangeRef.current("none")
  }, [supported, reasoningEffort])

  if (supported !== true) return null

  return (
    <>
      <div className="w-px h-4 bg-surface-elevated" />
      <div className="flex rounded-md bg-surface/50 p-0.5 border border-divider/50 h-[24px] items-center">
        {REASONING_LEVELS.map((level) => (
          <button
            key={level}
            onClick={() => onReasoningChange(level)}
            className={`px-1.5 text-[10px] font-medium rounded-sm transition-colors h-full flex items-center ${
              level === reasoningEffort
                ? "bg-surface-elevated text-ink shadow-[var(--shadow-sm)] ring-1 ring-divider"
                : "text-ink-subtle hover:text-ink"
            }`}
          >
            {level === "none" ? "off" : level}
          </button>
        ))}
      </div>
    </>
  )
}
