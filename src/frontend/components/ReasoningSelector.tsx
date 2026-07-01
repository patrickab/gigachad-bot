"use client"

import { REASONING_LEVELS } from "@/lib/config"

interface ReasoningSelectorProps {
  reasoningEffort: string
  onReasoningChange: (v: string) => void
}

export function ReasoningSelector({ reasoningEffort, onReasoningChange }: ReasoningSelectorProps) {
  return (
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
  )
}
