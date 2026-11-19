"use client"

import { useMemo } from "react"
import { Cpu } from "lucide-react"
import { PopoverSelector } from "@/components/ui/PopoverSelector"
import type { PiModel } from "./types"

/** Identity of a model across Pi's lists and state. */
function modelKey(model: PiModel): string {
  return `${model.provider}/${model.id}`
}

/**
 * The session's model, picked from Pi's `get_available_models`, one tab per provider (D10, F10).
 *
 * `selected` is the session's current model: Pi's default for a new session, nothing selected before Pi reports it. `compact` shortens its name for a side panel.
 */
export function PiModelSelector({ models, selected, onSelect, compact = false }: {
  models: PiModel[]
  selected: PiModel | null
  onSelect: (model: PiModel) => void
  compact?: boolean
}) {
  const categories = useMemo(() => {
    const byProvider = new Map<string, PiModel[]>()
    for (const model of models) byProvider.set(model.provider, [...(byProvider.get(model.provider) ?? []), model])
    return [...byProvider].map(([provider, items]) => ({ key: provider, label: provider, items }))
  }, [models])
  const selectedKey = selected ? modelKey(selected) : null
  const label = selected ? selected.name ?? selected.id : "Default"

  return (
    <PopoverSelector
      triggerLabel={`Model: ${label}`}
      triggerClassName="gap-1.5 px-2.5"
      triggerContent={<><Cpu className="h-3 w-3 shrink-0" aria-hidden="true" /><span className={compact ? "block min-w-0 max-w-24 truncate" : "block max-w-40 truncate"}>{label}</span></>}
      label="Model"
      categories={categories}
      itemKey={modelKey}
      isSelected={(model) => modelKey(model) === selectedKey}
      renderItem={(model) => <span className="flex-1 truncate">{model.name ?? model.id}</span>}
      onSelect={onSelect}
      emptyText="No models available"
      placement="below"
    />
  )
}

/** Pie glyph filling with the level, slashed circle for off (global `.reasoning-glyph` styles). */
function ThinkingGlyph({ level }: { level: string }) {
  return <span className="reasoning-glyph" data-level={level} aria-hidden="true" />
}

/** The session's thinking level, picked from the levels its model offers (`get_available_thinking_levels`). */
export function PiThinkingSelector({ levels, selected, onSelect }: { levels: string[]; selected: string; onSelect: (level: string) => void }) {
  return (
    <PopoverSelector
      triggerLabel={`Thinking: ${selected}`}
      triggerClassName="px-2.5"
      triggerContent={<ThinkingGlyph level={selected} />}
      label="Thinking"
      categories={[{ key: "levels", label: "Thinking", items: levels }]}
      itemKey={(level) => level}
      isSelected={(level) => level === selected}
      renderItem={(level) => <><ThinkingGlyph level={level} /><span className="flex-1 truncate capitalize">{level}</span></>}
      onSelect={onSelect}
      placement="below"
      panelClassName="w-40"
    />
  )
}