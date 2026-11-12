"use client"

import { useResource } from "@/hooks/useResource"
import { reasoningLevelsResource } from "@/lib/backend/resources"
import { PopoverSelector } from "@/components/ui/PopoverSelector"

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
  return useResource("Reasoning levels", reasoningLevelsResource(model), Boolean(model))
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
  if (!levels?.length) return null

  return <PopoverSelector
    triggerLabel={`Reasoning: ${levelLabel(reasoningEffort)}`}
    triggerContent={<ReasoningGlyph level={reasoningEffort} />}
    triggerClassName="px-2.5"
    label="Reasoning"
    categories={[{ key: "levels", label: "Reasoning", items: levels }]}
    itemKey={(level) => level}
    isSelected={(level) => level === reasoningEffort}
    renderItem={(level) => <><ReasoningGlyph level={level} /><span className="flex-1 truncate capitalize">{levelLabel(level)}</span></>}
    onSelect={onReasoningChange}
    placement={placement}
    panelClassName="w-40"
  />
}
