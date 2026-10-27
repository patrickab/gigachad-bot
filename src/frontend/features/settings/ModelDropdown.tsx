"use client"

import { Cpu, Settings2 } from "lucide-react"
import { displayName, orderProviders, qualifiedModelIds } from "@/lib/models"
import type { ModelDefaults, ModelProvider, ModelsResponse } from "@/lib/domain/types"
import { errorMessage } from "@/lib/utils"
import { PopoverSelector, PopoverSelectorAction } from "@/components/ui/PopoverSelector"
import { Skeleton } from "@/components/ui/Skeleton"
import { ModelSettingsError, ModelSettingsPanel, useModelSettings } from "@/features/settings/ModelSettingsPanel"
import { useCompactLabel } from "@/features/settings/useCompactLabel"

interface ProviderTab { label: string; models: string[] }

interface ModelDropdownProps {
  models: ModelsResponse | null
  loadError?: string | null
  selectedModel: string
  onSelect: (model: string) => void
  onProvidersChange: (providers: ModelProvider[]) => Promise<void>
  onTabOrderChange?: (order: string[]) => Promise<void>
  onDefaultsChange: (defaults: ModelDefaults) => Promise<void>
  /** Side of the trigger the panel opens on. */
  placement?: "above" | "below"
  /** Collapse the name when the expanded pill would cover this column. */
  chatColumn?: HTMLElement | null
}

/** Bare model name for the compact trigger: `litellm_proxy/anthropic/claude-sonnet-5` -> `claude-sonnet-5`. */
function shortName(model: string): string {
  return model.slice(model.lastIndexOf("/") + 1)
}

export function ModelDropdown({ models, loadError, selectedModel, onSelect, onProvidersChange, onTabOrderChange, onDefaultsChange, placement = "above", chatColumn }: ModelDropdownProps) {
  const { compact, rootRef, labelRef, contentRef } = useCompactLabel(chatColumn, models === null)
  const settings = useModelSettings(onProvidersChange, onDefaultsChange)

  if (models === null) return loadError
    ? <span className="mx-2 text-xs text-ink-muted">Models unavailable</span>
    : <Skeleton className="mx-2 h-4 w-28" />

  const byDisplayName = (a: string, b: string) => displayName(a).localeCompare(displayName(b))
  const available: ProviderTab[] = [
    { label: "Ollama", models: [...models.ollama].sort(byDisplayName) },
    ...models.providers.map((provider) => ({ label: provider.label, models: qualifiedModelIds(provider).sort(byDisplayName) })),
  ]
  const orderedProviders = orderProviders(available.filter((provider) => provider.models.length > 0), models.tab_order)
  const choices = available.flatMap((provider) => provider.models)
  const reorderProviders = (labels: string[]) => void onTabOrderChange?.(labels).catch((reason) => settings.setError(errorMessage(reason, "Could not save provider order")))
  const fullLabel = selectedModel ? displayName(selectedModel) : "Select model"

  return <div ref={rootRef} data-compact={compact} className="group/model min-w-0">
    <PopoverSelector
    triggerLabel={`Model: ${fullLabel}`}
    triggerClassName="h-8.5 gap-0 bg-paper px-[11px] hover:bg-surface-elevated"
    triggerContent={<>
      <Cpu className="h-3 w-3 shrink-0" />
      <span ref={labelRef} className="block max-w-[146px] overflow-hidden opacity-100 transition-[max-width,opacity] duration-250 ease-[cubic-bezier(0.22,1,0.36,1)] motion-reduce:transition-none group-[[data-compact=true]:not(:hover):not(:focus-within)]/model:max-w-0 group-[[data-compact=true]:not(:hover):not(:focus-within)]/model:opacity-0">
        <span ref={contentRef} className="block w-max pl-1.5">
          <span className="block max-w-[140px] truncate">{selectedModel ? shortName(selectedModel) : fullLabel}</span>
        </span>
      </span>
    </>}
    label="Model"
    categories={orderedProviders.map((provider) => ({ key: provider.label, label: provider.label, items: provider.models }))}
    itemKey={(model) => model}
    isSelected={(model) => model === selectedModel}
    renderItem={(model) => <span className="flex-1 truncate">{displayName(model)}</span>}
    onSelect={onSelect}
    onReorderCategories={reorderProviders}
    footer={<><ModelSettingsError error={settings.error} /><PopoverSelectorAction icon={Settings2} label="Manage models" onClick={() => settings.setConfiguring(true)} /></>}
    view={settings.configuring ? <ModelSettingsPanel settings={settings} models={models} choices={choices} /> : undefined}
    emptyText="No models added yet"
    placement={placement}
    panelClassName={settings.configuring ? "w-max min-w-72" : undefined}
    onClose={settings.close}
    />
  </div>
}
