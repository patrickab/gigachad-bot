"use client"

import { useState } from "react"
import { ChevronDown } from "lucide-react"
import type { ModelsResponse } from "@/lib/types"
import { ModelDropdown } from "./ModelDropdown"

interface ResearchModelsBarProps {
  models: ModelsResponse | null
  fastModel: string
  smartModel: string
  strategicModel: string
  onFastModelChange: (m: string) => void
  onSmartModelChange: (m: string) => void
  onStrategicModelChange: (m: string) => void
}

const TIERS = ["fast", "smart", "strategic"] as const
type Tier = (typeof TIERS)[number]

export function ResearchModelsBar({
  models,
  fastModel,
  smartModel,
  strategicModel,
  onFastModelChange,
  onSmartModelChange,
  onStrategicModelChange,
}: ResearchModelsBarProps) {
  const [activeTier, setActiveTier] = useState<Tier>("fast")

  const tierState: Record<Tier, { value: string; onChange: (m: string) => void }> = {
    fast: { value: fastModel, onChange: onFastModelChange },
    smart: { value: smartModel, onChange: onSmartModelChange },
    strategic: { value: strategicModel, onChange: onStrategicModelChange },
  }

  const selectedModel = tierState[activeTier].value
  const onModelSelect = tierState[activeTier].onChange

  return (
    <ModelDropdown
      models={models}
      selectedModel={selectedModel}
      onSelect={onModelSelect}
      accent="sky"
      extraTabs={TIERS.map((t) => ({ key: t, label: t }))}
      activeExtraTab={activeTier}
      onExtraTabChange={(key) => setActiveTier(key as Tier)}
    >
      {({ onToggle }) => (
        <button
          onClick={onToggle}
          className="flex items-center gap-1.5 rounded-lg px-2 py-1 text-sm font-medium text-ink hover:text-ink hover:bg-surface transition-colors"
        >
          <span>Model</span>
          <ChevronDown className="h-3 w-3 text-ink" />
        </button>
      )}
    </ModelDropdown>
  )
}
