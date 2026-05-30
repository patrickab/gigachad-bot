"use client"

import { SlidersHorizontal } from "lucide-react"
import { ParamSlider } from "./ParamSlider"

interface LLMParamsProps {
  temperature: number
  onTemperatureChange: (v: number) => void
  topP: number
  onTopPChange: (v: number) => void
}

export function LLMParams({
  temperature,
  onTemperatureChange,
  topP,
  onTopPChange,
}: LLMParamsProps) {
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 text-xs font-medium text-zinc-500">
        <SlidersHorizontal className="h-3.5 w-3.5" />
        Parameters
      </div>
      <div className="space-y-3">
        <ParamSlider
          label="Temperature"
          value={temperature}
          onChange={onTemperatureChange}
          min={0}
          max={2}
          step={0.05}
        />
        <ParamSlider
          label="Top-p"
          value={topP}
          onChange={onTopPChange}
          min={0}
          max={1}
          step={0.05}
        />
      </div>
    </div>
  )
}