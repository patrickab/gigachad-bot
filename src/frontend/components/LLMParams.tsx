"use client"

import { SlidersHorizontal } from "lucide-react"

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
        <div className="space-y-1">
          <div className="flex justify-between text-[10px] text-zinc-600">
            <span>Temperature</span>
            <span>{temperature.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="0"
            max="2"
            step="0.05"
            value={temperature}
            onChange={(e) => onTemperatureChange(Number.parseFloat(e.target.value))}
            className="w-full h-1 bg-zinc-800 rounded-full appearance-none cursor-pointer accent-sky-500"
          />
        </div>
        <div className="space-y-1">
          <div className="flex justify-between text-[10px] text-zinc-600">
            <span>Top-p</span>
            <span>{topP.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={topP}
            onChange={(e) => onTopPChange(Number.parseFloat(e.target.value))}
            className="w-full h-1 bg-zinc-800 rounded-full appearance-none cursor-pointer accent-sky-500"
          />
        </div>
      </div>
    </div>
  )
}
