"use client"
import type { CSSProperties } from "react"

interface ParamSliderProps {
  label: string
  value: number
  onChange: (v: number) => void
  min?: number
  max?: number
  step?: number
}

/** Labeled range input. The readout shows two decimals when `step` is fractional, else an integer. */
export function ParamSlider({
  label,
  value,
  onChange,
  min = 0,
  max = 1,
  step = 0.05,
}: ParamSliderProps) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-[10px] text-ink-faint">
        <span>{label}</span>
        <span>{value.toFixed(step < 1 ? 2 : 0)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number.parseFloat(e.target.value))}
        style={{ "--fill": `${((value - min) / (max - min)) * 100}%` } as CSSProperties}
        className="param-slider w-full cursor-pointer"
      />
    </div>
  )
}