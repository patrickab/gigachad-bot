"use client"
import type { CSSProperties } from "react"

interface ParamSliderProps {
  label: string
  value: number
  onChange: (v: number) => void
  min: number
  max: number
}

/** Labeled integer range input. */
export function ParamSlider({ label, value, onChange, min, max }: ParamSliderProps) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-[10px] text-ink-faint">
        <span>{label}</span>
        <span>{value.toFixed(0)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={1}
        value={value}
        onChange={(e) => onChange(Number.parseFloat(e.target.value))}
        style={{ "--fill": `${((value - min) / (max - min)) * 100}%` } as CSSProperties}
        className="param-slider w-full cursor-pointer"
      />
    </div>
  )
}