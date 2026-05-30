"use client"

interface ParamSliderProps {
  label: string
  value: number
  onChange: (v: number) => void
  min?: number
  max?: number
  step?: number
  accent?: string
}

export function ParamSlider({
  label,
  value,
  onChange,
  min = 0,
  max = 1,
  step = 0.05,
  accent = "accent-sky-500",
}: ParamSliderProps) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-[10px] text-zinc-600">
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
        className={`w-full h-1 bg-zinc-800 rounded-full appearance-none cursor-pointer ${accent}`}
      />
    </div>
  )
}