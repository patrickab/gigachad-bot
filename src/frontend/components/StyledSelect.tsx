"use client"

interface StyledSelectProps {
  options: { value: string; label: string }[]
  value: string
  onChange: (v: string) => void
}

export function StyledSelect({ options, value, onChange }: StyledSelectProps) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full rounded-lg border border-divider bg-surface px-3 py-2 text-xs text-ink outline-none focus:border-divider-strong transition-colors"
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>{o.label}</option>
      ))}
    </select>
  )
}