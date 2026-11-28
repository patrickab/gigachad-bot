import { cn } from "@/lib/utils"

/** A segmented switch: one `aria-pressed` button per option, the one matching `value` styled `activeClass`. */
export function ViewSwitch<T extends string | boolean>({ label = "Show as", value, onChange, options, className = "flex gap-px rounded-md bg-surface p-0.5", buttonClass = "rounded px-2 py-0.5 text-xs transition-colors", activeClass = "bg-surface-elevated text-ink", inactiveClass = "text-ink-subtle hover:text-ink" }: {
  label?: string
  value: T
  onChange: (value: T) => void
  options: readonly (readonly [T, string])[]
  className?: string
  buttonClass?: string
  activeClass?: string
  inactiveClass?: string
}) {
  return (
    <div role="group" aria-label={label} className={className}>
      {options.map(([option, optionLabel]) => (
        <button
          key={optionLabel}
          type="button"
          aria-pressed={value === option}
          onClick={() => onChange(option)}
          className={cn(buttonClass, value === option ? activeClass : inactiveClass)}
        >
          {optionLabel}
        </button>
      ))}
    </div>
  )
}
