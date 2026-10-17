"use client"

import { useEffect, useState } from "react"
import { cn } from "@/lib/utils"

// TEMPORARY: lets the composer design proposals be compared live. Delete this file,
// its use in ChatInput, and the losing variants once one is picked.
export const COMPOSER_VARIANTS = [
  { id: "classic", label: "Current" },
  { id: "line", label: "Line" },
  { id: "capsule", label: "Capsule" },
  { id: "ghost", label: "Ghost" },
] as const

export type ComposerVariant = (typeof COMPOSER_VARIANTS)[number]["id"]

const STORAGE_KEY = "composer_variant"

/** Selected variant persisted in localStorage. Starts at "classic" and hydrates after mount to avoid an SSR mismatch. */
export function useComposerVariant(): [ComposerVariant, (variant: ComposerVariant) => void] {
  const [variant, setVariant] = useState<ComposerVariant>("classic")

  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY)
      if (COMPOSER_VARIANTS.some((v) => v.id === stored)) setVariant(stored as ComposerVariant)
    } catch {}
  }, [])

  return [variant, (next) => { setVariant(next); try { localStorage.setItem(STORAGE_KEY, next) } catch {} }]
}

export function ComposerVariantSwitcher({ value, onChange }: { value: ComposerVariant; onChange: (variant: ComposerVariant) => void }) {
  return (
    <div role="radiogroup" aria-label="Composer design" className="absolute right-2 top-full z-30 mt-1.5 flex items-center gap-0.5 rounded-full border border-divider/60 bg-surface p-0.5 text-[10px]">
      {COMPOSER_VARIANTS.map((v) => (
        <button
          key={v.id}
          role="radio"
          aria-checked={value === v.id}
          onClick={() => onChange(v.id)}
          className={cn("rounded-full px-2 py-0.5 transition-colors", value === v.id ? "bg-ink text-paper" : "text-ink-subtle hover:text-ink")}
        >
          {v.label}
        </button>
      ))}
    </div>
  )
}
