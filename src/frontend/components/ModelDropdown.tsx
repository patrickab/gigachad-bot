"use client"

import { motion, AnimatePresence } from "framer-motion"
import { Check } from "lucide-react"
import type { Provider } from "@/lib/types"
import { cn } from "@/lib/utils"
import { displayName, groupByProvider, activeProviders } from "@/lib/models"
import { useState, useRef, useCallback, type ReactNode } from "react"
import type { ModelsResponse } from "@/lib/types"
import { useClickOutside } from "@/hooks/useClickOutside"

interface Tab {
  key: string
  label: string
}

interface ModelDropdownProps {
  models: ModelsResponse | null
  selectedModel: string
  onSelect: (model: string) => void
  accent?: "sky" | "amber"
  extraTabs?: Tab[]
  activeExtraTab?: string
  onExtraTabChange?: (key: string) => void
  children: (props: { open: boolean; onToggle: () => void }) => ReactNode
}

const ACCENT_CLASSES = {
  sky: {
    selected: "bg-sky-500/10 text-sky-400",
    check: "h-4 w-4",
  },
  amber: {
    selected: "bg-amber-500/10 text-amber-400",
    check: "h-3.5 w-3.5",
  },
} as const

export function ModelDropdown({
  models,
  selectedModel,
  onSelect,
  accent = "sky",
  extraTabs,
  activeExtraTab,
  onExtraTabChange,
  children,
}: ModelDropdownProps) {
  const [open, setOpen] = useState(false)
  const [activeTab, setActiveTab] = useState<Provider>("Ollama")
  const ref = useRef<HTMLDivElement>(null)

  const close = useCallback(() => setOpen(false), [])
  useClickOutside(ref, close)

  const available = groupByProvider(models)
  const providers = activeProviders(available)
  const currentModels = available[activeTab] ?? []
  const accentClasses = ACCENT_CLASSES[accent]

  return (
    <div className="relative z-50 flex flex-col items-start" ref={ref}>
      {children({ open, onToggle: () => setOpen((o) => !o) })}

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.15 }}
            className="absolute left-0 top-full mt-1 w-72 rounded-xl border border-zinc-800 bg-zinc-900 shadow-2xl flex flex-col overflow-hidden"
          >
            <div className="p-1.5 border-b border-zinc-800 shrink-0 space-y-1.5">
              {extraTabs && extraTabs.length > 0 && (
                <div className="flex rounded-lg bg-zinc-950 p-0.5">
                  {extraTabs.map((tab) => (
                    <button
                      key={tab.key}
                      onClick={() => onExtraTabChange?.(tab.key)}
                      className={cn(
                        "flex-1 rounded-md px-2 py-1.5 text-xs font-medium capitalize transition-colors",
                        tab.key === activeExtraTab
                          ? "bg-zinc-800 text-zinc-100 shadow-sm"
                          : "text-zinc-500 hover:text-zinc-300"
                      )}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>
              )}
              <div className="flex rounded-lg bg-zinc-950 p-0.5">
                {providers.map((p) => (
                  <button
                    key={p}
                    onClick={() => setActiveTab(p)}
                    className={cn(
                      "flex-1 rounded-md px-2 py-1.5 text-xs font-medium transition-colors",
                      p === activeTab
                        ? "bg-zinc-800 text-zinc-100 shadow-sm"
                        : "text-zinc-500 hover:text-zinc-300"
                    )}
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>

            <div className="max-h-[50vh] overflow-y-auto p-1.5 space-y-0.5">
              {currentModels.length === 0 ? (
                <div className="py-4 text-center text-xs text-zinc-500">No models available</div>
              ) : (
                currentModels.map((m) => (
                  <button
                    key={m}
                    onClick={() => {
                      onSelect(m)
                      setOpen(false)
                    }}
                    className={cn(
                      "flex w-full items-center justify-between rounded-md px-2 py-2 text-sm transition-colors",
                      m === selectedModel
                        ? accentClasses.selected
                        : "text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100"
                    )}
                  >
                    <span className="truncate">{displayName(m)}</span>
                    {m === selectedModel && <Check className={cn("shrink-0", accentClasses.check)} />}
                  </button>
                ))
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}