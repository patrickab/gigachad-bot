"use client"

import { motion, AnimatePresence } from "framer-motion"
import { ChevronDown, Check } from "lucide-react"
import type { ModelsResponse, Provider } from "@/lib/types"
import { cn } from "@/lib/utils"
import { useState, useRef, useEffect } from "react"

interface ResearchModelsBarProps {
  models: ModelsResponse | null
  fastModel: string
  smartModel: string
  strategicModel: string
  onFastModelChange: (m: string) => void
  onSmartModelChange: (m: string) => void
  onStrategicModelChange: (m: string) => void
}

const PROVIDER_PREFIXES: Record<Provider, string> = {
  Ollama: "ollama/",
  Gemini: "gemini/",
  OpenAI: "openai/",
}

const TIERS = ["fast", "smart", "strategic"] as const
type Tier = (typeof TIERS)[number]

function displayName(m: string) {
  for (const prefix of Object.values(PROVIDER_PREFIXES)) {
    if (m.startsWith(prefix)) return m.slice(prefix.length)
  }
  return m
}

export function ResearchModelsBar({
  models,
  fastModel,
  smartModel,
  strategicModel,
  onFastModelChange,
  onSmartModelChange,
  onStrategicModelChange,
}: ResearchModelsBarProps) {
  const [open, setOpen] = useState(false)
  const [activeTier, setActiveTier] = useState<Tier>("fast")
  const [activeTab, setActiveTab] = useState<Provider>("Ollama")
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    function handler(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener("mousedown", handler)
    return () => document.removeEventListener("mousedown", handler)
  }, [])

  const available = models
    ? {
        Ollama: models.ollama ?? [],
        Gemini: models.gemini ?? [],
        OpenAI: models.openai ?? [],
      }
    : { Ollama: [], Gemini: [], OpenAI: [] }

  const activeProviders = (Object.keys(available) as Provider[]).filter((k) => available[k].length > 0)
  const currentModels = available[activeTab] ?? []

  const tierState: Record<Tier, { value: string; onChange: (m: string) => void }> = {
    fast: { value: fastModel, onChange: onFastModelChange },
    smart: { value: smartModel, onChange: onSmartModelChange },
    strategic: { value: strategicModel, onChange: onStrategicModelChange },
  }
  const selectedModel = tierState[activeTier].value
  const onModelSelect = tierState[activeTier].onChange

  return (
    <div className="relative z-50 flex flex-col items-start" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 rounded-lg px-2 py-1 text-sm font-medium text-amber-400 hover:text-amber-300 hover:bg-zinc-900 transition-colors"
      >
        <span>Model</span>
        <ChevronDown className="h-3 w-3 text-amber-500" />
      </button>

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
              <div className="flex rounded-lg bg-zinc-950 p-0.5">
                {TIERS.map((tier) => (
                  <button
                    key={tier}
                    onClick={() => setActiveTier(tier)}
                    className={cn(
                      "flex-1 rounded-md px-2 py-1.5 text-xs font-medium capitalize transition-colors",
                      tier === activeTier
                        ? "bg-zinc-800 text-zinc-100 shadow-sm"
                        : "text-zinc-500 hover:text-zinc-300"
                    )}
                  >
                    {tier}
                  </button>
                ))}
              </div>
              <div className="flex rounded-lg bg-zinc-950 p-0.5">
                {activeProviders.map((p) => (
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
                      onModelSelect(m)
                      setOpen(false)
                    }}
                    className={cn(
                      "flex w-full items-center justify-between rounded-md px-2 py-2 text-sm transition-colors",
                      m === selectedModel
                        ? "bg-amber-500/10 text-amber-400"
                        : "text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100"
                    )}
                  >
                    <span className="truncate">{displayName(m)}</span>
                    {m === selectedModel && <Check className="h-3.5 w-3.5 shrink-0" />}
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
