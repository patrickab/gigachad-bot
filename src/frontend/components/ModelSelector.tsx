"use client"

import { motion, AnimatePresence } from "framer-motion"
import { ChevronDown, Check } from "lucide-react"
import type { ModelsResponse, Provider } from "@/lib/types"
import { cn } from "@/lib/utils"
import { useState, useRef, useEffect } from "react"
import { Skeleton } from "./Skeleton"

interface ModelSelectorProps {
  models: ModelsResponse | null
  selectedModel: string
  onSelect: (model: string) => void
}

const PROVIDER_PREFIXES: Record<Provider, string> = {
  Ollama: "ollama/",
  Gemini: "gemini/",
  OpenAI: "openai/",
}

export function ModelSelector({ models, selectedModel, onSelect }: ModelSelectorProps) {
  const [open, setOpen] = useState(false)
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

  function displayName(m: string) {
    for (const prefix of Object.values(PROVIDER_PREFIXES)) {
      if (m.startsWith(prefix)) return m.slice(prefix.length)
    }
    return m
  }

  return (
    <div className="relative z-50 flex flex-col items-start" ref={ref}>
      {models === null ? (
        <div className="flex flex-col gap-1.5 px-2">
          <Skeleton className="h-5 w-16" />
          <Skeleton className="h-3 w-28" />
        </div>
      ) : (
        <>
          <button
            onClick={() => setOpen(!open)}
            className="flex items-center gap-1.5 rounded-lg px-2 py-1 text-sm font-medium text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900 transition-colors"
          >
            <span>Model</span>
            <ChevronDown className="h-3 w-3 text-zinc-500" />
          </button>
          
          {selectedModel && (
            <span className="text-[10px] italic text-zinc-500 px-2 truncate max-w-[150px]">
              {displayName(selectedModel)}
            </span>
          )}

          <AnimatePresence>
            {open && (
              <motion.div
                initial={{ opacity: 0, y: -4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                transition={{ duration: 0.15 }}
                className="absolute left-0 top-full mt-1 w-72 rounded-xl border border-zinc-800 bg-zinc-900 shadow-2xl flex flex-col overflow-hidden"
              >
                <div className="p-1.5 border-b border-zinc-800 shrink-0">
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
                          onSelect(m)
                          setOpen(false)
                        }}
                        className={cn(
                          "flex w-full items-center justify-between rounded-md px-2 py-2 text-sm transition-colors",
                          m === selectedModel
                            ? "bg-sky-500/10 text-sky-400"
                            : "text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100"
                        )}
                      >
                        <span className="truncate">{displayName(m)}</span>
                        {m === selectedModel && <Check className="h-4 w-4 shrink-0" />}
                      </button>
                    ))
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </>
      )}
    </div>
  )
}

