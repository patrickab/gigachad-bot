"use client"

import { ChevronDown } from "lucide-react"
import type { ModelsResponse } from "@/lib/types"
import { displayName } from "@/lib/models"
import { ModelDropdown } from "./ModelDropdown"
import { Skeleton } from "./Skeleton"

interface ModelSelectorProps {
  models: ModelsResponse | null
  selectedModel: string
  onSelect: (model: string) => void
}

export function ModelSelector({ models, selectedModel, onSelect }: ModelSelectorProps) {
  if (models === null) {
    return (
      <div className="flex flex-col gap-1.5 px-2">
        <Skeleton className="h-5 w-16" />
        <Skeleton className="h-3 w-28" />
      </div>
    )
  }

  return (
    <ModelDropdown
      models={models}
      selectedModel={selectedModel}
      onSelect={onSelect}
      accent="sky"
    >
      {({ onToggle }) => (
        <>
          <button
            onClick={onToggle}
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
        </>
      )}
    </ModelDropdown>
  )
}
