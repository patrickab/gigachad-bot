"use client"

import { useState, useRef, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { MoreHorizontal, Save, RotateCcw, Loader2, X, BookOpen } from "lucide-react"
import { LLMParams } from "./LLMParams"
import { saveChatHistory } from "@/lib/api"
import { cn } from "@/lib/utils"

interface MoreOptionsMenuProps {
  prompts: string[]
  selectedPrompt: string | null
  onPromptSelect: (prompt: string) => void
  temperature: number
  onTemperatureChange: (v: number) => void
  topP: number
  onTopPChange: (v: number) => void
  onRefresh: () => void
  onReset: () => void
  researchEnabled?: boolean
  researchDepth?: number
  onResearchDepthChange?: (v: number) => void
  researchBreadth?: number
  onResearchBreadthChange?: (v: number) => void
  researchReasoning?: string
  onResearchReasoningChange?: (v: string) => void
  researchReportType?: string
  onResearchReportTypeChange?: (v: string) => void
  webSearchEnabled?: boolean
  webSearchNumQueries?: number
  onWebSearchNumQueriesChange?: (v: number) => void
  webSearchResultsPerQuery?: number
  onWebSearchResultsPerQueryChange?: (v: number) => void
}

export function MoreOptionsMenu({
  prompts,
  selectedPrompt,
  onPromptSelect,
  temperature,
  onTemperatureChange,
  topP,
  onTopPChange,
  onRefresh,
  onReset,
  researchEnabled,
  researchDepth,
  onResearchDepthChange,
  researchBreadth,
  onResearchBreadthChange,
  researchReasoning,
  onResearchReasoningChange,
  researchReportType,
  onResearchReportTypeChange,
  webSearchEnabled,
  webSearchNumQueries,
  onWebSearchNumQueriesChange,
  webSearchResultsPerQuery,
  onWebSearchResultsPerQueryChange,
}: MoreOptionsMenuProps) {
  const [open, setOpen] = useState(false)
  const [showSaveInput, setShowSaveInput] = useState(false)
  const [saveName, setSaveName] = useState("")
  const [saving, setSaving] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    function handler(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false)
        setShowSaveInput(false)
      }
    }
    document.addEventListener("mousedown", handler)
    return () => document.removeEventListener("mousedown", handler)
  }, [])

  async function handleSave() {
    if (!saveName.trim()) return
    setSaving(true)
    await saveChatHistory(`${saveName.trim()}.json`)
    setSaveName("")
    setShowSaveInput(false)
    setSaving(false)
    onRefresh()
    setOpen(false)
  }

  return (
    <div className="relative z-50" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center justify-center rounded-lg p-2 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900 transition-colors"
      >
        <MoreHorizontal className="h-5 w-5" />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -4, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -4, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className="absolute right-0 top-full mt-2 w-72 rounded-xl border border-zinc-800 bg-zinc-950 p-3 shadow-2xl flex flex-col gap-4"
          >
            {/* System Prompt */}
            <div className="space-y-1.5">
              <div className="flex items-center gap-2 text-xs font-medium text-zinc-500">
                <BookOpen className="h-3.5 w-3.5" />
                System Prompt
              </div>
              <select
                value={selectedPrompt ?? ""}
                onChange={(e) => onPromptSelect(e.target.value)}
                className="w-full rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 text-xs text-zinc-300 outline-none focus:border-zinc-700 transition-colors"
              >
                {prompts.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </div>

            {/* LLM Parameters */}
            <div className="pt-2 border-t border-zinc-800/50">
              <LLMParams
                temperature={temperature}
                onTemperatureChange={onTemperatureChange}
                topP={topP}
                onTopPChange={onTopPChange}
              />
            </div>

            {webSearchEnabled && (
              <div className="pt-2 border-t border-zinc-800/50 space-y-3">
                <div className="flex items-center gap-2 text-xs font-medium text-sky-400">
                  <span className="h-1.5 w-1.5 rounded-full bg-sky-400" />
                  Web Search
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between text-[10px] text-zinc-600">
                    <span>Query count</span>
                    <span>{webSearchNumQueries}</span>
                  </div>
                  <input
                    type="range"
                    min="1" max="5" step="1"
                    value={webSearchNumQueries}
                    onChange={(e) => onWebSearchNumQueriesChange?.(Number(e.target.value))}
                    className="w-full h-1 bg-zinc-800 rounded-full appearance-none cursor-pointer accent-sky-500"
                  />
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between text-[10px] text-zinc-600">
                    <span>Results/query</span>
                    <span>{webSearchResultsPerQuery}</span>
                  </div>
                  <input
                    type="range"
                    min="1" max="10" step="1"
                    value={webSearchResultsPerQuery}
                    onChange={(e) => onWebSearchResultsPerQueryChange?.(Number(e.target.value))}
                    className="w-full h-1 bg-zinc-800 rounded-full appearance-none cursor-pointer accent-sky-500"
                  />
                </div>
              </div>
            )}

            {researchEnabled && (
              <div className="pt-2 border-t border-zinc-800/50 space-y-3">
                <div className="flex items-center gap-2 text-xs font-medium text-amber-400">
                  <span className="h-1.5 w-1.5 rounded-full bg-amber-400" />
                  Research Parameters
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between text-[10px] text-zinc-600">
                    <span>Depth</span>
                    <span>{researchDepth}</span>
                  </div>
                  <input
                    type="range"
                    min="1" max="4" step="1"
                    value={researchDepth}
                    onChange={(e) => onResearchDepthChange?.(Number(e.target.value))}
                    className="w-full h-1 bg-zinc-800 rounded-full appearance-none cursor-pointer accent-amber-500"
                  />
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between text-[10px] text-zinc-600">
                    <span>Breadth</span>
                    <span>{researchBreadth}</span>
                  </div>
                  <input
                    type="range"
                    min="2" max="6" step="1"
                    value={researchBreadth}
                    onChange={(e) => onResearchBreadthChange?.(Number(e.target.value))}
                    className="w-full h-1 bg-zinc-800 rounded-full appearance-none cursor-pointer accent-amber-500"
                  />
                </div>
                <div className="space-y-1">
                  <span className="text-[10px] text-zinc-600">Reasoning effort</span>
                  <select
                    value={researchReasoning}
                    onChange={(e) => onResearchReasoningChange?.(e.target.value)}
                    className="w-full rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 text-xs text-zinc-300 outline-none focus:border-zinc-700"
                  >
                    <option value="none">None</option>
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                  </select>
                </div>
                <div className="space-y-1">
                  <span className="text-[10px] text-zinc-600">Report type</span>
                  <select
                    value={researchReportType}
                    onChange={(e) => onResearchReportTypeChange?.(e.target.value)}
                    className="w-full rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 text-xs text-zinc-300 outline-none focus:border-zinc-700"
                  >
                    <option value="deep">Deep (recursive)</option>
                    <option value="research_report">Standard (single-pass)</option>
                  </select>
                </div>
              </div>
            )}

            {/* Actions */}
            <div className="pt-2 border-t border-zinc-800/50 space-y-1">
              {showSaveInput ? (
                <div className="flex items-center gap-1 mb-2">
                  <input
                    autoFocus
                    value={saveName}
                    onChange={(e) => setSaveName(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") handleSave()
                      if (e.key === "Escape") setShowSaveInput(false)
                    }}
                    placeholder="chat name..."
                    className="flex-1 rounded-md border border-zinc-800 bg-zinc-900 px-2 py-1.5 text-xs text-zinc-200 outline-none focus:border-zinc-700 placeholder:text-zinc-600"
                  />
                  <button
                    onClick={handleSave}
                    disabled={saving || !saveName.trim()}
                    className="rounded-md p-1.5 text-zinc-400 hover:text-emerald-400 hover:bg-zinc-900 disabled:opacity-30 transition-colors"
                  >
                    {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}
                  </button>
                  <button
                    onClick={() => setShowSaveInput(false)}
                    className="rounded-md p-1.5 text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900 transition-colors"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => setShowSaveInput(true)}
                  className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-xs text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200 transition-colors"
                >
                  <Save className="h-3.5 w-3.5" />
                  Save Chat History
                </button>
              )}

              <button
                onClick={() => {
                  onReset()
                  setOpen(false)
                }}
                className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-xs text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200 transition-colors"
              >
                <RotateCcw className="h-3.5 w-3.5" />
                Reset History
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
