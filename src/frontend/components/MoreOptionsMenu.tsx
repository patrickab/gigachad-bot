"use client"

import { useState, useRef, useCallback } from "react"
import { AnimatePresence, motion } from "framer-motion"
import { MoreHorizontal, Save, RotateCcw, Loader2, X, BookOpen } from "lucide-react"
import { LLMParams } from "./LLMParams"
import { PillButton } from "./PillButton"
import { ParamSlider } from "./ParamSlider"
import { StyledSelect } from "./StyledSelect"
import type { Message } from "@/lib/types"
import { cn } from "@/lib/utils"
import { REASONING_LEVELS } from "@/lib/config"
import { useClickOutside } from "@/hooks/useClickOutside"
import { useSettings } from "@/contexts/SettingsContext"

interface MoreOptionsMenuProps {
  prompts: string[]
  onRefresh: () => void
  onReset: () => void
  onSave: (name: string, messages: Message[]) => Promise<void>
  messages: Message[]
  models?: { ollama: string[]; gemini: string[]; openai: string[] } | null
}

export function MoreOptionsMenu({
  prompts,
  onRefresh,
  onReset,
  onSave,
  messages,
  models,
}: MoreOptionsMenuProps) {
  const settings = useSettings()
  const [open, setOpen] = useState(false)
  const [showSaveInput, setShowSaveInput] = useState(false)
  const [saveName, setSaveName] = useState("")
  const [saving, setSaving] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  const close = useCallback(() => { setOpen(false); setShowSaveInput(false) }, [])
  useClickOutside(ref, close)

  async function handleSave() {
    if (!saveName.trim()) return
    setSaving(true)
    await onSave(`${saveName.trim()}.json`, messages)
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
              <StyledSelect
                options={prompts.map((p) => ({ value: p, label: p }))}
                value={settings.selectedPrompt ?? ""}
                onChange={settings.setSelectedPrompt}
              />
            </div>

            {/* LLM Parameters */}
            <div className="pt-2 border-t border-zinc-800/50">
              <LLMParams
                temperature={settings.temperature}
                onTemperatureChange={settings.setTemperature}
                topP={settings.topP}
                onTopPChange={settings.setTopP}
              />
            </div>

            {/* Image Downscale */}
            <div className="pt-2 border-t border-zinc-800/50">
              <label className="flex items-center justify-between cursor-pointer">
                <span className="text-xs text-zinc-500">Downscale images</span>
                <button
                  role="switch"
                  aria-checked={settings.downscaleImages}
                  onClick={() => settings.setDownscaleImages(!settings.downscaleImages)}
                  className={cn(
                    "relative h-5 w-9 rounded-full transition-colors",
                    settings.downscaleImages ? "bg-zinc-600" : "bg-zinc-800"
                  )}
                >
                  <span className={cn(
                    "absolute top-0.5 h-4 w-4 rounded-full bg-white transition-transform",
                    settings.downscaleImages ? "left-[18px]" : "left-0.5"
                  )} />
                </button>
              </label>
            </div>

            {settings.ocrModel !== undefined && (
              <div className="pt-2 border-t border-zinc-800/50 space-y-2">
                <div className="flex items-center gap-2 text-xs font-medium text-emerald-400">
                  <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                  LaTeX OCR
                </div>
                <div className="space-y-1">
                  <span className="text-[10px] text-zinc-600">Model</span>
                  <StyledSelect
                    options={[
                      ...(models?.ollama ?? []).filter((m) => m.startsWith("qwen") || m.includes("vl") || m.includes("vision")).map((m) => ({ value: m, label: m })),
                      ...(models?.gemini ?? []).map((m) => ({ value: m, label: m })),
                      ...(models?.openai ?? []).filter((m) => m.includes("vision") || m.includes("gpt-4o")).map((m) => ({ value: m, label: m })),
                    ]}
                    value={settings.ocrModel}
                    onChange={settings.setOCRModel}
                  />
                </div>
              </div>
            )}

            {settings.searchDepth !== undefined && (
              <div className="pt-2 border-t border-zinc-800/50 space-y-2">
                <div className="flex items-center gap-2 text-xs font-medium text-sky-400">
                  <span className="h-1.5 w-1.5 rounded-full bg-sky-400" />
                  Search Depth
                </div>
                <div className="flex gap-1">
                  <PillButton accent="zinc" active={settings.searchDepth === "quick"} onClick={() => settings.setSearchDepth("quick")}>
                    Quick
                  </PillButton>
                  <PillButton accent="zinc" active={settings.searchDepth === "adaptive"} onClick={() => settings.setSearchDepth("adaptive")}>
                    Adaptive
                  </PillButton>
                </div>
              </div>
            )}

            {settings.researchDepth !== undefined && (
              <div className="pt-2 border-t border-zinc-800/50 space-y-3">
                <div className="flex items-center gap-2 text-xs font-medium text-amber-400">
                  <span className="h-1.5 w-1.5 rounded-full bg-amber-400" />
                  Research Parameters
                </div>
                <ParamSlider
                  label="Depth"
                  value={settings.researchDepth}
                  onChange={settings.setResearchDepth}
                  min={1}
                  max={4}
                  step={1}
                  accent="accent-amber-500"
                />
                <ParamSlider
                  label="Breadth"
                  value={settings.researchBreadth}
                  onChange={settings.setResearchBreadth}
                  min={2}
                  max={6}
                  step={1}
                  accent="accent-amber-500"
                />
                <div className="space-y-1">
                  <span className="text-[10px] text-zinc-600">Reasoning effort</span>
                  <StyledSelect
                    options={REASONING_LEVELS.map((l) => ({
                      value: l,
                      label: l === "none" ? "None" : l.charAt(0).toUpperCase() + l.slice(1),
                    }))}
                    value={settings.researchReasoning}
                    onChange={settings.setResearchReasoning}
                  />
                </div>
                <div className="space-y-1">
                  <span className="text-[10px] text-zinc-600">Report type</span>
                  <StyledSelect
                    options={[
                      { value: "deep", label: "Deep (recursive)" },
                      { value: "research_report", label: "Standard (single-pass)" },
                    ]}
                    value={settings.researchReportType}
                    onChange={settings.setResearchReportType}
                  />
                </div>
              </div>
            )}

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