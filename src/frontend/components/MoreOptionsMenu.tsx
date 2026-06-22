"use client"

import { useState, useRef, useCallback } from "react"
import { AnimatePresence, motion } from "framer-motion"
import { MoreHorizontal, BookOpen } from "lucide-react"
import { LLMParams } from "./LLMParams"
import { PillButton } from "./PillButton"
import { ParamSlider } from "./ParamSlider"
import { StyledSelect } from "./StyledSelect"
import { cn } from "@/lib/utils"
import { REASONING_LEVELS } from "@/lib/config"
import { useClickOutside } from "@/hooks/useClickOutside"
import { useModeState } from "@/hooks/useModeState"
import type { TabConfig } from "@/components/TabManager"

interface MoreOptionsMenuProps {
  prompts: Record<string, string>
  config: TabConfig
  onConfigChange: (config: Partial<TabConfig>) => void
}

const FOCUS_MODES: { value: string; label: string }[] = [
  { value: "webSearch", label: "Web" },
  { value: "academicSearch", label: "Academic" },
  { value: "redditSearch", label: "Reddit" },
  { value: "youtubeSearch", label: "YouTube" },
  { value: "wolframAlphaSearch", label: "Wolfram" },
]

const OPTIMIZATION_MODES: { value: string; label: string }[] = [
  { value: "speed", label: "Speed" },
  { value: "balanced", label: "Balanced" },
  { value: "quality", label: "Quality" },
]

function Toggle({ on, onChange }: { on: boolean; onChange: () => void }) {
  return (
    <button
      role="switch"
      aria-checked={on}
      onClick={onChange}
      className={cn("relative h-5 w-9 rounded-full transition-colors", on ? "bg-ink-muted" : "bg-surface-elevated")}
    >
      <span className={cn("absolute top-0.5 h-4 w-4 rounded-full bg-ink transition-transform", on ? "left-[18px]" : "left-0.5")} />
    </button>
  )
}

export function MoreOptionsMenu({
  prompts,
  config,
  onConfigChange,
}: MoreOptionsMenuProps) {
  const { researchEnabled, searchEnabled } = useModeState()
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  const close = useCallback(() => { setOpen(false) }, [])
  useClickOutside(ref, close)

  return (
    <div className="relative z-50" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center justify-center rounded-lg p-2 text-ink-muted hover:text-ink hover:bg-surface transition-colors"
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
            className="absolute right-0 top-full mt-2 w-72 rounded-xl border border-divider bg-paper p-3 shadow-[var(--shadow-xl)] flex flex-col gap-4"
          >
            {!researchEnabled && (
              <>
                {/* System Prompt */}
                <div className="space-y-1.5">
                  <div className="flex items-center gap-2 text-xs font-medium text-ink-subtle">
                    <BookOpen className="h-3.5 w-3.5" />
                    System Prompt
                  </div>
                  <StyledSelect
                    options={Object.keys(prompts).map((p) => ({ value: p, label: p }))}
                    value={config.selectedPrompt ?? ""}
                    onChange={(v) => onConfigChange({ selectedPrompt: v || null })}
                  />
                </div>

                {/* LLM Parameters */}
                <div className="pt-2 border-t border-divider/50">
                  <LLMParams
                    temperature={config.temperature}
                    onTemperatureChange={(v) => onConfigChange({ temperature: v })}
                  />
                </div>

                {/* Image Downscale */}
                <div className="pt-2 border-t border-divider/50">
                  <label className="flex items-center justify-between cursor-pointer">
                    <span className="text-xs text-ink-subtle">Downscale images</span>
                    <button
                      role="switch"
                      aria-checked={config.downscaleImages}
                      onClick={() => onConfigChange({ downscaleImages: !config.downscaleImages })}
                      className={cn(
                        "relative h-5 w-9 rounded-full transition-colors",
                        config.downscaleImages ? "bg-ink-muted" : "bg-surface-elevated"
                      )}
                    >
                      <span className={cn(
                        "absolute top-0.5 h-4 w-4 rounded-full bg-ink transition-transform",
                        config.downscaleImages ? "left-[18px]" : "left-0.5"
                      )} />
                    </button>
                  </label>
                </div>
              </>
            )}

            {searchEnabled && (
              <div className="pt-2 border-t border-divider/50 space-y-3">
                {/* Focus mode (single-select, authoritative) */}
                <div className="space-y-1.5">
                  <div className="text-xs font-medium text-ink-subtle">Focus</div>
                  <div className="flex flex-wrap gap-1">
                    {FOCUS_MODES.map((m) => (
                      <PillButton
                        key={m.value}
                        accent="muted"
                        active={config.searchFocusMode === m.value}
                        onClick={() => onConfigChange({ searchFocusMode: m.value })}
                      >
                        {m.label}
                      </PillButton>
                    ))}
                  </div>
                </div>

                {/* Optimization mode */}
                <div className="space-y-1.5">
                  <div className="text-xs font-medium text-ink-subtle">Optimization</div>
                  <div className="flex gap-1">
                    {OPTIMIZATION_MODES.map((m) => (
                      <PillButton
                        key={m.value}
                        accent="muted"
                        active={config.searchOptimization === m.value}
                        onClick={() => onConfigChange({ searchOptimization: m.value })}
                      >
                        {m.label}
                      </PillButton>
                    ))}
                  </div>
                </div>

                {/* Image / video search (default off) */}
                <label className="flex items-center justify-between cursor-pointer">
                  <span className="text-xs text-ink-subtle">Image search</span>
                  <Toggle on={config.searchImages} onChange={() => onConfigChange({ searchImages: !config.searchImages })} />
                </label>
                <label className="flex items-center justify-between cursor-pointer">
                  <span className="text-xs text-ink-subtle">Video search</span>
                  <Toggle on={config.searchVideos} onChange={() => onConfigChange({ searchVideos: !config.searchVideos })} />
                </label>

                {/* Domain filter helper — prepended verbatim to the query */}
                <div className="space-y-1">
                  <span className="text-[10px] text-ink-faint">Domain filter</span>
                  <input
                    value={config.searchDomain}
                    onChange={(e) => onConfigChange({ searchDomain: e.target.value })}
                    placeholder="site:nature.com -reddit.com"
                    spellCheck={false}
                    className="w-full rounded-md border border-divider bg-surface px-2 py-1 text-xs text-ink placeholder:text-ink-faint outline-none focus:border-divider-strong"
                  />
                </div>

                {/* System instructions */}
                <div className="space-y-1">
                  <span className="text-[10px] text-ink-faint">System instructions</span>
                  <textarea
                    value={config.searchSystemInstructions}
                    onChange={(e) => onConfigChange({ searchSystemInstructions: e.target.value })}
                    placeholder="Optional guidance for the answer…"
                    rows={2}
                    spellCheck={false}
                    className="w-full resize-none rounded-md border border-divider bg-surface px-2 py-1 text-xs text-ink placeholder:text-ink-faint outline-none focus:border-divider-strong"
                  />
                </div>
              </div>
            )}

            {researchEnabled && config.researchDepth !== undefined && (
              <div className="pt-2 border-t border-divider/50 space-y-3">
                <div className="flex items-center gap-2 text-xs font-medium text-ink">
                  <span className="h-1.5 w-1.5 rounded-full bg-ink" />
                  Research Parameters
                </div>
                <ParamSlider
                  label="Depth"
                  value={config.researchDepth}
                  onChange={(v) => onConfigChange({ researchDepth: v })}
                  min={1}
                  max={4}
                  step={1}
                  accent="accent-ink"
                />
                <ParamSlider
                  label="Breadth"
                  value={config.researchBreadth}
                  onChange={(v) => onConfigChange({ researchBreadth: v })}
                  min={2}
                  max={6}
                  step={1}
                  accent="accent-ink"
                />
                <div className="space-y-1">
                  <span className="text-[10px] text-ink-faint">Reasoning effort</span>
                  <StyledSelect
                    options={REASONING_LEVELS.map((l) => ({
                      value: l,
                      label: l === "none" ? "None" : l.charAt(0).toUpperCase() + l.slice(1),
                    }))}
                    value={config.researchReasoning}
                    onChange={(v) => onConfigChange({ researchReasoning: v })}
                  />
                </div>
                <div className="space-y-1">
                  <span className="text-[10px] text-ink-faint">Report type</span>
                  <StyledSelect
                    options={[
                      { value: "deep", label: "Deep (recursive)" },
                      { value: "research_report", label: "Standard (single-pass)" },
                    ]}
                    value={config.researchReportType}
                    onChange={(v) => onConfigChange({ researchReportType: v })}
                  />
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}