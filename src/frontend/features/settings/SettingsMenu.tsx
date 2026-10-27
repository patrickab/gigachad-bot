"use client"

import { useState } from "react"
import { AnimatePresence, motion } from "framer-motion"
import { Check, ChevronDown, ChevronUp, ImageDown, Settings, BookOpen, Pencil, Telescope, SquareTerminal, Rocket } from "lucide-react"
import { ParamSlider } from "./ParamSlider"
import { StyledSelect } from "@/components/ui/StyledSelect"
import { ErrorText } from "@/components/ui/ErrorText"
import { Field } from "@/components/ui/Field"
import { MenuPopover } from "@/components/ui/MenuPopover"
import { cn } from "@/lib/utils"
import { REASONING_LEVELS } from "@/lib/backend/config"
import { displayName, qualifiedModelIds } from "@/lib/models"
import { useAsyncAction } from "@/hooks/useAsyncAction"
import type { ModelsResponse } from "@/lib/domain/types"
import { useHostAccess, useNeovimSetting, type TabConfig } from "@/contexts/SettingsContext"
import { startRedeploy } from "@/lib/backend/api"

interface SettingsMenuProps {
  prompts: Record<string, string>
  config: TabConfig
  onConfigChange: (config: Partial<TabConfig>) => void
  onEditPrompts?: () => void
  /** Populates the research model-tier selects; the chat model picker owns
   *  its own copy and stays in the topbar. */
  models: ModelsResponse | null
  /** Whether the Deep Research tool is on; its settings only show then. */
  researchEnabled: boolean
}

function ChevronToggle({ ariaLabel, open, onToggle }: { ariaLabel: string; open: boolean; onToggle: () => void }) {
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-expanded={open}
      aria-label={ariaLabel}
      className="p-0.5 rounded text-ink-faint hover:bg-hover hover:text-ink transition-colors"
    >
      <ChevronDown className={cn("h-3.5 w-3.5 transition-transform", open && "rotate-180")} />
    </button>
  )
}

/** Track sits one level below the panel (paper under surface-elevated) in both themes. */
function Switch({ label, checked, onToggle }: { label: string; checked: boolean; onToggle: () => void }) {
  return (
    <button
      type="button"
      role="switch"
      aria-label={label}
      aria-checked={checked}
      onClick={onToggle}
      className={cn("relative h-5 w-9 shrink-0 rounded-full border bg-paper shadow-[var(--inner-highlight)] transition-colors", checked ? "border-divider-strong" : "border-divider")}
    >
      <span className={cn("absolute left-0.5 top-0.5 h-3.5 w-3.5 rounded-full shadow-[var(--shadow-sm)] transition-all", checked ? "translate-x-4 bg-ink-subtle" : "translate-x-0 bg-ink-faint")} />
    </button>
  )
}

export function SettingsMenu({
  prompts,
  config,
  onConfigChange,
  onEditPrompts,
  models,
  researchEnabled,
}: SettingsMenuProps) {
  const [researchOpen, setResearchOpen] = useState(false)
  const [promptOpen, setPromptOpen] = useState(false)
  const neovim = useNeovimSetting()
  const hostAccess = useHostAccess()
  const [redeployStarted, setRedeployStarted] = useState(false)
  const redeploy = useAsyncAction(async () => {
    await startRedeploy()
    setRedeployStarted(true)
  }, "Redeploy failed")

  // Ollama plus every configured provider, so the tiers can pick any model the chat picker
  // can, without pulling in its provider-editing UI.
  const modelChoices = models
    ? [...models.ollama, ...models.providers.flatMap(qualifiedModelIds)].map((id) => ({ value: id, label: displayName(id) }))
    : []

  return (
    <MenuPopover trigger={(open, toggle) => (
      <button aria-label="Settings"
        onClick={toggle}
        className={cn(
          "w-full flex items-center justify-start gap-3 p-2 rounded-lg transition-colors text-ink-muted hover:bg-surface-elevated/60 hover:text-ink",
          open && "bg-surface-elevated/60 text-ink",
        )}
      >
        <Settings className="h-4 w-4 shrink-0" />
        <span className="flex-1 text-left text-sm font-medium truncate">Settings</span>
        <ChevronUp className={cn("h-3.5 w-3.5 text-ink-faint transition-transform", open && "rotate-180")} />
      </button>
    )}>
      {(open, close) => <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: 4, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 4, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className="absolute inset-x-1 bottom-full mb-1 max-h-[70vh] overflow-y-auto rounded-xl bg-surface-elevated p-3 shadow-[var(--shadow-md)] flex flex-col gap-4"
          >
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5 text-xs font-medium text-ink-subtle">
                  <BookOpen className="h-3.5 w-3.5" />
                  System Prompt
                  <ChevronToggle ariaLabel="System Prompt" open={promptOpen} onToggle={() => setPromptOpen((o) => !o)} />
                </div>
                {onEditPrompts && (
                  <button aria-label="Edit prompts" onClick={() => { onEditPrompts(); close() }} className="p-0.5 rounded hover:bg-hover text-ink-faint hover:text-ink transition-colors">
                    <Pencil className="h-3 w-3" />
                  </button>
                )}
              </div>
              {promptOpen && (
                <div role="listbox" aria-label="System Prompt" className="mt-2 flex flex-col gap-0.5">
                  {Object.keys(prompts).map((name) => (
                    <button
                      key={name}
                      type="button"
                      role="option"
                      aria-selected={name === config.selectedPrompt}
                      onClick={() => { onConfigChange({ selectedPrompt: name }); setPromptOpen(false) }}
                      className={cn(
                        "flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-xs transition-colors",
                        name === config.selectedPrompt ? "bg-surface text-ink" : "text-ink-muted hover:bg-surface hover:text-ink",
                      )}
                    >
                      <span className="min-w-0 flex-1 truncate">{name}</span>
                      {name === config.selectedPrompt && <Check className="h-3.5 w-3.5 shrink-0" />}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {researchEnabled && (
              <div className="space-y-1.5 pt-2 border-t border-divider/50">
                <div className="flex items-center gap-1.5 text-xs font-medium text-ink-subtle">
                  <Telescope className="h-3.5 w-3.5" />
                  Deep Research
                  <ChevronToggle ariaLabel="Deep Research" open={researchOpen} onToggle={() => setResearchOpen((o) => !o)} />
                </div>
                {researchOpen && (
                  <div className="space-y-3 pt-3">
                    <ParamSlider
                      label="Depth"
                      value={config.researchDepth}
                      onChange={(v) => onConfigChange({ researchDepth: v })}
                      min={1}
                      max={4}
                      step={1}
                    />
                    <ParamSlider
                      label="Breadth"
                      value={config.researchBreadth}
                      onChange={(v) => onConfigChange({ researchBreadth: v })}
                      min={2}
                      max={6}
                      step={1}
                    />
                    <Field label="Reasoning effort" compact>
                      <StyledSelect
                        options={REASONING_LEVELS.map((l) => ({
                          value: l,
                          label: l === "none" ? "None" : l.charAt(0).toUpperCase() + l.slice(1),
                        }))}
                        value={config.researchReasoning}
                        onChange={(v) => onConfigChange({ researchReasoning: v })}
                      />
                    </Field>
                    <Field label="Fast model" compact>
                      <StyledSelect options={modelChoices} value={config.researchFastModel} onChange={(v) => onConfigChange({ researchFastModel: v })} />
                    </Field>
                    <Field label="Smart model" compact>
                      <StyledSelect options={modelChoices} value={config.researchSmartModel} onChange={(v) => onConfigChange({ researchSmartModel: v })} />
                    </Field>
                    <Field label="Strategic model" compact>
                      <StyledSelect options={modelChoices} value={config.researchStrategicModel} onChange={(v) => onConfigChange({ researchStrategicModel: v })} />
                    </Field>
                    <Field label="Report type" compact>
                      <StyledSelect
                        options={[
                          { value: "deep", label: "Deep (recursive)" },
                          { value: "research_report", label: "Standard (single-pass)" },
                        ]}
                        value={config.researchReportType}
                        onChange={(v) => onConfigChange({ researchReportType: v })}
                      />
                    </Field>
                    {/* Search guidance is only used by the deep-research tool. */}
                    <Field label="Domain filter" compact>
                      <input
                        value={config.searchDomain}
                        onChange={(e) => onConfigChange({ searchDomain: e.target.value })}
                        placeholder="site:nature.com -reddit.com"
                        spellCheck={false}
                        className="w-full rounded-md border border-divider bg-surface px-2 py-1 text-xs text-ink placeholder:text-ink-faint outline-none focus:border-divider-strong"
                      />
                    </Field>
                    <Field label="System instructions" compact>
                      <textarea
                        value={config.searchSystemInstructions}
                        onChange={(e) => onConfigChange({ searchSystemInstructions: e.target.value })}
                        placeholder="Optional guidance for the answer…"
                        rows={2}
                        spellCheck={false}
                        className="w-full resize-none rounded-md border border-divider bg-surface px-2 py-1 text-xs text-ink placeholder:text-ink-faint outline-none focus:border-divider-strong"
                      />
                    </Field>
                  </div>
                )}
              </div>
            )}

            <div className="flex items-center justify-between pt-2 border-t border-divider/50">
              <div className="flex items-center gap-2 text-xs font-medium text-ink-subtle">
                <ImageDown className="h-3.5 w-3.5" />
                Downscale images
              </div>
              <Switch label="Downscale images" checked={config.downscaleImages} onToggle={() => onConfigChange({ downscaleImages: !config.downscaleImages })} />
            </div>

            {neovim.available && (
              <div className="flex items-center justify-between pt-2 border-t border-divider/50">
                <div className="flex items-center gap-2 text-xs font-medium text-ink-subtle">
                  <SquareTerminal className="h-3.5 w-3.5" />
                  Neovim editor
                </div>
                <Switch label="Neovim editor" checked={neovim.enabled} onToggle={() => neovim.setPreferred(!neovim.enabled)} />
              </div>
            )}

            {hostAccess && (
              <div className="flex items-center justify-between pt-2 border-t border-divider/50">
                <div className="flex items-center gap-2 text-xs font-medium text-ink-subtle">
                  <Rocket className="h-3.5 w-3.5" />
                  Redeploy
                </div>
                <button
                  type="button"
                  onClick={() => void redeploy.run()}
                  disabled={redeploy.pending || redeployStarted}
                  className="rounded px-2 py-1 text-xs text-ink-subtle transition-colors hover:bg-hover hover:text-ink disabled:opacity-60"
                >
                  {redeploy.pending ? "Starting…" : redeployStarted ? "Started" : "Redeploy"}
                </button>
              </div>
            )}
            <ErrorText>{redeploy.error}</ErrorText>
          </motion.div>
        )}
      </AnimatePresence>}
    </MenuPopover>
  )
}
