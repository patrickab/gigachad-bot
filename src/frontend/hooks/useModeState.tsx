"use client"

import { createContext, useCallback, useContext, useState, type ReactNode } from "react"

/** OCR still hijacks the composer; web search, deep research, and plot are tools the model
 *  calls itself. */
export type AppMode = "chat" | "ocr"

export const WEB_SEARCH_TOOL = "web_search"
export const DEEP_RESEARCH_TOOL = "deep_research"
export const PLOT_TOOL = "plot"

export interface ModeState {
  mode: AppMode
  /** Tool names the model may call on the next send. */
  enabledTools: string[]
  toggleTool: (name: string) => void
  researchEnabled: boolean
  searchEnabled: boolean
  plotEnabled: boolean
  ocrEnabled: boolean
  toggleResearch: () => void
  toggleSearch: () => void
  togglePlot: () => void
  toggleOCR: () => void
  setMode: (mode: AppMode) => void
}

const ModeContext = createContext<ModeState | null>(null)

export function useModeState(): ModeState {
  const ctx = useContext(ModeContext)
  if (!ctx) throw new Error("useModeState must be used within ModeProvider")
  return ctx
}

export function ModeProvider({ children }: { children: ReactNode }) {
  const [mode, setMode] = useState<AppMode>("chat")
  // Available by default: "search the web for X" must work without the user first arming a
  // pill. The pills withdraw a tool, they do not hand it out.
  const [enabledTools, setEnabledTools] = useState<string[]>([WEB_SEARCH_TOOL, DEEP_RESEARCH_TOOL, PLOT_TOOL])

  // Tools are independent: enabling search must not disable research, because one turn
  // can legitimately call both.
  const toggleTool = useCallback((name: string) => {
    setEnabledTools((prev) => (prev.includes(name) ? prev.filter((t) => t !== name) : [...prev, name]))
  }, [])

  const toggleResearch = useCallback(() => toggleTool(DEEP_RESEARCH_TOOL), [toggleTool])
  const toggleSearch = useCallback(() => toggleTool(WEB_SEARCH_TOOL), [toggleTool])
  const togglePlot = useCallback(() => toggleTool(PLOT_TOOL), [toggleTool])

  const toggleOCR = useCallback(() => {
    setMode((prev) => prev === "ocr" ? "chat" : "ocr")
  }, [])

  const value: ModeState = {
    mode,
    enabledTools,
    toggleTool,
    researchEnabled: enabledTools.includes(DEEP_RESEARCH_TOOL),
    searchEnabled: enabledTools.includes(WEB_SEARCH_TOOL),
    plotEnabled: enabledTools.includes(PLOT_TOOL),
    ocrEnabled: mode === "ocr",
    toggleResearch,
    toggleSearch,
    toggleOCR,
    togglePlot,
    setMode,
  }

  return (
    <ModeContext.Provider value={value}>
      {children}
    </ModeContext.Provider>
  )
}
