"use client"

import { createContext, useCallback, useContext, useState, type ReactNode } from "react"

export type AppMode = "chat" | "research" | "search" | "ocr" | "study"

export interface ModeState {
  mode: AppMode
  researchEnabled: boolean
  searchEnabled: boolean
  ocrEnabled: boolean
  studyEnabled: boolean
  toggleResearch: () => void
  toggleSearch: () => void
  toggleOCR: () => void
  toggleStudy: () => void
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

  const toggleResearch = useCallback(() => {
    setMode((prev) => prev === "research" ? "chat" : "research")
  }, [])

  const toggleSearch = useCallback(() => {
    setMode((prev) => prev === "search" ? "chat" : "search")
  }, [])

  const toggleOCR = useCallback(() => {
    setMode((prev) => prev === "ocr" ? "chat" : "ocr")
  }, [])

  const toggleStudy = useCallback(() => {
    setMode((prev) => prev === "study" ? "chat" : "study")
  }, [])

  const value: ModeState = {
    mode,
    researchEnabled: mode === "research",
    searchEnabled: mode === "search",
    ocrEnabled: mode === "ocr",
    studyEnabled: mode === "study",
    toggleResearch,
    toggleSearch,
    toggleOCR,
    toggleStudy,
    setMode,
  }

  return (
    <ModeContext.Provider value={value}>
      {children}
    </ModeContext.Provider>
  )
}