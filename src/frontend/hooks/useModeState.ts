"use client"

import { useCallback, useState } from "react"

export type AppMode = "chat" | "research" | "search" | "ocr"

export interface UseModeStateReturn {
  mode: AppMode
  researchEnabled: boolean
  morphicSearchEnabled: boolean
  ocrEnabled: boolean
  toggleResearch: () => void
  toggleMorphicSearch: () => void
  toggleOCR: () => void
  setMode: (mode: AppMode) => void
}

export function useModeState(): UseModeStateReturn {
  const [mode, setMode] = useState<AppMode>("chat")

  const toggleResearch = useCallback(() => {
    setMode((prev) => prev === "research" ? "chat" : "research")
  }, [])

  const toggleMorphicSearch = useCallback(() => {
    setMode((prev) => prev === "search" ? "chat" : "search")
  }, [])

  const toggleOCR = useCallback(() => {
    setMode((prev) => prev === "ocr" ? "chat" : "ocr")
  }, [])

  return {
    mode,
    researchEnabled: mode === "research",
    morphicSearchEnabled: mode === "search",
    ocrEnabled: mode === "ocr",
    toggleResearch,
    toggleMorphicSearch,
    toggleOCR,
    setMode,
  }
}