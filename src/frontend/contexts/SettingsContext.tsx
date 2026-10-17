"use client"

import { createContext, useContext, useEffect, useState, type ReactNode } from "react"
import { fetchBackendConfig } from "@/lib/backend/api"

export interface TabConfig {
  selectedModel: string
  selectedPrompt: string | null
  reasoningEffort: string
  downscaleImages: boolean
  researchFastModel: string
  researchSmartModel: string
  researchStrategicModel: string
  researchDepth: number
  researchBreadth: number
  researchReasoning: string
  researchReportType: string
  searchSystemInstructions: string
  searchDomain: string
}

// Selected model starts blank and is seeded from the backend's default chat model
// by `SettingsProvider`. Never hardcode a model name here.
export const DEFAULT_TAB_CONFIG: TabConfig = {
  selectedModel: "",
  selectedPrompt: null,
  reasoningEffort: "medium",
  downscaleImages: true,
  researchFastModel: "",
  researchSmartModel: "",
  researchStrategicModel: "",
  researchDepth: 2,
  researchBreadth: 4,
  researchReasoning: "medium",
  researchReportType: "deep",
  searchSystemInstructions: "",
  searchDomain: "",
}

interface SettingsValue {
  /** Config every new tab starts from. */
  tabDefaults: TabConfig
  ocrModel: string
}

const SettingsContext = createContext<SettingsValue | null>(null)

export function useSettings(): SettingsValue {
  const ctx = useContext(SettingsContext)
  if (!ctx) throw new Error("useSettings must be used within SettingsProvider")
  return ctx
}

/** Owns the defaults new tabs inherit. Seeded from backend config without overriding existing values. */
export function SettingsProvider({ children }: { children: ReactNode }) {
  const [tabDefaults, setTabDefaults] = useState(DEFAULT_TAB_CONFIG)
  const [ocrModel, setOCRModel] = useState("")

  useEffect(() => {
    fetchBackendConfig()
      .then((cfg) => {
        setTabDefaults((prev) => ({
          ...prev,
          selectedModel: prev.selectedModel || cfg.default_model,
          // Backend value applies only while the flag is still the `true` default.
          downscaleImages: prev.downscaleImages ? cfg.downscale_images : prev.downscaleImages,
        }))
        setOCRModel((prev) => prev || cfg.vision_model)
      })
      .catch(() => {
        // backend unreachable, keep hardcoded fallbacks
      })
  }, [])

  return (
    <SettingsContext.Provider value={{ tabDefaults, ocrModel }}>
      {children}
    </SettingsContext.Provider>
  )
}
