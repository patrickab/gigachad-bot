"use client"

import { createContext, useContext, useEffect, useState, type ReactNode } from "react"
import { fetchBackendConfig } from "@/lib/api"

export interface SettingsState {
  selectedModel: string
  setSelectedModel: (v: string) => void
  selectedPrompt: string | null
  setSelectedPrompt: (v: string | null) => void
  temperature: number
  setTemperature: (v: number) => void
  reasoningEffort: string
  setReasoningEffort: (v: string) => void
  downscaleImages: boolean
  setDownscaleImages: (v: boolean) => void
  researchFastModel: string
  setResearchFastModel: (v: string) => void
  researchSmartModel: string
  setResearchSmartModel: (v: string) => void
  researchStrategicModel: string
  setResearchStrategicModel: (v: string) => void
  researchDepth: number
  setResearchDepth: (v: number) => void
  researchBreadth: number
  setResearchBreadth: (v: number) => void
  researchReasoning: string
  setResearchReasoning: (v: string) => void
  researchReportType: string
  setResearchReportType: (v: string) => void
  ocrModel: string
  setOCRModel: (v: string) => void
}

const SettingsContext = createContext<SettingsState | null>(null)

export function useSettings(): SettingsState {
  const ctx = useContext(SettingsContext)
  if (!ctx) throw new Error("useSettings must be used within SettingsProvider")
  return ctx
}

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [selectedModel, setSelectedModel] = useState("")
  const [selectedPrompt, setSelectedPrompt] = useState<string | null>(null)
  const [temperature, setTemperature] = useState(0.2)
  const [reasoningEffort, setReasoningEffort] = useState("none")
  const [downscaleImages, setDownscaleImages] = useState(true)

  const [researchFastModel, setResearchFastModel] = useState("")
  const [researchSmartModel, setResearchSmartModel] = useState("")
  const [researchStrategicModel, setResearchStrategicModel] = useState("")
  const [researchDepth, setResearchDepth] = useState(2)
  const [researchBreadth, setResearchBreadth] = useState(4)
  const [researchReasoning, setResearchReasoning] = useState("medium")
  const [researchReportType, setResearchReportType] = useState("deep")

  const [ocrModel, setOCRModel] = useState("")

  useEffect(() => {
    fetchBackendConfig()
      .then((cfg) => {
        setSelectedModel((prev) => prev || cfg.default_model)
        setTemperature((prev) => prev === 0.2 ? cfg.temperature : prev)
        setDownscaleImages((prev) => prev === true ? cfg.downscale_images : prev)
        setOCRModel((prev) => prev || cfg.vision_model)
      })
      .catch(() => {
        // backend unreachable — keep hardcoded fallbacks
      })
  }, [])

  const value = {
    selectedModel, setSelectedModel,
    selectedPrompt, setSelectedPrompt,
    temperature, setTemperature,
    reasoningEffort, setReasoningEffort,
    downscaleImages, setDownscaleImages,
    researchFastModel, setResearchFastModel,
    researchSmartModel, setResearchSmartModel,
    researchStrategicModel, setResearchStrategicModel,
    researchDepth, setResearchDepth,
    researchBreadth, setResearchBreadth,
    researchReasoning, setResearchReasoning,
    researchReportType, setResearchReportType,
    ocrModel, setOCRModel,
  }

  return (
    <SettingsContext.Provider value={value}>
      {children}
    </SettingsContext.Provider>
  )
}