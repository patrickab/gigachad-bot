"use client"

import { createContext, useContext, useEffect, useState, type ReactNode } from "react"
import { fetchBackendConfig } from "@/lib/backend/api"
import { STORAGE_KEY_NEOVIM } from "@/lib/backend/config"

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

export interface NeovimSetting {
  /** The backend runs Neovim for this user: allow-listed, `nvim` installed, config present. */
  available: boolean
  /** Edit text documents in Neovim: chosen on this device and available. */
  enabled: boolean
  setPreferred: (preferred: boolean) => void
}

const NEOVIM_UNAVAILABLE: NeovimSetting = { available: false, enabled: false, setPreferred: () => {} }

interface SettingsValue {
  /** Config every new tab starts from. */
  tabDefaults: TabConfig
  ocrModel: string
  neovim: NeovimSetting
  /** omp-web's address for the omp tabs' frame, null until the backend config loaded. */
  ompWebUrl: string | null
  /** This login may use host-only features such as the factory. */
  hostAccess: boolean
}

const SettingsContext = createContext<SettingsValue | null>(null)

export function useSettings(): SettingsValue {
  const ctx = useContext(SettingsContext)
  if (!ctx) throw new Error("useSettings must be used within SettingsProvider")
  return ctx
}

/** Neovim editor setting, unavailable outside a `SettingsProvider`. */
export function useNeovimSetting(): NeovimSetting {
  return useContext(SettingsContext)?.neovim ?? NEOVIM_UNAVAILABLE
}

/** Whether this login has host access, false outside a `SettingsProvider`. */
export function useHostAccess(): boolean {
  return useContext(SettingsContext)?.hostAccess ?? false
}

function readNeovimPreference(): boolean {
  try { return localStorage.getItem(STORAGE_KEY_NEOVIM) === "true" } catch { return false }
}

/** Owns the defaults new tabs inherit. Seeded from backend config without overriding existing values. */
export function SettingsProvider({ children }: { children: ReactNode }) {
  const [tabDefaults, setTabDefaults] = useState(DEFAULT_TAB_CONFIG)
  const [ocrModel, setOCRModel] = useState("")
  const [neovimAvailable, setNeovimAvailable] = useState(false)
  const [neovimPreferred, setNeovimPreferred] = useState(readNeovimPreference)
  const [ompWebUrl, setOmpWebUrl] = useState<string | null>(null)
  const [hostAccess, setHostAccess] = useState(false)

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
        setNeovimAvailable(cfg.neovim_available)
        setOmpWebUrl(cfg.omp_web_url)
        setHostAccess(cfg.host_access)
      })
      .catch(() => {
        // backend unreachable, keep hardcoded fallbacks
      })
  }, [])

  const neovim: NeovimSetting = {
    available: neovimAvailable,
    enabled: neovimAvailable && neovimPreferred,
    setPreferred: (preferred) => {
      setNeovimPreferred(preferred)
      try { localStorage.setItem(STORAGE_KEY_NEOVIM, String(preferred)) } catch { /* quota */ }
    },
  }

  return (
    <SettingsContext.Provider value={{ tabDefaults, ocrModel, neovim, ompWebUrl, hostAccess }}>
      {children}
    </SettingsContext.Provider>
  )
}
