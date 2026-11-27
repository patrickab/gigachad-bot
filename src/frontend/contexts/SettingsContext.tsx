"use client"

import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react"
import { fetchPiSettings, savePiSettings, saveAnimatedBackground } from "@/lib/backend/api"
import { STORAGE_KEY_NEOVIM } from "@/lib/backend/config"
import { backendConfigResource } from "@/lib/backend/resources"
import { useRequiredContext } from "@/hooks/useRequiredContext"
import { useResource } from "@/hooks/useResource"
import { readStorage, writeStorage } from "@/lib/storage"

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
  /** LearnHarness's address for the learn frame, null until the backend config loaded and for logins without host access. */
  learnharnessUrl: string | null
  /** This login may use host-only features such as the factory. */
  hostAccess: boolean
  /** Whether the animated background shows: stored per user, null until the backend config loaded. */
  animatedBackground: boolean | null
  setAnimatedBackground: (enabled: boolean) => void
  /** Whether pi-host resumes runs a restart interrupted: null until loaded, and without host access or pi-host. */
  piAutoResume: boolean | null
  setPiAutoResume: (enabled: boolean) => void
}

const SettingsContext = createContext<SettingsValue | null>(null)

export function useSettings(): SettingsValue {
  return useRequiredContext(SettingsContext, "useSettings must be used within SettingsProvider")
}

/** Neovim editor setting, unavailable outside a `SettingsProvider`. */
export function useNeovimSetting(): NeovimSetting {
  return useContext(SettingsContext)?.neovim ?? NEOVIM_UNAVAILABLE
}

/** Whether this login has host access, false outside a `SettingsProvider`. */
export function useHostAccess(): boolean {
  return useContext(SettingsContext)?.hostAccess ?? false
}

/** The user's animated background setting, null outside a `SettingsProvider` and until it loaded. */
export function useAnimatedBackground(): boolean | null {
  return useContext(SettingsContext)?.animatedBackground ?? null
}

/** pi-host's auto-resume setting and its setter, null outside a `SettingsProvider` and until it loaded. */
export function usePiAutoResume(): [boolean | null, (enabled: boolean) => void] {
  const settings = useContext(SettingsContext)
  return [settings?.piAutoResume ?? null, settings?.setPiAutoResume ?? (() => {})]
}

/** Stores the animated background setting, a no-op outside a `SettingsProvider`. */
export function useSetAnimatedBackground(): (enabled: boolean) => void {
  return useContext(SettingsContext)?.setAnimatedBackground ?? (() => {})
}

function readNeovimPreference(): boolean {
  return readStorage(STORAGE_KEY_NEOVIM) === "true"
}

/** Owns the defaults new tabs inherit, derived from the shared backend config. */
export function SettingsProvider({ children }: { children: ReactNode }) {
  const config = useResource("Settings", backendConfigResource)
  const defaultModel = config?.default_model ?? ""
  const tabDefaults = useMemo(() => ({ ...DEFAULT_TAB_CONFIG, selectedModel: defaultModel }), [defaultModel])
  const ocrModel = config?.vision_model ?? ""
  const neovimAvailable = config?.neovim_available ?? false
  const learnharnessUrl = config?.learnharness_url ?? null
  const hostAccess = config?.host_access ?? false
  const [neovimPreferred, setNeovimPreferred] = useState(readNeovimPreference)
  // The optimistic toggle, over the stored value until set.
  const [animatedBackgroundChoice, setAnimatedBackgroundState] = useState<boolean | null>(null)
  const animatedBackground = animatedBackgroundChoice ?? config?.animated_background ?? null
  const [piAutoResume, setPiAutoResumeState] = useState<boolean | null>(null)

  useEffect(() => {
    if (!hostAccess) return
    const controller = new AbortController()
    // Without pi-host the setting stays hidden
    fetchPiSettings(controller.signal)
      .then((settings) => setPiAutoResumeState(settings.autoResume))
      .catch(() => {})
    return () => controller.abort()
  }, [hostAccess])

  const neovim: NeovimSetting = {
    available: neovimAvailable,
    enabled: neovimAvailable && neovimPreferred,
    setPreferred: (preferred) => {
      setNeovimPreferred(preferred)
      writeStorage(STORAGE_KEY_NEOVIM, String(preferred))
    },
  }

  const setAnimatedBackground = (enabled: boolean) => {
    setAnimatedBackgroundState(enabled)
    saveAnimatedBackground(enabled).catch(() => setAnimatedBackgroundState(!enabled))
  }

  const setPiAutoResume = (enabled: boolean) => {
    setPiAutoResumeState(enabled)
    savePiSettings({ autoResume: enabled }).catch(() => setPiAutoResumeState(!enabled))
  }

  return (
    <SettingsContext.Provider value={{ tabDefaults, ocrModel, neovim, learnharnessUrl, hostAccess, animatedBackground, setAnimatedBackground, piAutoResume, setPiAutoResume }}>
      {children}
    </SettingsContext.Provider>
  )
}
