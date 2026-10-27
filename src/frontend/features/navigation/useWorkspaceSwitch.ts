"use client"

import { useCallback, useEffect, useState, useSyncExternalStore, type SetStateAction } from "react"

export const WORKSPACE_SWITCH_MS = 160

function subscribeReducedMotion(onChange: () => void) {
  const media = window.matchMedia?.("(prefers-reduced-motion: reduce)")
  media?.addEventListener("change", onChange)
  return () => media?.removeEventListener("change", onChange)
}

function prefersReducedMotion() {
  return window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? true
}

/** Which tab is frontmost and which one dissolves out behind it; reduced motion skips the dissolve. */
export function useWorkspaceSwitch(initialTab: string) {
  const reduceMotion = useSyncExternalStore(subscribeReducedMotion, prefersReducedMotion, () => true)
  const [{ activeTab, outgoingTab }, setVisibleTabs] = useState<{ activeTab: string; outgoingTab: string | null }>({ activeTab: initialTab, outgoingTab: null })

  const setActiveTab = useCallback((next: SetStateAction<string>) => {
    setVisibleTabs((current) => {
      const id = typeof next === "function" ? next(current.activeTab) : next
      if (id === current.activeTab) return current
      return { activeTab: id, outgoingTab: reduceMotion ? null : current.activeTab }
    })
  }, [reduceMotion])

  useEffect(() => {
    if (!outgoingTab || !reduceMotion) return
    const timer = window.setTimeout(() => {
      setVisibleTabs((current) => current.activeTab === activeTab && current.outgoingTab === outgoingTab
        ? { ...current, outgoingTab: null }
        : current)
    }, 0)
    return () => window.clearTimeout(timer)
  }, [activeTab, outgoingTab, reduceMotion])

  /** Ends the switch once the incoming tab `id` finished dissolving in. */
  const onAnimationEnd = useCallback((id: string, animationName: string) => {
    if (id !== activeTab || animationName !== "workspace-dissolve-in") return
    setVisibleTabs((current) => current.activeTab === id ? { ...current, outgoingTab: null } : current)
  }, [activeTab])

  const phaseOf = (id: string) => id === activeTab ? outgoingTab && !reduceMotion ? "incoming" : "idle" : "outgoing"
  const isShown = (id: string) => id === activeTab || id === outgoingTab && !reduceMotion

  return { activeTab, setActiveTab, phaseOf, isShown, onAnimationEnd }
}
