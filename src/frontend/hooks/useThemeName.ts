"use client"

import { useSyncExternalStore } from "react"
import { activeThemeName, type ThemeName } from "@/lib/palette"

/** Re-run `onChange` whenever the `<html>` class list changes. */
export function subscribeRootClass(onChange: () => void): () => void {
  const observer = new MutationObserver(onChange)
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] })
  return () => observer.disconnect()
}

/** Active theme from the `<html>` class, re-rendering only when it flips. */
export function useThemeName(): ThemeName {
  return useSyncExternalStore(subscribeRootClass, activeThemeName, () => "dark")
}
