"use client"

import { useSyncExternalStore } from "react"
import { activeThemeName, type ThemeName } from "@/lib/palette"

function subscribe(onChange: () => void): () => void {
  const observer = new MutationObserver(onChange)
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] })
  return () => observer.disconnect()
}

/** Active theme from the `<html>` class, re-rendering only when it flips. */
export function useThemeName(): ThemeName {
  return useSyncExternalStore(subscribe, activeThemeName, () => "dark")
}
