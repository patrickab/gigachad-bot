"use client"

import { useSyncExternalStore } from "react"
import { STORAGE_KEY_GLASS } from "@/lib/backend/config"
import { subscribeRootClass } from "@/hooks/useThemeName"

const isGlass = () => document.documentElement.classList.contains("glass")

/** Turn glass mode on or off: the `glass` class on `<html>` (globals.css, "GLASS"), remembered for the next visit. */
export function setGlassMode(on: boolean): void {
  document.documentElement.classList.toggle("glass", on)
  try { localStorage.setItem(STORAGE_KEY_GLASS, on ? "on" : "off") } catch {}
}

/** Whether glass mode is on, read from the `<html>` class, which the layout's startup script sets before hydration. */
export function useGlassMode(): boolean {
  return useSyncExternalStore(subscribeRootClass, isGlass, () => false)
}
