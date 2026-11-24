"use client"

import { useSyncExternalStore } from "react"
import { STORAGE_KEY_KITTY } from "@/lib/backend/config"

function subscribe(onChange: () => void): () => void {
  const observer = new MutationObserver(onChange)
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] })
  return () => observer.disconnect()
}

const isKitty = () => document.documentElement.classList.contains("kitty")

/** Turn the pink kitty theme on or off: the `kitty` class on `<html>` (globals.css, "KITTY"), remembered for the next visit. */
export function setKittyMode(on: boolean): void {
  document.documentElement.classList.toggle("kitty", on)
  try { localStorage.setItem(STORAGE_KEY_KITTY, on ? "on" : "off") } catch { /* storage unavailable: the theme still applies for this visit */ }
}

/** Whether the kitty theme is on, read from the `<html>` class, which the layout's startup script sets before hydration. */
export function useKittyMode(): boolean {
  return useSyncExternalStore(subscribe, isKitty, () => false)
}
