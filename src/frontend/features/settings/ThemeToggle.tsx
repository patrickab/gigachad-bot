"use client"

import { Sun, Moon } from "lucide-react"
import { STORAGE_KEY_THEME } from "@/lib/backend/config"
import { useThemeName } from "@/hooks/useThemeName"

export function ThemeToggle() {
  const theme = useThemeName()

  function toggle() {
    const next = theme === "dark" ? "light" : "dark"
    try { localStorage.setItem(STORAGE_KEY_THEME, next) } catch {}
    const root = document.documentElement
    // `theme-transition` animates the swap only briefly so later color changes stay instant.
    root.classList.add("theme-transition")
    root.classList.toggle("light", next === "light")
    window.setTimeout(() => root.classList.remove("theme-transition"), 3000)
  }

  return (
    <button
      aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
      onClick={toggle}
      className="rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover transition-colors"
    >
      {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </button>
  )
}
