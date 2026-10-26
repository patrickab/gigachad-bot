"use client"

import { Sun, Moon } from "lucide-react"
import { STORAGE_KEY_THEME } from "@/lib/backend/config"
import { useThemeName } from "@/hooks/useThemeName"
import { switchColors } from "@/lib/themeTransition"

export function ThemeToggle() {
  const theme = useThemeName()

  function toggle() {
    switchColors(() => {
      const root = document.documentElement
      const light = !root.classList.contains("light")
      root.classList.toggle("light", light)
      try { localStorage.setItem(STORAGE_KEY_THEME, light ? "light" : "dark") } catch {}
    })
  }

  return (
    <button
      type="button"
      aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
      onClick={toggle}
      className="rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-divider-strong"
    >
      {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </button>
  )
}
