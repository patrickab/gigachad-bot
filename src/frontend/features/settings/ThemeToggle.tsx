"use client"

import { Sun, Moon } from "lucide-react"
import { STORAGE_KEY_THEME } from "@/lib/backend/config"
import { useThemeName } from "@/hooks/useThemeName"

// Theme switches share one document, even when several workspace tabs are mounted.
let activeTransition: ViewTransition | undefined

export function ThemeToggle() {
  const theme = useThemeName()

  function toggle() {
    const root = document.documentElement
    const applyTheme = () => {
      const light = !root.classList.contains("light")
      root.classList.toggle("light", light)
      try { localStorage.setItem(STORAGE_KEY_THEME, light ? "light" : "dark") } catch {}
    }

    activeTransition?.skipTransition()
    if (!document.startViewTransition || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      applyTheme()
      return
    }

    const radius = Math.hypot(window.innerWidth / 2, window.innerHeight / 2)
    root.style.setProperty("--theme-radius", `${radius}px`)
    root.classList.add("theme-transition")
    const transition = document.startViewTransition(applyTheme)
    activeTransition = transition
    // Rapid switches skip the old animation, but still apply each requested theme.
    void transition.ready.catch(() => {})
    void transition.finished.finally(() => {
      if (activeTransition !== transition) return
      activeTransition = undefined
      root.classList.remove("theme-transition")
      root.style.removeProperty("--theme-radius")
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
