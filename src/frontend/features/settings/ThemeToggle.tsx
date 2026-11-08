"use client"

import { Sun, Moon } from "lucide-react"
import { STORAGE_KEY_THEME } from "@/lib/backend/config"
import { useThemeName } from "@/hooks/useThemeName"
import { switchColors } from "@/lib/themeTransition"
import { cn } from "@/lib/utils"

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

  const iconClass = "absolute inset-0 h-4 w-4 transition-[opacity,transform] duration-200 ease-out motion-reduce:transition-none"
  // Both icons stay mounted so the switch cross-fades them with a slight turn.
  return (
    <button
      type="button"
      aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
      onClick={toggle}
      className="rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-divider-strong"
    >
      <span className="relative block h-4 w-4">
        <Sun aria-hidden className={cn(iconClass, theme === "dark" ? "rotate-0 opacity-100" : "-rotate-90 scale-75 opacity-0")} />
        <Moon aria-hidden className={cn(iconClass, theme === "dark" ? "rotate-90 scale-75 opacity-0" : "rotate-0 opacity-100")} />
      </span>
    </button>
  )
}
