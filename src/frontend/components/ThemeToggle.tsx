"use client"

import { useEffect, useState } from "react"
import { Sun, Moon } from "lucide-react"
import { STORAGE_KEY_THEME } from "@/lib/config"

export function ThemeToggle() {
  const [theme, setTheme] = useState<"light" | "dark">("dark")

  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY_THEME)
      if (stored === "light" || stored === "dark") setTheme(stored)
    } catch {}
  }, [])

  function toggle() {
    const next = theme === "dark" ? "light" : "dark"
    setTheme(next)
    try { localStorage.setItem(STORAGE_KEY_THEME, next) } catch {}
    document.documentElement.classList.toggle("light", next === "light")
  }

  return (
    <button
      onClick={toggle}
      className="rounded-lg p-2 text-ink-subtle hover:text-ink hover:bg-hover transition-colors"
    >
      {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </button>
  )
}
