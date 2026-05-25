"use client"

import { useEffect, useState } from "react"
import { Sun, Moon } from "lucide-react"

export function ThemeToggle() {
  const [theme, setTheme] = useState<"light" | "dark">("dark")

  useEffect(() => {
    const stored = localStorage.getItem("theme")
    if (stored === "light" || stored === "dark") setTheme(stored)
  }, [])

  function toggle() {
    const next = theme === "dark" ? "light" : "dark"
    setTheme(next)
    localStorage.setItem("theme", next)
    document.documentElement.className = next
  }

  return (
    <button
      onClick={toggle}
      className="rounded-lg p-2 text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900 transition-colors"
      title={theme === "dark" ? "Light mode" : "Dark mode"}
    >
      {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </button>
  )
}
