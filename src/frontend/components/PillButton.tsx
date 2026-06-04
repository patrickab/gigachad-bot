"use client"

import { type ReactNode } from "react"
import { cn } from "@/lib/utils"

type Accent = "sky" | "amber" | "emerald" | "zinc"

const ACCENT_MAP: Record<Accent, { bg: string; border: string; text: string; hoverBg: string; hoverBorder: string }> = {
  sky: {
    bg: "bg-sky-500/10",
    border: "border-sky-500/30",
    text: "text-sky-400",
    hoverBg: "hover:bg-sky-500/20",
    hoverBorder: "hover:border-sky-500/50",
  },
  amber: {
    bg: "bg-amber-500/10",
    border: "border-amber-500/30",
    text: "text-amber-400",
    hoverBg: "hover:bg-amber-500/20",
    hoverBorder: "hover:border-amber-500/50",
  },
  emerald: {
    bg: "bg-emerald-500/10",
    border: "border-emerald-500/30",
    text: "text-emerald-400",
    hoverBg: "hover:bg-emerald-500/20",
    hoverBorder: "hover:border-emerald-500/50",
  },
  zinc: {
    bg: "bg-zinc-700",
    border: "",
    text: "text-zinc-100",
    hoverBg: "hover:bg-zinc-600",
    hoverBorder: "",
  },
}

interface PillButtonProps {
  accent?: Accent
  active?: boolean
  icon?: ReactNode
  children: ReactNode
  onClick?: () => void
  className?: string
  disabled?: boolean
}

export function PillButton({ accent = "zinc", active, icon, children, onClick, className, disabled }: PillButtonProps) {
  const a = ACCENT_MAP[accent]
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={cn(
        "flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium transition-colors",
        active
          ? cn(a.bg, a.border, a.text, "border", a.hoverBg, a.hoverBorder)
          : "bg-zinc-900 text-zinc-500 hover:text-zinc-300",
        disabled && "opacity-50 cursor-not-allowed",
        className
      )}
    >
      {icon}
      {children}
    </button>
  )
}