"use client"

import { type ReactNode } from "react"
import { cn } from "@/lib/utils"

type Accent = "muted" | "danger" | "neutral"

const ACCENT_MAP: Record<Accent, { bg: string; border: string; text: string; hoverBg: string; hoverBorder: string }> = {
  muted: {
    bg: "bg-surface-elevated",
    border: "border-divider-strong",
    text: "text-ink",
    hoverBg: "hover:bg-hover-strong",
    hoverBorder: "hover:border-ink-muted",
  },
  danger: {
    bg: "bg-danger-soft",
    border: "border-danger-soft",
    text: "text-danger",
    hoverBg: "hover:bg-danger-soft",
    hoverBorder: "hover:border-danger",
  },
  neutral: {
    bg: "bg-surface",
    border: "",
    text: "text-ink-subtle",
    hoverBg: "hover:bg-hover",
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

export function PillButton({ accent = "neutral", active, icon, children, onClick, className, disabled }: PillButtonProps) {
  const a = ACCENT_MAP[accent]
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={cn(
        "flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium transition-colors",
        active
          ? cn(a.bg, a.border, a.text, "border", a.hoverBg, a.hoverBorder)
          : "bg-surface text-ink-subtle hover:text-ink",
        disabled && "opacity-50 cursor-not-allowed",
        className
      )}
    >
      {icon}
      {children}
    </button>
  )
}
