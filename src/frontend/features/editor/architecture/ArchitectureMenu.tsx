"use client"

import type { CSSProperties, ReactNode } from "react"
import { MenuPopover } from "@/components/ui/MenuPopover"

const MENU_STYLE: CSSProperties = {
  position: "absolute", zIndex: 10, top: "calc(100% + 4px)", padding: 4,
  border: "1px solid var(--divider)", borderRadius: 6, background: "var(--paper)", boxShadow: "var(--shadow-lg)",
}

/** A trigger button and the popover menu under it, closed by an outside press or Escape. */
export function ArchitectureMenu({ label, trigger, align, menuStyle, className, children }: {
  label: string
  trigger: (open: boolean, toggle: () => void) => ReactNode
  /** Which edge of the trigger the menu lines up with. */
  align: "left" | "right"
  menuStyle?: CSSProperties
  className?: string
  children: (close: () => void) => ReactNode
}) {
  return (
    <MenuPopover className={className} trigger={trigger}>
      {(open, close) => open && <div role="menu" aria-label={label} style={{ ...MENU_STYLE, [align]: 0, ...menuStyle }}>{children(close)}</div>}
    </MenuPopover>
  )
}
