"use client"

import { ArrowUp, LayoutGrid, Loader2, Plus, Square } from "lucide-react"
import { cn } from "@/lib/utils"
import { MenuPopover } from "@/components/ui/MenuPopover"
import { useEnabledTools } from "@/features/chat/EnabledToolsContext"
import { TOOLS } from "@/features/chat/tools"
import { ToolMenuItem, ToolMenuSection } from "./ToolMenuItem"
import type { AttachMenuItem } from "./ChatInput"

export const ICON_BUTTON = "rounded-full p-1.5 text-ink-faint transition-colors hover:bg-surface-elevated hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted disabled:opacity-30"
/** Send and stop share one circle: raised surface when actionable, outlined when not. */
const SEND_BUTTON = "ml-1.5 flex h-8 w-8 items-center justify-center rounded-full border transition-all duration-200 focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-2 focus-visible:outline-ink-muted"
const SEND_READY = "border-divider bg-surface-elevated text-ink hover:border-divider-strong active:scale-[0.96]"
const SEND_IDLE = "border-divider/50 text-ink-faint"
const MENU = "rounded-xl border border-divider bg-paper p-2 shadow-[var(--shadow-xl)]"

/** The composer's "+" menu; picking an entry closes it first. */
export function AttachMenu({ items, disabled }: { items: (AttachMenuItem & { iconClassName?: string })[]; disabled?: boolean }) {
  return (
    <MenuPopover
      trigger={(open, toggle) => (
        <button aria-label="Add attachment" onClick={toggle} disabled={disabled} className={cn(ICON_BUTTON, open && "bg-surface-elevated text-ink")}>
          <Plus className="h-4 w-4" />
        </button>
      )}
    >
      {(open, close) => open && (
        <div className={`absolute bottom-full left-0 mb-2 w-48 ${MENU}`}>
          {items.map(({ label, icon: Icon, onClick, iconClassName = "text-ink-muted" }) => (
            <button
              key={label}
              onClick={() => { close(); onClick() }}
              className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-xs text-ink hover:bg-surface-elevated/50 transition-colors"
            >
              <Icon className={`h-3.5 w-3.5 ${iconClassName}`} />{label}
            </button>
          ))}
        </div>
      )}
    </MenuPopover>
  )
}

/** Toggles for gigachad's chat tools; it stays open while toggling. */
export function ToolsMenu({ disabled }: { disabled?: boolean }) {
  const { enabledTools, toggleTool } = useEnabledTools()
  return (
    <MenuPopover
      trigger={(open, toggle) => (
        <button aria-label="Toggle tools" onClick={toggle} disabled={disabled} className={cn(ICON_BUTTON, open && "bg-surface-elevated text-ink")}>
          <LayoutGrid className="h-4 w-4" />
        </button>
      )}
    >
      {(open) => open && (
        <div className={`absolute bottom-full right-0 mb-2 w-56 ${MENU}`}>
          <ToolMenuSection title="Tools">
            {TOOLS.map((tool) => (
              <ToolMenuItem key={tool.name} icon={tool.icon} label={tool.selectorLabel} active={enabledTools.includes(tool.name)} onClick={() => toggleTool(tool.name)} />
            ))}
          </ToolMenuSection>
        </div>
      )}
    </MenuPopover>
  )
}

/** Stop while streaming, a spinner while an attachment is extracted, send otherwise. */
export function SendButton({ isStreaming, extracting, ready, disabled, onCancel, onSend }: {
  isStreaming?: boolean
  extracting?: boolean
  /** The draft has text or attachments. */
  ready: boolean
  disabled?: boolean
  onCancel?: () => void
  onSend: () => void
}) {
  if (isStreaming) {
    return (
      <button onClick={onCancel} className={cn(SEND_BUTTON, SEND_READY)} aria-label="Stop generating">
        <Square className="h-3 w-3 fill-current" />
      </button>
    )
  }
  if (extracting) {
    return (
      <button aria-label="Extracting attachment" disabled className={cn(SEND_BUTTON, SEND_IDLE)}>
        <Loader2 className="h-4 w-4 animate-spin" />
      </button>
    )
  }
  return (
    <button aria-label="Send message" onClick={onSend} disabled={disabled || !ready} className={cn(SEND_BUTTON, ready && !disabled ? SEND_READY : SEND_IDLE)}>
      <ArrowUp className="h-4 w-4" />
    </button>
  )
}
