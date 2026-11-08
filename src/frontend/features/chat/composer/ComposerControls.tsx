"use client"

import { LayoutGrid, Loader2, Plus, Square, ArrowUp } from "lucide-react"
import { cn } from "@/lib/utils"
import { COMPOSER_ICON_BUTTON, composerSendButton } from "@/components/ui/Composer"
import { MenuPopover } from "@/components/ui/MenuPopover"
import { useHostAccess } from "@/contexts/SettingsContext"
import { useEnabledTools } from "@/features/chat/EnabledToolsContext"
import { availableTools } from "@/features/chat/tools"
import { ToolMenuItem, ToolMenuSection } from "./ToolMenuItem"
import type { AttachMenuItem } from "./ChatInput"

const MENU = "rounded-xl border border-divider bg-surface-elevated p-1 shadow-[var(--shadow-lg)]"

/** The composer's "+" menu; picking an entry closes it first. */
export function AttachMenu({ items, disabled }: { items: (AttachMenuItem & { iconClassName?: string })[]; disabled?: boolean }) {
  return (
    <MenuPopover
      trigger={(open, toggle) => (
        <button aria-label="Add attachment" onClick={toggle} disabled={disabled} className={cn(COMPOSER_ICON_BUTTON, open && "bg-surface-elevated text-ink")}>
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
              className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-1.5 text-left text-xs text-ink transition-colors hover:bg-hover focus-visible:bg-hover focus-visible:outline focus-visible:outline-1 focus-visible:-outline-offset-1 focus-visible:outline-ink-muted"
            >
              <Icon className={`h-3.5 w-3.5 shrink-0 ${iconClassName}`} />{label}
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
  const hostAccess = useHostAccess()
  return (
    <MenuPopover
      trigger={(open, toggle) => (
        <button aria-label="Toggle tools" onClick={toggle} disabled={disabled} className={cn(COMPOSER_ICON_BUTTON, open && "bg-surface-elevated text-ink")}>
          <LayoutGrid className="h-4 w-4" />
        </button>
      )}
    >
      {(open) => open && (
        <div className={`absolute bottom-full right-0 mb-2 w-56 ${MENU}`}>
          <ToolMenuSection title="Tools">
            {availableTools(hostAccess).map((tool) => (
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
      <button onClick={onCancel} className={composerSendButton(true)} aria-label="Stop generating">
        <Square className="h-3 w-3 fill-current" />
      </button>
    )
  }
  if (extracting) {
    return (
      <button aria-label="Extracting attachment" disabled className={composerSendButton(false)}>
        <Loader2 className="h-4 w-4 animate-spin" />
      </button>
    )
  }
  return (
    <button aria-label="Send message" onClick={onSend} disabled={disabled || !ready} className={composerSendButton(ready && !disabled)}>
      <ArrowUp className="h-4 w-4" />
    </button>
  )
}
