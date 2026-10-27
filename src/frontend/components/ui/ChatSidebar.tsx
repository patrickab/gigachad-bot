"use client"

import { useState, type ReactNode } from "react"
import { AssistantChat, type AssistantChatState } from "@/components/content/AssistantChat"
import { PanelToggleButton } from "@/components/ui/PanelToggleButton"
import { SidePanelFrame } from "@/components/ui/SidePanel"

/** Opening width, narrow so the main content keeps the space. */
const OPEN_WIDTH_PX = 320
/** Gap between a floating top bar and the open panel. */
const TOP_GAP_PX = 8

/** Mirror of the left sidebar toggle: on the top bar's grid while closed, in the panel's top-left corner while open. */
function ChatSidebarToggle({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  return <PanelToggleButton open={open} onClick={() => onOpenChange(!open)} openLabel="Open chat" closeLabel="Close chat" />
}

/**
 * Main content with a resizable assistant chat panel on its right, and the panel's own toggle.
 *
 * - `children` take the remaining width, the panel never overlaps them
 * - Closed, the toggle has a row under a top bar `topInset` high, on the bar's grid (`h-11 px-3`), under the bar's right-most control
 * - Open, the panel starts under the top bar and holds the toggle in its header, the top bar stays untouched
 * - `status` shows above the messages, e.g. while the chat's context loads
 */
export function ChatSidebar({ open, onOpenChange, chat, inputLabel, disabled = false, status, topInset = 0, children }: {
  open: boolean
  onOpenChange: (open: boolean) => void
  chat: AssistantChatState
  inputLabel: string
  disabled?: boolean
  status?: ReactNode
  topInset?: number
  children: ReactNode
}) {
  const [width, setWidth] = useState(OPEN_WIDTH_PX)

  return (
    <div className="relative flex h-full">
      <div className="relative min-w-0 flex-1">{children}</div>
      {open ? (
        <div className="flex shrink-0 flex-col" style={{ paddingTop: topInset ? topInset + TOP_GAP_PX : 0 }}>
          <SidePanelFrame
            width={width}
            onWidthChange={setWidth}
            className="mb-3 mr-3 min-h-0 flex-1 rounded-2xl border border-divider/50 bg-surface shadow-[var(--shadow-lg)]"
          >
            <section aria-label={inputLabel} className="flex min-h-0 flex-1 flex-col">
              {/* Insets match the left sidebar's header icon */}
              <div className="flex h-10 shrink-0 items-center px-4">
                <ChatSidebarToggle open onOpenChange={onOpenChange} />
              </div>
              {status}
              <AssistantChat chat={chat} inputLabel={inputLabel} disabled={disabled} surfaceClassName="bg-surface-elevated" />
            </section>
          </SidePanelFrame>
        </div>
      ) : (
        // Clicks between the row's edges reach the content beneath, like the top bar
        <div className="pointer-events-none absolute right-0 z-10 flex h-11 items-center px-3" style={{ top: topInset }}>
          <div className="pointer-events-auto flex">
            <ChatSidebarToggle open={false} onOpenChange={onOpenChange} />
          </div>
        </div>
      )}
    </div>
  )
}
