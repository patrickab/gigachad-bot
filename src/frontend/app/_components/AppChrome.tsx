"use client"

import { useCallback, useState, type ReactNode } from "react"
import { ErrorBoundary } from "@/components/ui/ErrorBoundary"
import type { WorkspaceItem } from "@/contexts/TabContext"
import type { ChatReveal, ChatRevealHit } from "@/features/chat/ChatContainer"
import { ChatPicker } from "@/features/chat/ChatPicker"
import { TabPicker } from "@/features/navigation/TabPicker"
import { PiSessionPicker } from "@/features/pi/PiSessionPicker"
import { PromptEditor } from "@/features/settings/PromptEditor"
import { AppSidebar } from "./AppSidebar"

interface AppChromeProps {
  onOpenChat: (historyFile: string, qaIndex?: number, hit?: ChatRevealHit) => void
  onOpenProject: (slug: string) => void
  onProjectRenamed: (oldSlug: string, newSlug: string) => void
  onNewChat: (slug: string | null) => void
  onOpenItem: (item: WorkspaceItem, inNewTab?: boolean) => void
  /** The last chat reveal: its question is marked in the sidebar's branch rows. */
  reveal: ChatReveal
  /** The tab stack. */
  children: ReactNode
}

/** What the app shows once around its tabs: the sidebar, the pickers and the prompt editor, which stays open over every tab. */
export function AppChrome({ onOpenChat, onOpenProject, onProjectRenamed, onNewChat, onOpenItem, reveal, children }: AppChromeProps) {
  const [promptEditorOpen, setPromptEditorOpen] = useState(false)
  const editPrompts = useCallback(() => setPromptEditorOpen(true), [])
  return (
    // Clips the page: a surface running under the sidebar reaches left of the tab stack
    <div className="relative flex h-screen overflow-hidden">
      <ErrorBoundary resetLabel="Reload sidebar">
        <AppSidebar onOpenProject={onOpenProject} onProjectRenamed={onProjectRenamed} onNewChat={onNewChat} onOpenItem={onOpenItem} onEditPrompts={editPrompts} reveal={reveal} />
      </ErrorBoundary>
      {children}
      <ErrorBoundary resetLabel="Reload">
        <TabPicker />
        <ChatPicker onOpenChat={onOpenChat} />
        <PiSessionPicker onOpenItem={onOpenItem} />
        <PromptEditor open={promptEditorOpen} onClose={() => setPromptEditorOpen(false)} />
      </ErrorBoundary>
    </div>
  )
}
