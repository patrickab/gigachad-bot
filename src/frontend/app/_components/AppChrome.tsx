"use client"

import { useCallback, useState, type ReactNode } from "react"
import { ErrorBoundary } from "@/components/ui/ErrorBoundary"
import { SaveAsModal } from "@/components/ui/SaveAsModal"
import { useTab, type WorkspaceItem } from "@/contexts/TabContext"
import type { ChatReveal, ChatRevealHit } from "@/features/chat/ChatContainer"
import { ChatPicker } from "@/features/chat/ChatPicker"
import { TabPicker } from "@/features/navigation/TabPicker"
import { PiSessionPicker } from "@/features/pi/PiSessionPicker"
import { PromptEditor } from "@/features/settings/PromptEditor"
import { AppSidebar } from "./AppSidebar"

/** A tab's request to name and save its chat, shown while that tab is in front. */
export interface SaveAsRequest { tabId: string; save: (name: string) => Promise<void> }

interface AppChromeProps {
  onOpenChat: (historyFile: string, qaIndex?: number, hit?: ChatRevealHit) => void
  onOpenProject: (slug: string) => void
  onProjectRenamed: (oldSlug: string, newSlug: string) => void
  onNewChat: (slug: string | null) => void
  onOpenItem: (item: WorkspaceItem, inNewTab?: boolean) => void
  /** The last chat reveal: its question is marked in the sidebar's branch rows. */
  reveal: ChatReveal
  saveAs: SaveAsRequest | null
  onSaveAsDone: () => void
  /** The tab stack. */
  children: ReactNode
}

/**
 * What the app shows once around its tabs: the sidebar, the pickers and the two dialogs, all acting on the tab in front.
 * A dialog belongs to the tab that opened it: hidden while another tab is in front, back when it returns.
 */
export function AppChrome({ onOpenChat, onOpenProject, onProjectRenamed, onNewChat, onOpenItem, reveal, saveAs, onSaveAsDone, children }: AppChromeProps) {
  const { tab } = useTab()
  const [promptEditorTab, setPromptEditorTab] = useState<string | null>(null)
  // Changes on a tab switch only, when the sidebar renders anyway
  const editPrompts = useCallback(() => setPromptEditorTab(tab.id), [tab.id])
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
        <PromptEditor open={promptEditorTab === tab.id} onClose={() => setPromptEditorTab(null)} />
        <SaveAsModal open={saveAs?.tabId === tab.id} onClose={onSaveAsDone} onSave={(name) => void saveAs?.save(name).then(onSaveAsDone)} />
      </ErrorBoundary>
    </div>
  )
}
