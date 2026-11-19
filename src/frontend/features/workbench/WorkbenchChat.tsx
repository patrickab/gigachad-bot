"use client"

import { useRef, useState } from "react"
import type { PiInsertion } from "@/features/pi/PiComposer"
import type { PiShellCommands } from "@/features/pi/PiTab"
import { DocumentChat } from "./DocumentChat"
import { useDocumentChat, type DocumentChat as DocumentChatState } from "./useDocumentChat"

/** What a workbench's owner hands its chat: the session the tab's item shows, the drawer and the composer's commands. */
export interface WorkbenchChatProps {
  /** The Pi session this tab's item shows, undefined before the first one. */
  chatSessionId: string | undefined
  onChatSessionChange: (sessionId: string) => void
  drawerOpen: boolean
  onDrawerOpenChange: (open: boolean) => void
  shellCommands: PiShellCommands
}

/** One workbench's chat and the text waiting to go into its composer draft. */
export interface WorkbenchChatState {
  chat: DocumentChatState
  insertion: PiInsertion | null
  /** Put `text` into the composer draft, opening the drawer if it is closed. */
  quote: (text: string) => void
  onInserted: () => void
}

/** Width of the chat panel (the tab's file panel card, like the PDF chat), the workbench's content shares what is left. */
const DRAWER_WIDTH_PX = 360

/** The Pi chat linked to `key` (`useDocumentChat`), naming the file to a new session with `lead(cwd)`, with quotes into its draft. */
export function useWorkbenchChat(key: string, lead: (cwd: string) => string, { chatSessionId, onChatSessionChange, onDrawerOpenChange }: WorkbenchChatProps): WorkbenchChatState {
  const chat = useDocumentChat(key, lead, { sessionId: chatSessionId, onSessionChange: onChatSessionChange })
  const [insertion, setInsertion] = useState<PiInsertion | null>(null)
  const insertionCount = useRef(0)
  const quote = (text: string) => {
    onDrawerOpenChange(true)
    setInsertion({ id: ++insertionCount.current, text })
  }
  return { chat, insertion, quote, onInserted: () => setInsertion(null) }
}

/** The chat drawer of a workbench in the tab's file panel. Its header holds the session's model and thinking controls. */
export function WorkbenchChat({ state, drawerOpen, onDrawerOpenChange, shellCommands }: Pick<WorkbenchChatProps, "drawerOpen" | "onDrawerOpenChange" | "shellCommands"> & { state: WorkbenchChatState }) {
  if (!drawerOpen) return null
  return (
    <DocumentChat
      chat={state.chat}
      width={DRAWER_WIDTH_PX}
      onClose={() => onDrawerOpenChange(false)}
      shellCommands={shellCommands}
      insertion={state.insertion}
      onInserted={state.onInserted}
    />
  )
}
