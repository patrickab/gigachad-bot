"use client"

import { useRef, useState, type ComponentProps } from "react"
import { TopBarPill, TopBarTransparent } from "@/features/navigation/TopBar"
import { InTopBar } from "@/features/navigation/TopBarSlots"
import { OmpModelSelector, OmpReasoningSelector, type ComposerInsertion, type ShellCommands } from "@/features/omp/OmpComposer"
import { OmpChatPanel } from "@/features/omp/OmpChatPanel"
import { useFileChat, type FileChat } from "@/features/omp/useFileChat"
import type { OmpView } from "@/lib/domain/types"

/** What a workbench's owner hands its chat: the session the tab's item shows, the drawer and the composer's commands and model picker. */
export interface WorkbenchChatProps {
  /** The chat session this tab's item shows, undefined before the first one. */
  ompView: OmpView | undefined
  onViewChange: (view: OmpView) => void
  drawerOpen: boolean
  onDrawerOpenChange: (open: boolean) => void
  shellCommands: ShellCommands
  modelProps: Omit<ComponentProps<typeof OmpModelSelector>, "view">
}

/** One workbench's chat and the text waiting to go into its composer draft. */
export interface WorkbenchChatState {
  chat: FileChat
  insertion: ComposerInsertion | null
  /** Put `text` into the composer draft, opening the drawer if it is closed. */
  quote: (text: string) => void
  onInserted: () => void
}

/** Width of the chat panel (the tab's file panel card, like the PDF assistant), the workbench's content shares what is left. */
const DRAWER_WIDTH_PX = 360

/** The chat linked to `key` (`useFileChat`), new sessions start in `cwd` and their first message with `lead`, with quotes into its draft. */
export function useWorkbenchChat(key: string, cwd: string, lead: string, { ompView, onViewChange, onDrawerOpenChange }: WorkbenchChatProps): WorkbenchChatState {
  const chat = useFileChat(key, cwd, ompView, onViewChange, lead)
  const [insertion, setInsertion] = useState<ComposerInsertion | null>(null)
  const insertionCount = useRef(0)
  const quote = (text: string) => {
    onDrawerOpenChange(true)
    setInsertion({ id: ++insertionCount.current, text })
  }
  return { chat, insertion, quote, onInserted: () => setInsertion(null) }
}

/** The chat drawer of a workbench in the tab's file panel, and the session's reasoning and model pickers in the top bar's `workbenchAgent` slot. */
export function WorkbenchChat({ state, drawerOpen, onDrawerOpenChange, shellCommands, modelProps }: Omit<WorkbenchChatProps, "ompView" | "onViewChange"> & { state: WorkbenchChatState }) {
  const { chat, insertion, onInserted } = state
  const { view } = chat
  return <>
    {drawerOpen && (
      <OmpChatPanel
        chat={chat}
        width={DRAWER_WIDTH_PX}
        onClose={() => onDrawerOpenChange(false)}
        shellCommands={shellCommands}
        insertion={insertion}
        onInserted={onInserted}
      />
    )}
    <InTopBar name="workbenchAgent">
      {view && <>
        <TopBarPill><OmpReasoningSelector view={view} /></TopBarPill>
        <TopBarTransparent><OmpModelSelector view={view} {...modelProps} /></TopBarTransparent>
      </>}
    </InTopBar>
  </>
}
