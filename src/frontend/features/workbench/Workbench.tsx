"use client"

import { useCallback, useEffect, useRef, useState, type ComponentProps, type ReactNode } from "react"
import { useNeovimSetting } from "@/contexts/SettingsContext"
import { NeovimEditor, useNeovimTerminal, type NeovimSelection } from "@/features/editor/NeovimEditor"
import { LOAD_MESSAGES } from "@/features/editor/FilePanel"
import { TEXT_CODEC, useRootFileDocument } from "@/features/editor/useRootFileDocument"
import { TopBarPill, TopBarTransparent } from "@/features/navigation/TopBar"
import { InTopBar } from "@/features/navigation/TopBarSlots"
import { OmpModelSelector, OmpReasoningSelector, useOmpStreaming, type ComposerInsertion, type ShellCommands } from "@/features/omp/OmpComposer"
import { OmpChatPanel } from "@/features/omp/OmpChatPanel"
import { useFileChat } from "@/features/omp/useFileChat"
import { writeRootFile } from "@/lib/backend/api"
import type { OmpView } from "@/lib/domain/types"
import { baseName } from "@/lib/paths"
import { cn } from "@/lib/utils"
import { quoteSelection } from "./quote"
import { useAgentSync } from "./useAgentSync"
import { WorkbenchViewSwitch, useWorkbenchView, usePaneLayout, type PaneLayout } from "./WorkbenchSplit"

/** What the rendered pane draws from: where it sits, the synced text, and why it could not load. */
export interface WorkbenchPane {
  layout: PaneLayout
  /** The text Neovim and the disk last agreed on: the loaded file, then every save and every agent edit. Null while loading or after a failed load. */
  text: string | null
  /** Why the file could not load, null otherwise. */
  loadError: string | null
}

export interface WorkbenchProps {
  /** Absolute disk path of the file. */
  path: string
  /** The directory a new chat session starts in. */
  cwd: string
  /** The chat session this tab's item shows, undefined before the first one. */
  ompView: OmpView | undefined
  onViewChange: (view: OmpView) => void
  drawerOpen: boolean
  onDrawerOpenChange: (open: boolean) => void
  /** `:q` in Neovim. */
  onClose: () => void
  shellCommands: ShellCommands
  modelProps: Omit<ComponentProps<typeof OmpModelSelector>, "view">
  renderPane: (pane: WorkbenchPane) => ReactNode
  /** Shown in the `workbenchStatus` top-bar slot after the view switch. */
  status?: ReactNode
  /** A `:w` stored `text`. */
  onSaved?: (text: string) => void
  /** An agent edit or another device's save of the file reached Neovim as `text`. */
  onPulled?: (text: string) => void
  /** Called once an omp run's last edits are in Neovim. */
  onRunEnd?: () => void
  /** Fence language of the lines `<leader>a` quotes into the chat. */
  quoteLanguage: string
}

/** Width of the chat panel (the tab's file panel card, like the PDF assistant), editor and rendered file share what is left. */
const DRAWER_WIDTH_PX = 360
/** Neovim sits on a page like the PDF's (`.pdf-page`), its size set by the pane layout. */
const EDITOR_PAGE = "rounded-2xl px-3 pt-3 shadow-[var(--shadow-md)]"
/** The view that shows no session, so no agent runs in it. */
const NO_VIEW: OmpView = { kind: "settings" }

const CENTERED_NOTE = "flex min-h-0 flex-1 items-center justify-center text-xs text-ink-faint"

/**
 * One file as an Overleaf-like workspace: Neovim on the left, the rendered file in the middle, the omp chat in a drawer on the right.
 *
 * - While omp runs, its edits on disk flow into Neovim as they arrive, each one updates the synced text and reaches `onPulled`
 * - Visual-mode `<leader>a` in Neovim quotes the selected lines into the chat draft, opening the drawer if it is closed
 * - The chat's session is linked to the file, a new one starts in `cwd` and its first message names the file (`File: <path>`)
 * - Another device's save of the file reaches Neovim, merged into unsaved edits by the backend
 * - The top bar's agent controls and status render into its `workbenchAgent` and `workbenchStatus` slots
 */
export function Workbench({
  path, cwd, ompView, onViewChange, drawerOpen, onDrawerOpenChange, onClose, shellCommands, modelProps,
  renderPane, status, onSaved, onPulled, onRunEnd, quoteLanguage,
}: WorkbenchProps) {
  const neovim = useNeovimSetting()
  // A rename changes `path` under the running editor: its session goes on and learns the new name, it never restarts
  const [sessionKey] = useState(path)
  const chat = useFileChat(path, cwd, ompView, onViewChange)
  const { view } = chat
  const streaming = useOmpStreaming(view ?? NO_VIEW)

  const write = useCallback((text: string) => writeRootFile(path, text), [path])
  const { value: content, loadError, reload, saveFromNeovim } = useRootFileDocument({ path, codec: TEXT_CODEC, write, blankWhenMissing: false })
  useEffect(() => { reload() }, [reload])
  // The text Neovim and the disk last agreed on: what was loaded, then what each save or push stored.
  // The ref serves the agent sync, the state the rendered pane
  const syncedRef = useRef<string | null>(null)
  const [syncedText, setSyncedText] = useState<string | null>(null)
  useEffect(() => {
    if (content !== null && syncedRef.current === null) syncedRef.current = content
  }, [content])

  const save = async (text: string) => {
    const stored = await saveFromNeovim(text)
    syncedRef.current = stored
    setSyncedText(stored)
    onSaved?.(stored)
    return stored
  }

  const [insertion, setInsertion] = useState<ComposerInsertion | null>(null)
  const insertionCount = useRef(0)
  const quoteIntoChat = (selection: NeovimSelection) => {
    onDrawerOpenChange(true)
    setInsertion({ id: ++insertionCount.current, text: quoteSelection(path, selection, quoteLanguage) })
  }

  const editor = useNeovimTerminal(
    neovim.enabled && content !== null ? { key: sessionKey, path, filename: baseName(path), content } : null,
    save,
    onClose,
    { askKeymap: true, onSelection: quoteIntoChat },
  )
  const { setContent: pushToNeovim } = editor
  const setContent = useCallback((text: string) => {
    const taken = pushToNeovim(text)
    if (taken) setSyncedText(text)
    return taken
  }, [pushToNeovim])
  useAgentSync({ path, streaming, setContent, syncedRef, onPulled, onRunEnd })

  const [workbenchView, setWorkbenchView] = useWorkbenchView()
  const splitRef = useRef<HTMLDivElement>(null)
  const layout = usePaneLayout(splitRef, workbenchView)

  // Editor and rendered file share the width and line up at the top, rendered pages scroll on below.
  // A hidden pane stays mounted, so Neovim keeps its session
  return (
    <div className="flex h-full">
      {/* The padding sits outside the measured area: `useElementSize` reads the content box */}
      <div className="flex min-w-0 flex-1 px-3">
        <div ref={splitRef} className="flex min-w-0 flex-1">
          <div className={cn("flex min-w-0 flex-1 flex-col items-center", workbenchView === "render" && "hidden")}>
            <div className="flex flex-col" style={{ marginTop: layout.top, width: layout.width, height: layout.height }}>
              {loadError ? <div className={CENTERED_NOTE}>{LOAD_MESSAGES[loadError]}</div>
                : content === null ? <div className={CENTERED_NOTE}>Loading…</div>
                : !neovim.enabled ? <div className={CENTERED_NOTE}>Neovim is not enabled.</div>
                : <NeovimEditor terminal={editor.terminal} status={editor.status} onRestart={editor.restart} className={EDITOR_PAGE} />}
            </div>
          </div>
          <div className={cn("min-w-0 flex-1", workbenchView === "editor" && "hidden")}>
            {renderPane({ layout, text: syncedText ?? content, loadError: loadError ? LOAD_MESSAGES[loadError] : null })}
          </div>
        </div>
      </div>
      {drawerOpen && (
        <OmpChatPanel
          chat={chat}
          width={DRAWER_WIDTH_PX}
          onClose={() => onDrawerOpenChange(false)}
          shellCommands={shellCommands}
          insertion={insertion}
          onInserted={() => setInsertion(null)}
        />
      )}
      <InTopBar name="workbenchAgent">
        {view && <>
          <TopBarPill><OmpReasoningSelector view={view} /></TopBarPill>
          <TopBarTransparent><OmpModelSelector view={view} {...modelProps} /></TopBarTransparent>
        </>}
      </InTopBar>
      <InTopBar name="workbenchStatus">
        <TopBarPill>
          <WorkbenchViewSwitch view={workbenchView} onChange={setWorkbenchView} />
          {status}
        </TopBarPill>
      </InTopBar>
    </div>
  )
}
