"use client"

import { useRef, useState, type ComponentProps, type ReactNode } from "react"
import { useNeovimSetting } from "@/contexts/SettingsContext"
import { NeovimEditor, type NeovimSelection } from "@/features/editor/NeovimEditor"
import { LOAD_MESSAGES } from "@/features/editor/FilePanel"
import { TopBarPill, TopBarTransparent } from "@/features/navigation/TopBar"
import { InTopBar } from "@/features/navigation/TopBarSlots"
import { OmpModelSelector, OmpReasoningSelector, type ComposerInsertion, type ShellCommands } from "@/features/omp/OmpComposer"
import { OmpChatPanel } from "@/features/omp/OmpChatPanel"
import { fileLead, projectLead } from "@/features/omp/fileLead"
import { useFileChat } from "@/features/omp/useFileChat"
import type { OmpView } from "@/lib/domain/types"
import { cn } from "@/lib/utils"
import { quoteSelection } from "./quote"
import { useFollowBuffers } from "./useFollowBuffers"
import { useWorkbenchEditor } from "./useWorkbenchEditor"
import { WorkbenchViewSwitch, useWorkbenchView, usePaneLayout, type PaneLayout } from "./WorkbenchSplit"

/** What the rendered pane draws from: where it sits, the synced text, and why it could not load. */
export interface WorkbenchPane {
  layout: PaneLayout
  /** The text Neovim and the disk last agreed on for the shown file: the loaded file, then every save and every agent edit. Null while loading or after a failed load. */
  text: string | null
  /** Why the file could not load, null otherwise. */
  loadError: string | null
}

/** The LaTeX project a workbench's files belong to. */
interface WorkbenchProject {
  /** The project folder, the key of the project's chat. */
  folder: string
  /** The root file, null when none was found. */
  root: string | null
}

export interface WorkbenchProps {
  /** Absolute disk path of the shown file. The first path starts Neovim, a later one switches its buffer, so one workbench can walk the files of a project. */
  path: string
  /** The directory a new chat session starts in. */
  cwd: string
  /** The project every file of the workbench belongs to: its folder's chat is shared by them all. Without one the chat belongs to the shown file. */
  project?: WorkbenchProject
  /** The chat session this tab's item shows, undefined before the first one. */
  ompView: OmpView | undefined
  onViewChange: (view: OmpView) => void
  /** Neovim shows another file than `path` (`:bnext`), `path` should follow. Also asks to go back to the shown file when another one could not open. */
  onPathChange: (path: string) => void
  drawerOpen: boolean
  onDrawerOpenChange: (open: boolean) => void
  /** `:q` in Neovim. */
  onClose: () => void
  shellCommands: ShellCommands
  modelProps: Omit<ComponentProps<typeof OmpModelSelector>, "view">
  renderPane: (pane: WorkbenchPane) => ReactNode
  /** Shown in the `workbenchStatus` top-bar slot after the view switch. */
  status?: ReactNode
  /** A `:w` stored `text` as the file at `path`. */
  onSaved?: (text: string, path: string) => void
  /** Fence language of the lines `<leader>a` quotes into the chat. */
  quoteLanguage: string
}

/** Width of the chat panel (the tab's file panel card, like the PDF assistant), editor and rendered file share what is left. */
const DRAWER_WIDTH_PX = 360
/** Neovim sits on a page like the PDF's (`.pdf-page`), its size set by the pane layout. */
const EDITOR_PAGE = "rounded-2xl px-3 pt-3 shadow-[var(--shadow-md)]"

const CENTERED_NOTE = "flex min-h-0 flex-1 items-center justify-center text-xs text-ink-faint"

/**
 * One file, or the files of one LaTeX project, as an Overleaf-like workspace: Neovim on the left, the rendered file in the middle, the omp chat in a drawer on the right.
 *
 * - Every open file follows its file on disk, whoever writes it (omp, another agent, a terminal, another device): a change is read and handed to Neovim, the backend merges it into unsaved edits
 * - Visual-mode `<leader>a` in Neovim quotes the selected lines into the chat draft, opening the drawer if it is closed
 * - The chat's session is linked to the project folder, else to the file. A new one starts in `cwd` and its first message names the project (`Project: <folder>, root: <root>`) or the file (`File: <path>`)
 * - A pasted image goes next to the shown file and its reference lands below the cursor
 * - The top bar's agent controls and status render into its `workbenchAgent` and `workbenchStatus` slots
 */
export function Workbench({
  path, cwd, project, ompView, onViewChange, onPathChange, drawerOpen, onDrawerOpenChange, onClose, shellCommands, modelProps,
  renderPane, status, onSaved, quoteLanguage,
}: WorkbenchProps) {
  const neovim = useNeovimSetting()
  const chat = useFileChat(
    project?.folder ?? path,
    cwd,
    ompView,
    onViewChange,
    project ? projectLead(project.folder, project.root, cwd) : fileLead(path, cwd),
  )
  const { view } = chat

  const [insertion, setInsertion] = useState<ComposerInsertion | null>(null)
  const insertionCount = useRef(0)
  const quoteIntoChat = (selection: NeovimSelection) => {
    onDrawerOpenChange(true)
    setInsertion({ id: ++insertionCount.current, text: quoteSelection(selection, quoteLanguage) })
  }

  const editor = useWorkbenchEditor({ path, enabled: neovim.enabled, onPathChange, onSaved, onClose, onSelection: quoteIntoChat })
  const { content, loadError, notice } = editor
  useFollowBuffers({ synced: editor.synced, paths: editor.openPaths, setContent: editor.setContent })

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
            {renderPane({ layout, text: editor.shownText, loadError: loadError ? LOAD_MESSAGES[loadError] : null })}
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
          {notice && <span role="alert" className="truncate px-1 text-[11px] text-danger">{notice}</span>}
        </TopBarPill>
      </InTopBar>
    </div>
  )
}
