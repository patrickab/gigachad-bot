"use client"

import type { ReactNode } from "react"
import { useNeovimSetting } from "@/contexts/SettingsContext"
import { NeovimEditor } from "@/features/editor/NeovimEditor"
import { LOAD_MESSAGES } from "@/features/editor/FilePanel"
import { fileLead, projectLead } from "@/features/omp/fileLead"
import { EDITOR_PAGE } from "./pages"
import { quoteSelection } from "./quote"
import { useFollowBuffers } from "./useFollowBuffers"
import { WorkbenchChat, useWorkbenchChat, type WorkbenchChatProps } from "./WorkbenchChat"
import { useWorkbenchEditor } from "./useWorkbenchEditor"
import { EditorNote, WorkbenchFrame } from "./WorkbenchFrame"
import type { PaneLayout } from "./WorkbenchSplit"

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

export interface WorkbenchProps extends WorkbenchChatProps {
  /** Absolute disk path of the shown file. The first path starts Neovim, a later one switches its buffer, so one workbench can walk the files of a project. */
  path: string
  /** The directory a new chat session starts in. */
  cwd: string
  /** The project every file of the workbench belongs to: its folder's chat is shared by them all. Without one the chat belongs to the shown file. */
  project?: WorkbenchProject
  /** Neovim shows another file than `path` (`:bnext`), `path` should follow. Also asks to go back to the shown file when another one could not open. */
  onPathChange: (path: string) => void
  /** `:q` in Neovim. */
  onClose: () => void
  renderPane: (pane: WorkbenchPane) => ReactNode
  /** Shown in the `workbenchStatus` top-bar slot after the view switch. */
  status?: ReactNode
  /** A `:w` stored `text` as the file at `path`. */
  onSaved?: (text: string, path: string) => void
  /** Fence language of the lines `<leader>a` quotes into the chat. */
  quoteLanguage: string
}

/**
 * One file, or the files of one LaTeX project, as an Overleaf-like workspace: Neovim on the left, the rendered file in the middle, the omp chat in a drawer on the right.
 *
 * - Every open file follows its file on disk, whoever writes it (omp, another agent, a terminal, another device): a change is read and handed to Neovim, the backend merges it into unsaved edits
 * - Visual-mode `<leader>a` in Neovim quotes the selected lines into the chat draft, opening the drawer if it is closed
 * - The chat's session is linked to the project folder, else to the file. A new one starts in `cwd` and its first message names the project (`Project: <folder>, root: <root>`) or the file (`File: <path>`)
 * - A pasted image goes next to the shown file and its reference lands below the cursor
 * - The top bar's agent controls and status render into its `workbenchAgent` and `workbenchStatus` slots
 */
export function Workbench({ path, cwd, project, onPathChange, onClose, renderPane, status, onSaved, quoteLanguage, ...chatProps }: WorkbenchProps) {
  const neovim = useNeovimSetting()
  const chat = useWorkbenchChat(project?.folder ?? path, cwd, project ? projectLead(project.folder, project.root, cwd) : fileLead(path, cwd), chatProps)

  const editor = useWorkbenchEditor({ path, enabled: neovim.enabled, onPathChange, onSaved, onClose, onSelection: (selection) => chat.quote(quoteSelection(selection, quoteLanguage)) })
  const { content, loadError, notice } = editor
  useFollowBuffers({ synced: editor.synced, paths: editor.openPaths, setContent: editor.setContent })

  return (
    <WorkbenchFrame
      editor={loadError ? <EditorNote>{LOAD_MESSAGES[loadError]}</EditorNote>
        : content === null ? <EditorNote>Loading…</EditorNote>
        : !neovim.enabled ? <EditorNote>Neovim is not enabled.</EditorNote>
        : <NeovimEditor terminal={editor.terminal} status={editor.status} onRestart={editor.restart} className={EDITOR_PAGE} />}
      renderPane={(layout) => renderPane({ layout, text: editor.shownText, loadError: loadError ? LOAD_MESSAGES[loadError] : null })}
      chat={<WorkbenchChat state={chat} {...chatProps} />}
      status={<>
        {status}
        {notice && <span role="alert" className="truncate px-1 text-[11px] text-danger">{notice}</span>}
      </>}
    />
  )
}
