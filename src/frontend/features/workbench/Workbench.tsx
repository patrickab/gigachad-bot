"use client"

import type { ReactNode } from "react"
import { flushSync } from "react-dom"
import { useNeovimSetting } from "@/contexts/SettingsContext"
import { NeovimEditor } from "@/features/editor/NeovimEditor"
import { LOAD_MESSAGES } from "@/features/editor/FilePanel"
import type { NeovimCommand } from "@/features/editor/neovimProtocol"
import { fileLead, projectLead } from "@/features/omp/fileLead"
import { EDITOR_PAGE } from "./pages"
import { quoteSelection } from "./quote"
import { useFollowBuffers } from "./useFollowBuffers"
import { WorkbenchChat, useWorkbenchChat, type WorkbenchChatProps } from "./WorkbenchChat"
import { useWorkbenchEditor } from "./useWorkbenchEditor"
import type { WorkbenchNoticeChannel } from "./useWorkbenchNotice"
import { EditorNote, WorkbenchFrame, WorkbenchNotice } from "./WorkbenchFrame"
import { useWorkbenchView, type PaneLayout } from "./WorkbenchSplit"

/** The source of the editor's notices in the workbench's notice channel. */
const EDITOR_NOTICE = "editor"

/** What the rendered pane and the `center` controls may do to Neovim. Refusals show in the workbench's notice. */
export interface WorkbenchEditorActions {
  /** Put Neovim's cursor on 1-based `line` and 0-based UTF-8 byte `column` (0 without one) of the file at `path`, switching buffers when it shows another file. `focus` also brings Neovim's page back when the view hides it and focuses Neovim. */
  reveal: (path: string, line: number, column?: number, options?: { focus?: boolean }) => void
  /** Hand `command` to Neovim. False while the session is not open, the notice then says `refusal`. */
  send: (command: NeovimCommand, refusal: string) => boolean
  focus: () => void
}

/** What the rendered pane draws from: where it sits, the synced text, why it could not load, and what it may do to Neovim. */
export interface WorkbenchPane {
  layout: PaneLayout
  /** The text Neovim and the disk last agreed on for the shown file: the loaded file, then every save and every agent edit. Null while loading or after a failed load. */
  text: string | null
  /** Why the file could not load, null otherwise. */
  loadError: string | null
  /** Null while Neovim is not enabled. */
  editor: WorkbenchEditorActions | null
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
  /** Controls in the top bar's `center` slot, given what they may do to Neovim. Only while Neovim is enabled. */
  center?: (editor: WorkbenchEditorActions) => ReactNode
  /** The notice channel the owner reports into, the editor reports there too. */
  notices: WorkbenchNoticeChannel
  /** A `:w` stored `text` as the file at `path`. */
  onSaved?: (text: string, path: string) => void
  /** Fence language of the lines `<leader>a` quotes into the chat. */
  quoteLanguage: string
  /** Normal-mode `<leader>v` in a `.tex` buffer asked for the rendered spot of 1-based `line` of the file at `path`. Resolves whether the rendered pane moved to it, the workbench then brings the pane back when the view hides it. */
  onForwardSearch?: (path: string, line: number) => Promise<boolean>
}

/**
 * One file, or the files of one LaTeX project, as an Overleaf-like workspace: Neovim on the left, the rendered file in the middle, the omp chat in a drawer on the right.
 *
 * - Every open file follows its file on disk, whoever writes it (omp, another agent, a terminal, another device): a change is read and handed to Neovim, the backend merges it into unsaved edits
 * - Visual-mode `<leader>a` in Neovim quotes the selected lines into the chat draft, opening the drawer if it is closed
 * - Normal-mode `<leader>v` in a `.tex` buffer goes to `onForwardSearch` with its file and cursor line
 * - The chat's session is linked to the project folder, else to the file. A new one starts in `cwd` and its first message names the project (`Project: <folder>, root: <root>`) or the file (`File: <path>`)
 * - A pasted image is stored in `imgs/` next to the shown file and pasted as its reference, Ctrl+Shift+V fills an `example-image` placeholder on the cursor line with it
 * - The top bar's agent controls and status render into its `workbenchAgent` and `workbenchStatus` slots, `center` into its `center` slot while Neovim is enabled
 * - One notice in the status says why the latest thing beside the editor did not work, the owner's and the editor's alike
 */
export function Workbench({ path, cwd, project, onPathChange, onClose, renderPane, status, center, notices, onSaved, quoteLanguage, onForwardSearch, ...chatProps }: WorkbenchProps) {
  const neovim = useNeovimSetting()
  const chat = useWorkbenchChat(project?.folder ?? path, cwd, project ? projectLead(project.folder, project.root, cwd) : fileLead(path, cwd), chatProps)
  const [view, setView, show] = useWorkbenchView()
  const { notice, report } = notices

  const editor = useWorkbenchEditor({
    path,
    enabled: neovim.enabled,
    onPathChange,
    onSaved,
    onClose,
    onSelection: (selection) => chat.quote(quoteSelection(selection, quoteLanguage)),
    onForwardSearch: onForwardSearch && ((target, line) => void onForwardSearch(target, line).then((moved) => {
      if (moved) show("render")
    })),
    onNotice: (text) => report(EDITOR_NOTICE, text),
  })
  const { content, loadError } = editor
  useFollowBuffers({ synced: editor.synced, paths: editor.openPaths, setContent: editor.setContent })

  const actions: WorkbenchEditorActions | null = neovim.enabled ? {
    reveal: (target, line, column, { focus = false } = {}) => {
      editor.reveal(target, line, column)
      if (!focus) return
      // A page the view brings back takes focus only once it is laid out
      flushSync(() => show("editor"))
      editor.terminal?.focus()
    },
    send: editor.send,
    focus: () => editor.terminal?.focus(),
  } : null

  return (
    <WorkbenchFrame
      view={view}
      onViewChange={setView}
      editor={loadError ? <EditorNote>{LOAD_MESSAGES[loadError]}</EditorNote>
        : content === null ? <EditorNote>Loading…</EditorNote>
        : !neovim.enabled ? <EditorNote>Neovim is not enabled.</EditorNote>
        : <NeovimEditor terminal={editor.terminal} status={editor.status} onRestart={editor.restart} className={EDITOR_PAGE} />}
      renderPane={(layout) => renderPane({
        layout,
        text: editor.shownText,
        loadError: loadError ? LOAD_MESSAGES[loadError] : null,
        editor: actions,
      })}
      chat={<WorkbenchChat state={chat} {...chatProps} />}
      status={<>
        {status}
        {notice && <WorkbenchNotice>{notice}</WorkbenchNotice>}
      </>}
      center={actions ? center?.(actions) : undefined}
    />
  )
}
