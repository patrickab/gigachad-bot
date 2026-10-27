"use client"

import { useRef, useState } from "react"
import { SelectionBubble } from "@/components/ui/SelectionBubble"
import { ErrorText } from "@/components/ui/ErrorText"
import { cn, isModKey } from "@/lib/utils"
import { useSelectionAction } from "@/hooks/useSelectionAction"
import { useNeovimSetting } from "@/contexts/SettingsContext"
import { NeovimEditor } from "@/features/editor/NeovimEditor"
import { ChangedFiles, ChangeSummary } from "./diffTree"
import { selectedComment } from "./diffSelection"
import { FileSection } from "./FileSection"
import { Annotations } from "./ReviewAnnotations"
import { LAYOUTS, useDiffLayout } from "./useDiffLayout"
import { useDiffNeovim } from "./useDiffNeovim"
import { jumpToNote, useDiffScroll } from "./useDiffScroll"
import { NO_NOTES, useReviewSession, type ReviewTarget } from "./useReviewSession"
import { useRootDiff } from "./useRootDiff"

const FOCUS_RING = "focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"

function Placeholder({ children }: { children: string }) {
  return <div className="flex h-full items-center justify-center px-4 text-center text-xs text-ink-faint">{children}</div>
}

/** Show working-tree changes with optional line notes and batched review delivery. `root` is fixed per mount: key the viewer by it. */
export function DiffViewer({ root, review }: { root: string; review?: ReviewTarget }) {
  const scrollContainer = useRef<HTMLDivElement>(null)
  const neovim = useNeovimSetting()
  const [revision, setRevision] = useState(0)
  const { changes, patches } = useRootDiff(root, revision)
  const files = changes?.files ?? []
  const { editing, openingPath, editError, clearEditError, editFile, session } = useDiffNeovim(root, () => setRevision((value) => value + 1))
  const opening = openingPath !== null
  const blocked = !!editing || opening
  const { selected, registerSection, jumpToFile, trackScroll } = useDiffScroll(scrollContainer, files, blocked)
  const { layout, choose: chooseLayout, split } = useDiffLayout(scrollContainer)
  const { notes, notesByPath, draft, setDraft, actions, send, sending, sendError, copied } = useReviewSession(root, review, files, patches)
  const { selected: selectionAction, clear: clearSelectionAction } = useSelectionAction(scrollContainer, (container) => selectedComment(container, patches), { live: true })

  const activePath = editing ? editing.path.slice(root.length + 1) : files.some((file) => file.path === selected) ? selected : files[0]?.path ?? null
  const selectFile = (path: string) => {
    if (blocked) return
    clearEditError()
    jumpToFile(path)
  }

  const content = changes === undefined ? <Placeholder>Loading changes</Placeholder>
    : !changes ? <Placeholder>Could not read this directory</Placeholder>
    : !changes.is_git_repository ? <Placeholder>This file mount is not a git repository</Placeholder>
    : !activePath ? <Placeholder>No changes</Placeholder>
    : files.map((file) => <FileSection
      key={file.path} path={file.path} added={file.added} removed={file.removed} patch={patches[file.path]}
      editable={neovim.enabled && file.status !== "deleted"} opening={opening} openingThis={openingPath === file.path}
      onEdit={editFile} register={registerSection}
      split={split} notes={notesByPath.get(file.path) ?? NO_NOTES} draft={draft?.anchor.path === file.path ? draft : null} actions={actions}
    />)

  return <div className="flex h-full">
    <nav aria-label="Changed files" className="mb-3 ml-3 flex w-72 shrink-0 flex-col rounded-2xl border border-divider/50 bg-surface shadow-[var(--shadow-lg)]">
      <div className="min-h-0 flex-1 overflow-y-auto px-2 py-3">
        {files.length > 0 && <ChangeSummary files={files} />}
        <ChangedFiles files={files} activePath={activePath} onSelect={selectFile} root={root} notesByPath={notesByPath} />
        {changes && [...notesByPath].filter(([path]) => !files.some((file) => file.path === path)).map(([path, missingNotes]) => <div key={path} className="mt-3">
          <p className="truncate px-1 text-xs text-ink-faint">{path} · no longer in diff</p>
          <Annotations notes={missingNotes} draft={draft} showNewDraft={false} outdated actions={actions} className="pl-1" />
        </div>)}
      </div>
      {notes.length > 0 && <div aria-label="Review" className="border-t border-divider/50 px-3 py-2.5">
        <div className="flex items-center gap-2 text-xs text-ink-muted">
          <span className="flex-1 tabular-nums">{notes.length} {notes.length === 1 ? "note" : "notes"}</span>
          <button
            type="button"
            disabled={(review && !review.canSend) || sending}
            onClick={() => void send()}
            className={cn("rounded-full bg-ink px-3 py-1.5 text-xs font-medium text-paper transition-opacity hover:opacity-90 focus-visible:outline-offset-2 disabled:opacity-40", FOCUS_RING)}
          >{sending ? "Exporting…" : review ? "Send review" : "Copy review"}</button>
        </div>
        {review && !review.canSend && <p className="mt-1.5 text-[11px] text-ink-faint">Sendable once the agent waits for you</p>}
        {!review && copied && <p role="status" className="mt-1.5 text-[11px] text-ink-faint">Review copied</p>}
        <ErrorText className="mt-1.5">{sendError}</ErrorText>
      </div>}
    </nav>
    <div className="flex min-w-0 flex-1 flex-col">
      <div className="flex h-7 shrink-0 items-center gap-3 px-3 text-[10px] text-ink-faint">
        <span className="min-w-0 flex-1 truncate">{activePath ?? ""}</span>
        {editing ? <span>HEAD · working file · :w saves · :qa returns</span> : <div role="group" aria-label="Diff layout" className="flex shrink-0 gap-0.5">
          {LAYOUTS.map((option) => <button
            key={option}
            type="button"
            aria-pressed={layout === option}
            onClick={() => chooseLayout(option)}
            className={cn("rounded px-1.5 py-0.5 capitalize", FOCUS_RING, layout === option ? "text-ink" : "hover:text-ink")}
          >{option}</button>)}
        </div>}
      </div>
      <ErrorText className="px-3 py-1">{editError}</ErrorText>
      {editing && <div className="min-h-0 flex-1 overflow-hidden"><NeovimEditor terminal={session.terminal} status={session.status} onRestart={session.restart} className="h-full" /></div>}
      <div
        ref={scrollContainer}
        role="region"
        aria-label="Diff"
        tabIndex={0}
        onScroll={trackScroll}
        onKeyDown={(event) => {
          const arrow = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0
          if (!arrow || notes.length === 0 || event.target instanceof HTMLTextAreaElement || event.altKey || isModKey(event) || event.shiftKey) return
          event.preventDefault()
          jumpToNote(event.currentTarget, arrow)
        }}
        className={cn("@container min-h-0 flex-1 overflow-auto focus-visible:outline focus-visible:outline-1 focus-visible:-outline-offset-1 focus-visible:outline-ink-muted/40", editing && "hidden")}
      >
        {content}
      </div>
      {selectionAction && !editing && <SelectionBubble
        bounds={selectionAction.bounds}
        label="Add comment to selection"
        onActivate={() => {
          setDraft({ anchor: selectionAction.anchor, noteId: null, comment: "" })
          clearSelectionAction()
          window.getSelection()?.removeAllRanges()
        }}
      />}
    </div>
  </div>
}
