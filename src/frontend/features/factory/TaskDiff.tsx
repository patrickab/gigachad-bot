"use client"

import { useRef } from "react"
import { useColumnWidth } from "@/components/ui/columnResize"
import { SelectionBubble } from "@/components/ui/SelectionBubble"
import { selectedComment } from "@/features/editor/diff/diffSelection"
import { ChangedFiles, ChangeSummary } from "@/features/editor/diff/diffTree"
import { FileSection } from "@/features/editor/diff/FileSection"
import type { ReviewNote } from "@/features/editor/diff/review"
import { Annotations } from "@/features/editor/diff/ReviewAnnotations"
import { LAYOUTS, useDiffLayout } from "@/features/editor/diff/useDiffLayout"
import { useDiffScroll } from "@/features/editor/diff/useDiffScroll"
import { NO_NOTES, type Draft, type ReviewActions } from "@/features/editor/diff/useReviewSession"
import type { FilePatch } from "@/features/editor/diff/useRootDiff"
import { useSelectionAction } from "@/hooks/useSelectionAction"
import { cn } from "@/lib/utils"
import type { TaskDiffFile } from "./useTaskDiff"

const FOCUS_RING = "focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"

/** A task's files are not edited from its diff. */
const noEdit = () => {}

/** The pending review comments of the task, the open draft and what edits them, from `useTaskReview`. */
export interface TaskDiffReview {
  notesByPath: Map<string, ReviewNote[]>
  draft: Draft | null
  setDraft: (draft: Draft) => void
  actions: ReviewActions
  patches: Record<string, FilePatch>
}

function Placeholder({ children }: { children: string }) {
  return <div className="flex h-full items-center justify-center px-4 text-center text-xs text-ink-faint">{children}</div>
}

/**
 * A task's changes as the editor's diff view shows a checkout's: the changed files as a tree, one section per file,
 * unified or split, laid out like `DiffViewer`. `files` come parsed from `useTaskDiff` (undefined while the first read runs).
 *
 * - `root` names the checkout in the tree's label
 * - Selected code offers a comment on its line; comments show under their line, those whose line or file left the diff
 *   above their file or under the tree
 */
export function TaskDiff({ files, root, review }: { files: TaskDiffFile[] | undefined; root: string; review: TaskDiffReview }) {
  const scrollContainer = useRef<HTMLDivElement>(null)
  const shown = files ?? []
  const { selected, registerSection, jumpToFile, trackScroll } = useDiffScroll(scrollContainer, shown, false)
  const { layout, choose: chooseLayout, split } = useDiffLayout(scrollContainer)
  const { width: columnWidth, startResize } = useColumnWidth()
  const { notesByPath, draft, setDraft, actions, patches } = review
  const { selected: selectionAction, clear: clearSelectionAction } = useSelectionAction(scrollContainer, (container) => selectedComment(container, patches), { live: true })

  const activePath = shown.some((file) => file.path === selected) ? selected : shown[0]?.path ?? null

  return <div className="flex h-full">
    <nav aria-label="Changed files" className="mb-3 ml-3 flex w-72 shrink-0 flex-col rounded-2xl border border-divider/50 bg-surface shadow-[var(--shadow-lg)]">
      <div className="min-h-0 flex-1 overflow-y-auto px-2 py-3">
        {shown.length > 0 && <ChangeSummary files={shown} />}
        <ChangedFiles files={shown} activePath={activePath} onSelect={jumpToFile} root={root} notesByPath={notesByPath} />
        {files && [...notesByPath].filter(([path]) => !files.some((file) => file.path === path)).map(([path, missingNotes]) => <div key={path} className="mt-3">
          <p className="truncate px-1 text-xs text-ink-faint">{path} · no longer in diff</p>
          <Annotations notes={missingNotes} draft={draft} showNewDraft={false} outdated actions={actions} className="pl-1" />
        </div>)}
      </div>
    </nav>
    <div className="flex min-w-0 flex-1 flex-col">
      <div className="flex h-7 shrink-0 items-center gap-3 px-3 text-[10px] text-ink-faint">
        <span className="min-w-0 flex-1 truncate">{activePath ?? ""}</span>
        <div role="group" aria-label="Diff layout" className="flex shrink-0 gap-0.5">
          {LAYOUTS.map((option) => <button
            key={option}
            type="button"
            aria-pressed={layout === option}
            onClick={() => chooseLayout(option)}
            className={cn("rounded px-1.5 py-0.5 capitalize", FOCUS_RING, layout === option ? "text-ink" : "hover:text-ink")}
          >{option}</button>)}
        </div>
      </div>
      <div
        ref={scrollContainer}
        role="region"
        aria-label="Diff"
        tabIndex={0}
        onScroll={trackScroll}
        className="@container min-h-0 flex-1 overflow-y-auto px-3 pb-3 focus-visible:outline focus-visible:outline-1 focus-visible:-outline-offset-1 focus-visible:outline-ink-muted/40"
      >
        {!files ? <Placeholder>Loading changes</Placeholder>
          : files.length === 0 ? <Placeholder>No changes</Placeholder>
          : <div data-chat-column className="mx-auto flex w-full flex-col gap-4" style={{ maxWidth: columnWidth }}>
            {files.map((file) => <FileSection
              key={file.path} path={file.path} added={file.added} removed={file.removed} patch={file.patch}
              editable={false} opening={false} openingThis={false} onEdit={noEdit} register={registerSection}
              split={split} notes={notesByPath.get(file.path) ?? NO_NOTES} draft={draft?.anchor.path === file.path ? draft : null}
              actions={actions} onResizeStart={startResize}
            />)}
          </div>}
      </div>
      {selectionAction && <SelectionBubble
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
