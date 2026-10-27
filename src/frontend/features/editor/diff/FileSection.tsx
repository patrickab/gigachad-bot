import { Fragment, memo, useMemo } from "react"
import { toSplitRows, type PatchLine } from "@/lib/diff"
import { cn } from "@/lib/utils"
import { LineCountLabel } from "./diffTree"
import { locateNote, type ReviewNote, type ReviewSide } from "./review"
import { Annotations } from "./ReviewAnnotations"
import { NO_NOTES, type Draft, type ReviewActions } from "./useReviewSession"
import type { FilePatch } from "./useRootDiff"

type PlacedNotes = { byIndex: Map<number, ReviewNote[]>; outdated: ReviewNote[]; draftIndex: number | null }

const LINE_CLASS: Record<PatchLine["kind"], string> = {
  hunk: "bg-surface text-ink-faint",
  added: "bg-diff-added/10 text-diff-added",
  removed: "bg-diff-removed/10 text-diff-removed",
  context: "text-ink-muted",
  note: "text-ink-faint italic",
}

const SIGN: Partial<Record<PatchLine["kind"], string>> = { added: "+", removed: "-" }

const NUMBER_CLASS = "w-12 shrink-0 select-none pr-2 text-right tabular-nums text-ink-faint"

function LineNumbers({ numbers }: { numbers: (number | null)[] }) {
  return <>{numbers.map((number, column) => <span key={column} className={NUMBER_CLASS}>{number}</span>)}</>
}

function placeNotes(notes: ReviewNote[], draft: Draft | null, lines: PatchLine[]): PlacedNotes {
  const byIndex = new Map<number, ReviewNote[]>()
  const outdated: ReviewNote[] = []
  for (const note of notes) {
    const index = locateNote(note, lines)
    if (index === null) outdated.push(note)
    else byIndex.set(index, [...(byIndex.get(index) ?? []), note])
  }
  const draftIndex = draft && draft.noteId === null ? locateNote(draft.anchor, lines) : null
  return { byIndex, outdated, draftIndex }
}

type PatchViewProps = { path: string; lines: PatchLine[]; split: boolean; placed: PlacedNotes; draft: Draft | null; actions: ReviewActions }

/** Memoized: a patch keeps its `lines` array until its file changes, so polls and scrolling skip the large line lists. */
const PatchView = memo(function PatchView({ path, lines, split, placed, draft, actions }: PatchViewProps) {
  const rows = useMemo(() => split ? toSplitRows(lines) : null, [split, lines])
  const annotations = (index: number | null, className: string) => index !== null && <Annotations
    notes={placed.byIndex.get(index) ?? NO_NOTES} draft={draft} showNewDraft={placed.draftIndex === index}
    outdated={false} actions={actions} className={className}
  />

  if (rows) {
    const cell = (index: number | null, side: ReviewSide) => {
      const line = index === null ? null : lines[index]!
      if (!line) return <div className="bg-surface/40" />
      const number = side === "old" ? line.oldNumber : line.newNumber
      return <div className={cn("flex min-w-0", LINE_CLASS[line.kind])}>
        <LineNumbers numbers={[number]} />
        <span className="w-4 shrink-0 select-none text-center text-ink-faint">{SIGN[line.kind]}</span>
        <span data-diff-code data-path={path} data-index={index} data-side={side} data-column={side} className="min-w-0 whitespace-pre-wrap break-all pr-4">{line.text}</span>
      </div>
    }
    return <div className="py-2 font-mono text-xs leading-5">
      {rows.map((row, position) => "full" in row
        ? <div key={position} className={cn("pl-16", LINE_CLASS[lines[row.full]!.kind])}>{lines[row.full]!.text}</div>
        : <Fragment key={position}>
          <div className="grid grid-cols-2">{cell(row.old, "old")}{cell(row.new, "new")}</div>
          {annotations(row.old, "pl-16")}
          {row.new !== row.old && annotations(row.new, "pl-16")}
        </Fragment>)}
    </div>
  }

  return <div className="w-max min-w-full py-2 font-mono text-xs leading-5">
    {lines.map((line, index) => {
      const side: ReviewSide = line.kind === "removed" ? "old" : "new"
      return <Fragment key={index}>
        <div className={cn("flex", LINE_CLASS[line.kind])}>
          <LineNumbers numbers={[line.oldNumber, line.newNumber]} />
          <span className="w-4 shrink-0 select-none text-center text-ink-faint">{SIGN[line.kind]}</span>
          <span data-diff-code data-path={path} data-index={index} data-side={side} data-column="unified" className="whitespace-pre pr-4">{line.text}</span>
        </div>
        {annotations(index, "sticky left-0 w-[min(44rem,100cqw)] pl-28")}
      </Fragment>
    })}
  </div>
})

/** One file of the diff, props are primitives or stable references so unchanged files skip re-rendering. */
export const FileSection = memo(function FileSection({ path, added, removed, patch, editable, opening, openingThis, onEdit, register, split, notes, draft, actions }: {
  path: string; added: number | null; removed: number | null; patch: FilePatch | undefined
  editable: boolean; opening: boolean; openingThis: boolean
  onEdit: (path: string) => void; register: (path: string, element: HTMLElement | null) => void
  split: boolean; notes: ReviewNote[]; draft: Draft | null; actions: ReviewActions
}) {
  const lines = patch?.lines ?? null
  const placed = useMemo(() => placeNotes(notes, draft, lines ?? []), [notes, draft, lines])
  return <section aria-label={`Diff for ${path}`} ref={(element) => register(path, element)} className="min-h-24 border-b border-divider/50 last:min-h-full">
    <h3 className="sticky left-0 top-0 z-10 flex items-center gap-2 bg-paper px-3 py-2 font-mono text-xs text-ink-muted">
      <span className="min-w-0 flex-1 truncate">{path}</span>
      {added !== null && removed !== null && <span aria-label={`Line changes in ${path}`} className="flex shrink-0 gap-2 tabular-nums">
        <LineCountLabel counts={{ added, removed }} />
      </span>}
      {editable && <button
        disabled={opening}
        onClick={() => onEdit(path)}
        className="-my-1 shrink-0 rounded px-2 py-1 font-sans text-xs text-ink-faint hover:text-ink disabled:opacity-50"
      >{openingThis ? "Opening…" : "Edit diff"}</button>}
    </h3>
    {patch && <Annotations
      notes={placed.outdated} draft={draft} showNewDraft={false} outdated actions={actions}
      className="sticky left-0 w-[min(44rem,100cqw)] pl-3"
    />}
    {lines ? <PatchView path={path} lines={lines} split={split} placed={placed} draft={draft} actions={actions} /> : <div className="px-3 py-6 text-xs text-ink-faint">
      {!patch ? "Loading diff" : patch.failed ? "Could not load diff" : "No text diff for this file"}
    </div>}
  </section>
})
