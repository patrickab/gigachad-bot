// Ported from kahme247/ompweb components/MessageView-diff-view.tsx (MIT, Copyright (c) 2026 agegr).
"use client"

import { useMemo } from "react"
import { isRecord } from "@/lib/utils"
import { parsePiDisplayDiff, parseUnifiedPatch, type SplitDiffCell } from "./patch"
import "./tools.css"

/** Diff text a tool result carries: `details.patch` (unified) first, else `details.diff` (unified or Pi's display diff). */
export function getResultDiff(details: unknown): string | null {
  if (!isRecord(details)) return null
  if (typeof details.patch === "string" && details.patch) return details.patch
  if (typeof details.diff === "string" && details.diff) return details.diff
  return null
}

const MARKERS: Record<SplitDiffCell["type"], string> = { added: "+", removed: "-", context: " ", empty: " " }

/** Side-by-side view of a diff, falling back to coloured patch lines when the text does not parse. */
export function PairedDiffResult({ text }: { text: string }) {
  const files = useMemo(() => parseUnifiedPatch(text) ?? parsePiDisplayDiff(text), [text])
  if (!files) return <PatchTextView text={text} />
  const showFileHeaders = files.length > 1

  return (
    <div className="pi-diff">
      {files.map((file, fileIndex) => (
        <div key={fileIndex} className="pi-diff-file">
          {showFileHeaders && (
            <div className="pi-diff-grid pi-diff-file-header">
              <div className="pi-diff-file-title">{file.oldPath || "Before"}</div>
              <div className="pi-diff-file-title">{file.newPath || "After"}</div>
            </div>
          )}
          <div className="pi-diff-grid">
            {file.rows.map((row, rowIndex) => row.type === "hunk" ? null : (
              <div key={rowIndex} className="pi-diff-row">
                <SplitDiffCellView cell={row.left} />
                <SplitDiffCellView cell={row.right} />
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}

/** One side of a split diff row: line number, +/- marker, text. */
function SplitDiffCellView({ cell }: { cell: SplitDiffCell }) {
  return (
    <div className="pi-diff-cell" data-type={cell.type}>
      <span className="pi-diff-lineno">{cell.lineNo ?? ""}</span>
      <span className="pi-diff-marker">{MARKERS[cell.type]}</span>
      <span className="pi-diff-text">{cell.text || "\u00a0"}</span>
    </div>
  )
}

/** Raw diff text with added, removed and hunk lines coloured. */
function PatchTextView({ text }: { text: string }) {
  return (
    <div className="pi-diff">
      {text.split(/\r?\n/).map((line, index) => {
        const type = line.startsWith("@@") ? "hunk"
          : line.startsWith("+") && !line.startsWith("+++") ? "added"
          : line.startsWith("-") && !line.startsWith("---") ? "removed"
          : "context"
        return (
          <div key={index} className="pi-patch-line" data-type={type}>
            <span className="pi-diff-lineno">{index + 1}</span>
            <span className="pi-diff-text">{line || "\u00a0"}</span>
          </div>
        )
      })}
    </div>
  )
}

/** Plain tool output text, red when the tool failed. */
export function PairedResult({ text, isEmpty, isError }: { text: string; isEmpty: boolean; isError: boolean }) {
  return (
    <div className="pi-tool-call-output" data-error={isError || undefined}>
      <pre className="pi-tool-call-output-text">{isEmpty ? "(no output)" : text}</pre>
    </div>
  )
}
