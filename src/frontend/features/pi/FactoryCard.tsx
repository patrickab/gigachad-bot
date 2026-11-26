"use client"

import { Factory } from "lucide-react"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"
import { Disclosure } from "@/components/ui/Disclosure"
import { formatCost } from "./subagents"
import { SEVERITIES, type FactoryCard as Card, type FactoryCardFinding, type FindingSeverity } from "./factoryCard"

interface FactoryCardProps {
  card: Card
  /** Opens a finding's file, at its line when it names one. */
  onOpenFile?: (path: string, line?: number) => void
  className?: string
}

const countLabel = (severity: FindingSeverity, count: number) => `${count} ${severity === "blocker" && count !== 1 ? "blockers" : severity}`

/** The title row's facts at its right: tag, state, `done/total`, cost. */
function meta(card: Card): string {
  return [
    card.tag,
    card.state,
    card.progress && `${card.progress.done}/${card.progress.total}`,
    card.cost !== undefined && formatCost(card.cost / 100),
  ].filter(Boolean).join(" · ")
}

function FindingItem({ finding, onOpenFile }: { finding: FactoryCardFinding; onOpenFile?: FactoryCardProps["onOpenFile"] }) {
  const { file, line } = finding
  const place = file && (line !== undefined ? `${file}:${line}` : file)
  return (
    <li className="flex min-w-0 gap-2">
      {place && (onOpenFile ? (
        <button
          type="button"
          onClick={() => onOpenFile(file, line)}
          className="shrink-0 rounded font-mono text-ink-muted underline-offset-2 hover:text-ink hover:underline focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"
        >
          {place}
        </button>
      ) : <span className="shrink-0 font-mono text-ink-muted">{place}</span>)}
      <span className="min-w-0 break-words">{finding.text}</span>
    </li>
  )
}

/** The findings of one severity behind a collapsed count. */
function FindingGroup({ severity, findings, onOpenFile }: { severity: FindingSeverity; findings: FactoryCardFinding[]; onOpenFile?: FactoryCardProps["onOpenFile"] }) {
  return (
    <Disclosure summary={<span>{countLabel(severity, findings.length)}</span>} summaryClassName={severity === "blocker" ? "text-warning" : undefined}>
      <ul aria-label={`${severity} findings`} className="flex flex-col gap-0.5 pb-1 pl-5 text-xs text-ink-faint">
        {findings.map((finding, index) => <FindingItem key={index} finding={finding} onOpenFile={onOpenFile} />)}
      </ul>
    </Disclosure>
  )
}

/**
 * A factory card (contract `cards`) drawn as the factory computed it: the factory icon, the title (attention in the
 * warning ink, quiet muted), tag, state, progress and cost at the right; opened, its rows, its findings grouped by
 * severity and collapsed to counts, its text as markdown. An attention card starts open. Nothing is derived here.
 */
export function FactoryCard({ card, onOpenFile, className = "my-1" }: FactoryCardProps) {
  const facts = meta(card)
  const groups = SEVERITIES
    .map((severity) => ({ severity, findings: (card.findings ?? []).filter((finding) => finding.severity === severity) }))
    .filter((group) => group.findings.length > 0)
  const rows = card.rows ?? []
  const text = card.text?.trim()
  const hasBody = rows.length > 0 || groups.length > 0 || Boolean(text)
  const summary = (
    <>
      <span className="min-w-0 flex-1 truncate">{card.title}</span>
      {facts && <span className="shrink-0 tabular-nums text-ink-faint">{facts}</span>}
    </>
  )
  const ink = card.tone === "attention" ? "text-warning" : undefined
  if (!hasBody) return <Disclosure className={className} icon={Factory} summary={summary} summaryClassName={ink} />
  return (
    <Disclosure className={className} icon={Factory} summary={summary} summaryClassName={ink} defaultOpen={card.tone === "attention"}>
      <div className="flex flex-col gap-1 pb-1 pl-5 text-xs">
        {rows.length > 0 && (
          <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-0.5">
            {rows.map((row, index) => (
              <div key={index} className="contents">
                <dt className="text-ink-faint">{row.label}</dt>
                <dd className="break-words text-ink-muted">{row.text}</dd>
              </div>
            ))}
          </dl>
        )}
        {groups.length > 0 && (
          <div>
            {groups.map((group) => <FindingGroup key={group.severity} {...group} onOpenFile={onOpenFile} />)}
          </div>
        )}
        {text && <LaTeXMarkdown content={text} compact />}
      </div>
    </Disclosure>
  )
}
