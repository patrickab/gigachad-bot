// The factory's card (contract `cards`, version 1): the readable form of a factory notice or `factory` tool result,
// computed by the factory and drawn as it is. Unknown fields are dropped, malformed entries skipped.

import { isRecord } from "@/lib/utils"

export const CARD_KINDS = ["wave", "task", "scout", "proof", "contract", "decision", "action"] as const
export const SEVERITIES = ["blocker", "major", "minor"] as const

export type FactoryCardKind = (typeof CARD_KINDS)[number]
export type FindingSeverity = (typeof SEVERITIES)[number]

export interface FactoryCardRow { label: string; text: string }
export interface FactoryCardFinding { severity: FindingSeverity; file?: string; line?: number; text: string }

export interface FactoryCard {
  v: 1
  kind: FactoryCardKind
  /** attention: it needs a ruling (a question, rejected, blocked, can never start, failing). */
  tone: "attention" | "quiet"
  title: string
  tag?: string
  id?: string
  state?: string
  /** Cents. */
  cost?: number
  progress?: { done: number; total: number }
  rows?: FactoryCardRow[]
  findings?: FactoryCardFinding[]
  /** Markdown. */
  text?: string
  /** The notice key of the card this one answers (a decision). */
  answers?: string
}

const isString = (value: unknown): value is string => typeof value === "string"
const isNumber = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value)
const includes = <T extends string>(values: readonly T[], value: unknown): value is T => values.includes(value as T)

function row(value: unknown): FactoryCardRow | null {
  return isRecord(value) && isString(value.label) && isString(value.text) ? { label: value.label, text: value.text } : null
}

function finding(value: unknown): FactoryCardFinding | null {
  if (!isRecord(value) || !includes(SEVERITIES, value.severity) || !isString(value.text)) return null
  return {
    severity: value.severity,
    text: value.text,
    ...(isString(value.file) && { file: value.file }),
    ...(isNumber(value.line) && { line: value.line }),
  }
}

function list<T>(value: unknown, item: (entry: unknown) => T | null): T[] | undefined {
  if (!Array.isArray(value)) return undefined
  return value.map(item).filter((entry): entry is T => entry !== null)
}

/** The card in `details.card`, or null when there is none of version 1. */
export function factoryCard(details: unknown): FactoryCard | null {
  const card = isRecord(details) ? details.card : undefined
  if (!isRecord(card) || card.v !== 1 || !includes(CARD_KINDS, card.kind) || !isString(card.title)) return null
  const progress = card.progress
  return {
    v: 1,
    kind: card.kind,
    tone: card.tone === "attention" ? "attention" : "quiet",
    title: card.title,
    ...(isString(card.tag) && { tag: card.tag }),
    ...(isString(card.id) && { id: card.id }),
    ...(isString(card.state) && { state: card.state }),
    ...(isNumber(card.cost) && { cost: card.cost }),
    ...(isRecord(progress) && isNumber(progress.done) && isNumber(progress.total) && { progress: { done: progress.done, total: progress.total } }),
    ...(Array.isArray(card.rows) && { rows: list(card.rows, row) }),
    ...(Array.isArray(card.findings) && { findings: list(card.findings, finding) }),
    ...(isString(card.text) && { text: card.text }),
    ...(isString(card.answers) && { answers: card.answers }),
  }
}
