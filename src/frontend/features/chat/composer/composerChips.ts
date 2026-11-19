"use client"

import { useCallback, useEffect, useId, useSyncExternalStore } from "react"
import { readStorage, removeStorage, writeStorage } from "@/lib/storage"

/**
 * Composer chips: labeled text staged above a composer's draft and appended verbatim to the message on send.
 *
 * Consumes a conversation key, the same `draftKey` `useComposerDraft` keeps the draft under (a chat's history file,
 * a Pi session). Provides a module store any feature calls without a handle on the composer:
 * - `stageChip(conversationKey, {label, text})` stages a chip, e.g. a file selection, a diff hunk or test output
 * - `useComposerChips(conversationKey)` is what `ChatInput` renders, removes and sends
 * Chips of a keyed conversation persist in `localStorage` under `gigachad:chips:<key>` like drafts. A composer
 * without a key keeps its chips in memory and drops them when it unmounts.
 */

export interface ComposerChip {
  id: string
  label: string
  text: string
}

const CHIP_STORAGE_PREFIX = "gigachad:chips:"
/** Prefix of the keys `useComposerChips` invents for keyless composers, never persisted. */
const IN_MEMORY_PREFIX = "\0memory:"
/** A paste longer than this many characters or lines becomes a chip instead of filling the draft. */
export const LARGE_PASTE_CHARS = 2000
export const LARGE_PASTE_LINES = 30

const NO_CHIPS: readonly ComposerChip[] = []
const chipsByKey = new Map<string, readonly ComposerChip[]>()
const listeners = new Set<() => void>()

function persisted(key: string): boolean {
  return !key.startsWith(IN_MEMORY_PREFIX)
}

function chipsOf(key: string): readonly ComposerChip[] {
  const cached = chipsByKey.get(key)
  if (cached) return cached
  const stored = persisted(key) ? parseChips(readStorage(`${CHIP_STORAGE_PREFIX}${key}`)) : NO_CHIPS
  chipsByKey.set(key, stored)
  return stored
}

function parseChips(raw: string | null): readonly ComposerChip[] {
  if (raw === null) return NO_CHIPS
  try {
    const parsed: unknown = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed.filter(isChip) : NO_CHIPS
  } catch {
    return NO_CHIPS
  }
}

function isChip(value: unknown): value is ComposerChip {
  const chip = value as Partial<ComposerChip> | null
  return typeof chip?.id === "string" && typeof chip.label === "string" && typeof chip.text === "string"
}

function setChips(key: string, chips: readonly ComposerChip[]): void {
  const next = chips.length === 0 ? NO_CHIPS : chips
  chipsByKey.set(key, next)
  if (persisted(key)) {
    const storageKey = `${CHIP_STORAGE_PREFIX}${key}`
    if (next.length === 0) removeStorage(storageKey)
    else writeStorage(storageKey, JSON.stringify(next))
  }
  for (const listener of listeners) listener()
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener)
  return () => { listeners.delete(listener) }
}

/** Stage `chip` above the composer of `conversationKey`, after the chips already there. Returns the new chip's id. */
export function stageChip(conversationKey: string, chip: { label: string; text: string }): string {
  const id = crypto.randomUUID()
  setChips(conversationKey, [...chipsOf(conversationKey), { id, label: chip.label, text: chip.text }])
  return id
}

/** Drop the chip `id` from `conversationKey`. */
export function removeChip(conversationKey: string, id: string): void {
  setChips(conversationKey, chipsOf(conversationKey).filter((chip) => chip.id !== id))
}

/** Drop every chip of `conversationKey`, e.g. once they are sent. */
export function clearChips(conversationKey: string): void {
  if (chipsOf(conversationKey).length > 0) setChips(conversationKey, NO_CHIPS)
}

/** `draft` followed by each chip's text verbatim, blank-line separated, empty parts skipped. */
export function withChips(draft: string, chips: readonly ComposerChip[]): string {
  return [draft, ...chips.map((chip) => chip.text)].filter((part) => part.length > 0).join("\n\n")
}

/** The chip a paste of `text` becomes, null when it is small enough for the draft. A trailing newline is not a line. */
export function largePasteChip(text: string): { label: string; text: string } | null {
  const lines = text.replace(/\r?\n$/, "").split(/\r?\n/).length
  if (text.length <= LARGE_PASTE_CHARS && lines <= LARGE_PASTE_LINES) return null
  return { label: `pasted text · ${lines} ${lines === 1 ? "line" : "lines"}`, text }
}

/** The chips of `conversationKey` and the key they live under; a null key gets a private in-memory one. */
export function useComposerChips(conversationKey: string | null) {
  const instanceId = useId()
  const key = conversationKey ?? `${IN_MEMORY_PREFIX}${instanceId}`
  const chips = useSyncExternalStore(subscribe, () => chipsOf(key), () => NO_CHIPS)

  useEffect(() => {
    if (persisted(key)) return
    return () => { chipsByKey.delete(key) }
  }, [key])

  return {
    chips,
    stage: useCallback((chip: { label: string; text: string }) => stageChip(key, chip), [key]),
    remove: useCallback((id: string) => removeChip(key, id), [key]),
    clear: useCallback(() => clearChips(key), [key]),
  }
}
