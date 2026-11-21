"use client"

import { useCallback, useState, type SetStateAction } from "react"
import { useStoredString } from "@/hooks/useStoredString"
import { readStorage, removeStorage } from "@/lib/storage"
import { clearChips } from "./composerChips"

const DRAFT_STORAGE_PREFIX = "gigachad:draft:"

/** The draft key of a chat that has no history file yet, by its tab: what is typed into a new chat survives a reload like any other draft. */
export function newChatDraftKey(tabId: string): string {
  return `tab:${tabId}`
}

/** Drop the stored draft and chips of `draftKey`, e.g. a new chat's once its tab is closed. */
export function forgetComposerDraft(draftKey: string): void {
  removeStorage(`${DRAFT_STORAGE_PREFIX}${draftKey}`)
  clearChips(draftKey)
}

/**
 * The composer's unsent text, kept in `localStorage` under `draftKey` so it survives closing the tab or reloading.
 *
 * - `draftKey` names the conversation (a chat's history file, a Pi session), null keeps the text in memory only
 * - An empty draft removes its key
 */
export function useComposerDraft(draftKey: string | null): [string, (update: SetStateAction<string>) => void] {
  const storageKey = draftKey === null ? null : `${DRAFT_STORAGE_PREFIX}${draftKey}`
  const [stored, setStored] = useStoredString(storageKey)
  const [inMemory, setInMemory] = useState("")

  const setText = useCallback((update: SetStateAction<string>) => {
    if (storageKey === null) { setInMemory(update); return }
    const next = typeof update === "function" ? update(readStorage(storageKey) ?? "") : update
    setStored(next === "" ? null : next)
  }, [storageKey, setStored])

  return [storageKey === null ? inMemory : stored ?? "", setText]
}
