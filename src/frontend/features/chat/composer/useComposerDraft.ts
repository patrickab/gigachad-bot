"use client"

import { useCallback, useState, type SetStateAction } from "react"
import { useStoredString } from "@/hooks/useStoredString"
import { readStorage } from "@/lib/storage"

const DRAFT_STORAGE_PREFIX = "gigachad:draft:"

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
