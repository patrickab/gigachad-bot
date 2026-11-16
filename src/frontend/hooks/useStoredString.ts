"use client"

import { useCallback, useSyncExternalStore } from "react"
import { readStorage, removeStorage, writeStorage } from "@/lib/storage"

/** Window event fired after every `writeStoredString`, so same-tab readers refresh. */
const STORED_STRING_EVENT = "gigachad:stored-string"

function subscribe(callback: () => void) {
  window.addEventListener(STORED_STRING_EVENT, callback)
  window.addEventListener("storage", callback)
  return () => {
    window.removeEventListener(STORED_STRING_EVENT, callback)
    window.removeEventListener("storage", callback)
  }
}

/** Store `value` under `key` (null removes it) and notify every `useStoredString` reader in this tab. */
function writeStoredString(key: string, value: string | null): void {
  if (value === null) removeStorage(key)
  else writeStorage(key, value)
  window.dispatchEvent(new Event(STORED_STRING_EVENT))
}

/** The `localStorage` string under `key` (null when missing or `key` is null), kept in sync across components and tabs. Setting null removes it. */
export function useStoredString(key: string | null): [string | null, (next: string | null) => void] {
  const value = useSyncExternalStore(subscribe, () => (key === null ? null : readStorage(key)), () => null)
  const setValue = useCallback((next: string | null) => {
    if (key !== null) writeStoredString(key, next)
  }, [key])
  return [value, setValue]
}
