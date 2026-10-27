"use client"

import { useCallback, useSyncExternalStore } from "react"
import { readStorage, writeStorage } from "@/lib/storage"

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

/** Store `value` under `key` and notify every `useStoredString` reader in this tab. */
function writeStoredString(key: string, value: string): void {
  writeStorage(key, value)
  window.dispatchEvent(new Event(STORED_STRING_EVENT))
}

/** The `localStorage` string under `key` (null when missing or `key` is null), kept in sync across components and tabs. */
export function useStoredString(key: string | null): [string | null, (next: string) => void] {
  const value = useSyncExternalStore(subscribe, () => (key === null ? null : readStorage(key)), () => null)
  const setValue = useCallback((next: string) => {
    if (key !== null) writeStoredString(key, next)
  }, [key])
  return [value, setValue]
}
