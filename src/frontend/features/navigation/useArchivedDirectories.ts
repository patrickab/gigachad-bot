"use client"

import { useMemo, useSyncExternalStore } from "react"

const STORAGE_KEY = "archived_code_directories"
const CHANGE_EVENT = "archived-code-directories"

function subscribe(callback: () => void) {
  window.addEventListener(CHANGE_EVENT, callback)
  window.addEventListener("storage", callback)
  return () => {
    window.removeEventListener(CHANGE_EVENT, callback)
    window.removeEventListener("storage", callback)
  }
}

function parse(stored: string): string[] {
  try {
    const paths: unknown = JSON.parse(stored)
    return Array.isArray(paths) ? paths.filter((path) => typeof path === "string") : []
  } catch { return [] }
}

/** Directories of the Code | Filemounts section archived on this device, shared across tabs. */
export function useArchivedDirectories() {
  const stored = useSyncExternalStore(subscribe, () => localStorage.getItem(STORAGE_KEY) ?? "[]", () => "[]")
  const archived = useMemo(() => new Set(parse(stored)), [stored])

  const setArchived = (path: string, value: boolean) => {
    const next = new Set(parse(localStorage.getItem(STORAGE_KEY) ?? "[]"))
    if (value) next.add(path)
    else next.delete(path)
    localStorage.setItem(STORAGE_KEY, JSON.stringify([...next]))
    window.dispatchEvent(new Event(CHANGE_EVENT))
  }

  return { archived, setArchived }
}
