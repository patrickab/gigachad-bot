"use client"

import { useMemo } from "react"
import { useStoredString } from "@/hooks/useStoredString"
import { readStorage } from "@/lib/storage"

const STORAGE_KEY = "archived_code_directories"

function parse(stored: string): string[] {
  try {
    const paths: unknown = JSON.parse(stored)
    return Array.isArray(paths) ? paths.filter((path) => typeof path === "string") : []
  } catch { return [] }
}

/** Directories of the Code section archived on this device, shared across tabs. */
export function useArchivedDirectories() {
  const [stored, store] = useStoredString(STORAGE_KEY)
  const archived = useMemo(() => new Set(parse(stored ?? "[]")), [stored])

  const setArchived = (path: string, value: boolean) => {
    const next = new Set(parse(readStorage(STORAGE_KEY) ?? "[]"))
    if (value) next.add(path)
    else next.delete(path)
    store(JSON.stringify([...next]))
  }

  return { archived, setArchived }
}
