"use client"

import { useMemo } from "react"
import { useStoredJson } from "@/hooks/useStoredJson"

const STORAGE_KEY = "archived_code_directories"

const parsePaths = (raw: unknown): string[] => Array.isArray(raw) ? raw.filter((path): path is string => typeof path === "string") : []

/** Directories of the Code section archived on this device, shared across tabs. */
export function useArchivedDirectories() {
  const [paths, update] = useStoredJson(STORAGE_KEY, parsePaths)
  const archived = useMemo(() => new Set(paths), [paths])

  const setArchived = (path: string, value: boolean) => update((latest) =>
    value ? [...new Set([...latest, path])] : latest.filter((entry) => entry !== path))

  return { archived, setArchived }
}
