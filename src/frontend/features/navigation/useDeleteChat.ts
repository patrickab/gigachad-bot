"use client"

import { useCallback } from "react"
import { useBranches } from "@/features/navigation/BranchContext"
import { useUndoDelete } from "@/contexts/UndoDeleteContext"
import { baseName } from "@/lib/paths"

/** Schedules an undoable delete of a chat together with its branches. */
export function useDeleteChat() {
  const { cascadeDelete } = useBranches()
  const { schedule } = useUndoDelete()
  return useCallback((file: string) => {
    schedule(`history:${file}`, baseName(file), async () => { await cascadeDelete(file) })
  }, [cascadeDelete, schedule])
}
