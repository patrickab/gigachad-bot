"use client"

import { useCallback, useState } from "react"

/** Report what went wrong in `source`, null once its next attempt worked. */
export type ReportNotice = (source: string, notice: string | null) => void

/** A workbench's one notice: why the latest thing beside the editor did not work. */
export interface WorkbenchNoticeChannel {
  /** The shown notice, null for none. */
  notice: string | null
  report: ReportNotice
}

/**
 * The notice channel of one workbench, shown in its top bar's status.
 *
 * - The latest reported notice wins, whatever its source
 * - A success of the source the shown notice came from clears it, a success of another source leaves it
 */
export function useWorkbenchNotice(): WorkbenchNoticeChannel {
  const [shown, setShown] = useState<{ source: string; notice: string } | null>(null)
  const report = useCallback<ReportNotice>((source, notice) => {
    setShown((current) => notice !== null ? { source, notice } : current?.source === source ? null : current)
  }, [])
  return { notice: shown?.notice ?? null, report }
}
