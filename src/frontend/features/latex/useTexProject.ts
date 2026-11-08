"use client"

import { useCallback, useEffect, useState } from "react"
import { useLatestRequest } from "@/hooks/useLatestRequest"
import { chooseTexRoot, loadTexProject } from "@/lib/backend/api"
import type { TexProject } from "@/lib/domain/latex"
import { errorMessage } from "@/lib/utils"

/**
 * The LaTeX project of the file at `path`: its folder, its root and, without a root, the files that could be one.
 *
 * - Another `path` or `reload()` asks the backend again, the project it had stays until the answer comes, so a buffer switch never blanks the workbench
 * - `error` says why the latest read failed, the owner shows it only while there is no project yet
 * - `choose(root)` writes the root into the file at `path` and takes the project the backend answers, a failure throws
 */
export function useTexProject(path: string) {
  const [project, setProject] = useState<TexProject | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [attempt, setAttempt] = useState(0)
  // Only the newest request may set the project: a read in flight when the user chooses a root must not undo the choice
  const { run } = useLatestRequest()

  useEffect(() => {
    void run(
      loadTexProject(path),
      (loaded) => {
        setProject(loaded)
        setError(null)
      },
      (cause) => setError(errorMessage(cause, "Could not read this LaTeX project")),
    )
  }, [path, attempt, run])

  const reload = useCallback(() => setAttempt((count) => count + 1), [])

  const choose = useCallback(async (root: string) => {
    await run(chooseTexRoot(path, root), setProject)
  }, [path, run])

  return { project, error, reload, choose }
}
