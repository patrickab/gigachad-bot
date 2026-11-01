"use client"

import { useCallback, useMemo, useRef, useState } from "react"
import { usePolling } from "@/hooks/usePolling"
import { useTabActive } from "@/contexts/TabContext"
import { ApiError, buildLatex, folderStamp } from "@/lib/backend/api"
import type { LatexBuild } from "@/lib/domain/latex"
import { errorMessage } from "@/lib/utils"
import { createBuildQueue, pdfUrlOf } from "./latexBuild"

const SERVICE_UNAVAILABLE = 503
/** Time between looks at the project folder while the agent runs, so its edits show in the PDF as they land. */
const STREAMING_STAMP_MS = 500
/** Time between looks at the project folder otherwise, files changed by other tools or devices. */
const IDLE_STAMP_MS = 3000

/** Why no build could run at all: `notInstalled` when the backend has no `latexmk`. */
export interface LatexFailure {
  message: string
  notInstalled: boolean
}

export interface LatexBuildState {
  building: boolean
  /** The last build that ran, null before the first. */
  result: LatexBuild | null
  failure: LatexFailure | null
  /** Raw URL of the latest good PDF, a new one with every successful build. Null while there is none. */
  pdfUrl: string | null
  /** The PDF's path, the same across builds of one root. */
  pdfPath: string | null
}

interface InternalState extends Omit<LatexBuildState, "pdfUrl"> {
  revision: number
  /** The root this state belongs to. */
  root: string | null
}

const initialState = (root: string | null): InternalState => ({ building: false, result: null, failure: null, pdfPath: null, revision: 0, root })

/** The project folder's stamp when the last build started: any file changed since then makes the next stamp differ. */
interface BuiltStamp {
  folder: string
  stamp: string | null
}

/**
 * Build the project's `root` `.tex` on demand, and again whenever a file in its `folder` changes.
 *
 * - `trigger()` resolves when a build started after the call has finished, builds never overlap and extra triggers collapse into one follow-up
 * - Without a `root` nothing builds
 * - A failed build keeps the last good PDF, an error reply is a result with `ok: false`, a failed request a `failure`
 * - Another `root` starts clean: the old PDF is gone and an answer of an old build is dropped
 * - The folder's stamp is read every 500 ms while `streaming`, every 3 s otherwise (while the page is visible), a stamp other than the one of the last build's start triggers a build.
 *   Build output does not change the stamp
 */
export function useLatexBuild(root: string | null, { folder, streaming }: { folder: string; streaming: boolean }): LatexBuildState & { trigger: () => Promise<void> } {
  const [stored, setState] = useState(() => initialState(root))
  const state = stored.root === root ? stored : initialState(root)
  if (stored !== state) setState(state)
  const built = useRef<BuiltStamp>({ folder, stamp: null })

  const run = useCallback(async () => {
    if (root === null) return
    setState((current) => current.root === root ? { ...current, building: true } : current)
    // Read before the build, so a file saved while it runs differs from this stamp and builds again
    built.current = { folder, stamp: await folderStamp(folder).catch(() => null) }
    try {
      const result = await buildLatex(root)
      setState((current) => current.root !== root ? current : {
        root,
        building: false,
        result,
        failure: null,
        pdfPath: result.pdf_path ?? current.pdfPath,
        revision: result.ok || (result.pdf_path !== null && result.pdf_path !== current.pdfPath) ? current.revision + 1 : current.revision,
      })
    } catch (error) {
      const notInstalled = error instanceof ApiError && error.status === SERVICE_UNAVAILABLE
      setState((current) => current.root !== root ? current : { ...current, building: false, failure: { message: errorMessage(error, "Build failed"), notInstalled } })
    }
  }, [root, folder])
  const trigger = useMemo(() => createBuildQueue(run), [run])
  const tabActive = useTabActive()

  usePolling(async () => {
    let stamp: string
    try {
      stamp = await folderStamp(folder)
    } catch {
      // The next look retries
      return
    }
    const known = built.current.folder === folder ? built.current.stamp : null
    built.current = { folder, stamp }
    if (known !== null && stamp !== known) void trigger()
  }, { intervalMs: streaming ? STREAMING_STAMP_MS : IDLE_STAMP_MS, active: root !== null && tabActive })

  const pdfUrl = state.pdfPath ? pdfUrlOf(state.pdfPath, state.revision) : null
  return { building: state.building, result: state.result, failure: state.failure, pdfPath: state.pdfPath, pdfUrl, trigger }
}
