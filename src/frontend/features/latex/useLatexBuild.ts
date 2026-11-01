"use client"

import { useCallback, useMemo, useState } from "react"
import { ApiError, buildLatex } from "@/lib/backend/api"
import type { LatexBuild } from "@/lib/domain/latex"
import { errorMessage } from "@/lib/utils"
import { createBuildQueue, pdfUrlOf } from "./latexBuild"

const SERVICE_UNAVAILABLE = 503

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
  /** The PDF's path, the same across builds of one file. */
  pdfPath: string | null
}

interface InternalState extends Omit<LatexBuildState, "pdfUrl"> {
  revision: number
}

const INITIAL: InternalState = { building: false, result: null, failure: null, pdfPath: null, revision: 0 }

/**
 * Build the `.tex` file at `path` on demand.
 *
 * - `trigger()` resolves when a build started after the call has finished, builds never overlap and extra triggers collapse into one follow-up
 * - A failed build keeps the last good PDF, an error reply is a result with `ok: false`, a failed request a `failure`
 */
export function useLatexBuild(path: string): LatexBuildState & { trigger: () => Promise<void> } {
  const [state, setState] = useState(INITIAL)

  const run = useCallback(async () => {
    setState((current) => ({ ...current, building: true }))
    try {
      const result = await buildLatex(path)
      setState((current) => ({
        building: false,
        result,
        failure: null,
        pdfPath: result.pdf_path ?? current.pdfPath,
        revision: result.ok || (result.pdf_path !== null && result.pdf_path !== current.pdfPath) ? current.revision + 1 : current.revision,
      }))
    } catch (error) {
      const notInstalled = error instanceof ApiError && error.status === SERVICE_UNAVAILABLE
      setState((current) => ({ ...current, building: false, failure: { message: errorMessage(error, "Build failed"), notInstalled } }))
    }
  }, [path])
  const trigger = useMemo(() => createBuildQueue(run), [run])

  const pdfUrl = state.pdfPath ? pdfUrlOf(state.pdfPath, state.revision) : null
  return { building: state.building, result: state.result, failure: state.failure, pdfPath: state.pdfPath, pdfUrl, trigger }
}
