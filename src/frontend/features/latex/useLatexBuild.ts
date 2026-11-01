"use client"

import { useCallback, useMemo, useState } from "react"
import { useFollowFiles } from "@/hooks/useFollowFiles"
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
  /** The PDF's path, the same across builds of one root. */
  pdfPath: string | null
}

interface InternalState extends Omit<LatexBuildState, "pdfUrl"> {
  revision: number
  /** The root this state belongs to. */
  root: string | null
}

const initialState = (root: string | null): InternalState => ({ building: false, result: null, failure: null, pdfPath: null, revision: 0, root })

/**
 * Build the project's `root` `.tex` on demand, and again whenever a file in its `folder` changes.
 *
 * - `trigger()` resolves when a build started after the call has finished, builds never overlap and extra triggers collapse into one follow-up
 * - Without a `root` nothing builds and nothing is followed
 * - A failed build keeps the last good PDF, an error reply is a result with `ok: false`, a failed request a `failure`
 * - Another `root` starts clean: the old PDF is gone and an answer of an old build is dropped
 * - A change of any file in the `folder` (an open buffer, a `.bib`, an image, an edit of the agent, a shell or another device) triggers a build, build output does not
 */
export function useLatexBuild(root: string | null, { folder }: { folder: string }): LatexBuildState & { trigger: () => Promise<void> } {
  const [stored, setState] = useState(() => initialState(root))
  const state = stored.root === root ? stored : initialState(root)
  if (stored !== state) setState(state)

  const run = useCallback(async () => {
    if (root === null) return
    setState((current) => current.root === root ? { ...current, building: true } : current)
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
  }, [root])
  const trigger = useMemo(() => createBuildQueue(run), [run])
  useFollowFiles([], () => { void trigger() }, root === null ? [] : [folder])

  const pdfUrl = state.pdfPath ? pdfUrlOf(state.pdfPath, state.revision) : null
  return { building: state.building, result: state.result, failure: state.failure, pdfPath: state.pdfPath, pdfUrl, trigger }
}
