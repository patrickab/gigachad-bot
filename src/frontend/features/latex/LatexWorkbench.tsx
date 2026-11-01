"use client"

import { useEffect } from "react"
import { Workbench, type WorkbenchProps } from "@/features/workbench/Workbench"
import { LatexBuildStatus } from "./LatexBuildStatus"
import { LatexPdfPane } from "./LatexPdfPane"
import { useLatexBuild } from "./useLatexBuild"

type LatexWorkbenchProps = Omit<WorkbenchProps, "renderLabel" | "renderPane" | "status" | "onSaved" | "onPulled" | "onRunEnd" | "quoteLanguage">

/**
 * One `.tex` file in the workbench: Neovim, its built PDF and the omp chat drawer.
 *
 * - The PDF builds on open, after every `:w`, on every agent edit or remote save as it arrives, and once more when an omp run ends
 * - Builds go through one queue, so they never overlap and a burst of edits collapses into one follow-up build
 * - A failed build, e.g. of a half-written file, keeps the last good PDF
 * - The build status sits in the top bar's `workbenchStatus` slot
 */
export function LatexWorkbench(props: LatexWorkbenchProps) {
  const build = useLatexBuild(props.path)
  const { trigger } = build
  useEffect(() => { void trigger() }, [trigger])
  const rebuild = () => { void trigger() }

  const pdfFailure = build.failure?.message ?? (build.result && !build.result.ok ? "Build failed" : null)
  return (
    <Workbench
      {...props}
      renderLabel="PDF"
      renderPane={({ layout }) => (
        <LatexPdfPane
          url={build.pdfUrl}
          documentKey={build.pdfPath}
          building={build.building}
          emptyMessage={pdfFailure}
          pageBox={layout}
        />
      )}
      status={<LatexBuildStatus build={build} />}
      onSaved={rebuild}
      onPulled={rebuild}
      onRunEnd={rebuild}
      quoteLanguage="latex"
    />
  )
}
