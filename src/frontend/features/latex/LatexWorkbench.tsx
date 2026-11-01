"use client"

import { useEffect } from "react"
import { Workbench, type WorkbenchProps } from "@/features/workbench/Workbench"
import { LatexBuildStatus } from "./LatexBuildStatus"
import { LatexPdfPane } from "./LatexPdfPane"
import { useLatexBuild } from "./useLatexBuild"

type LatexWorkbenchProps = Omit<WorkbenchProps, "renderLabel" | "renderPane" | "status" | "onSaved" | "onRunEnd" | "quoteLanguage">

/**
 * One `.tex` file in the workbench: Neovim, its built PDF and the omp chat drawer.
 *
 * - The PDF builds once on open, after every `:w` while omp is idle, and once when an omp run ends
 * - While omp edits, the PDF blurs until the run's build arrives
 * - The build status sits in the top bar's `workbenchStatus` slot
 */
export function LatexWorkbench(props: LatexWorkbenchProps) {
  const build = useLatexBuild(props.path)
  const { trigger } = build
  useEffect(() => { void trigger() }, [trigger])

  const pdfFailure = build.failure?.message ?? (build.result && !build.result.ok ? "Build failed" : null)
  return (
    <Workbench
      {...props}
      renderLabel="PDF"
      renderPane={({ layout, editing }) => (
        <LatexPdfPane
          url={build.pdfUrl}
          documentKey={build.pdfPath}
          building={build.building}
          editing={editing}
          emptyMessage={pdfFailure}
          pageBox={layout}
        />
      )}
      status={<LatexBuildStatus build={build} />}
      onSaved={(_text, agentRunning) => { if (!agentRunning) void trigger() }}
      onRunEnd={trigger}
      quoteLanguage="latex"
    />
  )
}
