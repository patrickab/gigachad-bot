"use client"

import { useEffect, useEffectEvent } from "react"
import { LoadFailure, LoadingPlaceholder } from "@/components/ui/DocumentLoadState"
import { Workbench, type WorkbenchProps } from "@/features/workbench/Workbench"
import type { TexProject } from "@/lib/domain/latex"
import { LatexBuildStatus } from "./LatexBuildStatus"
import { LatexPdfPane } from "./LatexPdfPane"
import { LatexRootPicker } from "./LatexRootPicker"
import { useLatexBuild } from "./useLatexBuild"
import { useTexProject } from "./useTexProject"

type ProjectWorkbenchProps = Omit<WorkbenchProps, "project" | "renderPane" | "status" | "onSaved" | "quoteLanguage">

type LatexWorkbenchProps = ProjectWorkbenchProps & {
  /** The project folder the tab's item remembers, undefined before the first one. */
  texFolder: string | undefined
  /** The shown file's project folder, for the tab's item to remember. */
  onTexFolder: (folder: string) => void
}

/** The PDF pane's note while the project has no root and the file shows no candidates. */
const NO_ROOT = "No root document"

/**
 * The LaTeX project of the shown file in the workbench: Neovim over its files, the built PDF and the omp chat drawer.
 *
 * - The project (folder, root) is read from the shown file, a new `path` under the same workbench is a buffer switch and reads it again
 * - The chat belongs to the project folder, so every file of the project shares it
 * - The root builds, not the shown file: on open, and whenever any file in the folder changes, whoever wrote it (a `:w`, the agent, another device, a `.bib`, an image)
 * - Builds go through one queue, so they never overlap and a burst of edits collapses into one follow-up build
 * - A failed build, e.g. of a half-written file, keeps the last good PDF
 * - Without a root the PDF pane lists the files that could be one, choosing writes it into the shown file
 * - The build status sits in the top bar's `workbenchStatus` slot
 */
export function LatexWorkbench({ texFolder, onTexFolder, ...props }: LatexWorkbenchProps) {
  const { project, error, reload, choose } = useTexProject(props.path)
  const rememberFolder = useEffectEvent(onTexFolder)
  const folder = project?.folder
  useEffect(() => {
    if (folder !== undefined && folder !== texFolder) rememberFolder(folder)
  }, [folder, texFolder])

  if (!project) {
    return <div className="pt-11">{error ? <LoadFailure message={error} onRetry={reload} /> : <LoadingPlaceholder />}</div>
  }
  return <ProjectWorkbench {...props} project={project} reload={reload} choose={choose} />
}

function ProjectWorkbench({ project, reload, choose, ...props }: ProjectWorkbenchProps & {
  project: TexProject
  reload: () => void
  choose: (root: string) => Promise<void>
}) {
  const build = useLatexBuild(project.root, { folder: project.folder })
  const { trigger } = build
  useEffect(() => { void trigger() }, [trigger])

  const pdfFailure = build.failure?.message ?? (build.result && !build.result.ok ? "Build failed" : null)
  return (
    <Workbench
      {...props}
      project={{ folder: project.folder, root: project.root }}
      renderPane={({ layout }) => project.root === null && project.candidates.length > 0 ? (
        <LatexRootPicker folder={project.folder} candidates={project.candidates} onChoose={choose} />
      ) : (
        <LatexPdfPane
          url={build.pdfUrl}
          documentKey={build.pdfPath}
          building={build.building}
          emptyMessage={pdfFailure ?? (project.root === null ? NO_ROOT : null)}
          pageBox={layout}
        />
      )}
      status={<LatexBuildStatus build={build} path={props.path} folder={project.folder} />}
      onSaved={(_, savedPath) => {
        // A saved magic comment may change the root
        if (savedPath === props.path) reload()
      }}
      quoteLanguage="latex"
    />
  )
}
