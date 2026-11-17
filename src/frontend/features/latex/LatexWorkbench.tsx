"use client"

import { useEffect, useEffectEvent } from "react"
import { LoadFailure, LoadingPlaceholder } from "@/components/ui/DocumentLoadState"
import { Workbench, type WorkbenchProps } from "@/features/workbench/Workbench"
import { useWorkbenchNotice } from "@/features/workbench/useWorkbenchNotice"
import { pdfFileNameOf, type TexProject } from "@/lib/domain/latex"
import { AddSlide } from "./AddSlide"
import { DownloadPdfButton } from "./DownloadPdfButton"
import { LatexBuildStatus } from "./LatexBuildStatus"
import { LatexPdfPane } from "./LatexPdfPane"
import { LatexRootPicker } from "./LatexRootPicker"
import { PresentButton } from "./present/PresentButton"
import { useLatexBuild } from "./useLatexBuild"
import { useLatexTemplates } from "./useLatexTemplates"
import { useForwardSearch, useInverseSearch, useSlideGap } from "./useSynctex"
import { useTexProject } from "./useTexProject"

type ProjectWorkbenchProps = Omit<WorkbenchProps, "project" | "renderPane" | "status" | "center" | "notices" | "onSaved" | "quoteLanguage" | "onForwardSearch">

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
 * - The build status sits in the top bar's `workbenchStatus` slot, with a button that saves the PDF as `<root name>.pdf` and one that opens the presenter window
 * - With Neovim, double-clicking the PDF puts Neovim's cursor on the source line of that spot (SyncTeX), in whichever project file it is, and focuses Neovim
 * - `<leader>v` in Neovim scrolls the PDF to the spot of the cursor line and marks it, bringing the PDF pane back when the view hides it
 * - A presentation root (its theme is a template set) gets an "Add slide" button in the top bar's `center` slot, the picked slide primitive lands after the frame under Neovim's cursor
 * - With Neovim, a click in the gap between two pages of a presentation puts Neovim's cursor after the slide above it and marks the gap, so the next slide lands there. The insert, another gap or Escape clears the mark
 * - One notice in the top bar says why the latest lookup, insert or file switch did not work
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
  const { theme, primitives } = useLatexTemplates(project.root)
  const notices = useWorkbenchNotice()
  const inverseSearch = useInverseSearch(project.root, notices.report)
  const forwardSearch = useForwardSearch(project.root, notices.report)
  const slideGap = useSlideGap(project.root, notices.report)
  const { trigger } = build
  useEffect(() => { void trigger() }, [trigger])

  const pdfFailure = build.failure?.message ?? (build.result && !build.result.ok ? "Build failed" : null)
  return (
    <Workbench
      {...props}
      project={{ folder: project.folder, root: project.root }}
      notices={notices}
      renderPane={({ layout, editor }) => project.root === null && project.candidates.length > 0 ? (
        <LatexRootPicker folder={project.folder} candidates={project.candidates} onChoose={choose} />
      ) : (
        <LatexPdfPane
          url={build.pdfUrl}
          documentKey={build.pdfPath}
          building={build.building}
          emptyMessage={pdfFailure ?? (project.root === null ? NO_ROOT : null)}
          pageBox={layout}
          onPageDoubleClick={editor === null ? undefined : (page, x, y, word) => inverseSearch.reveal(page, x, y, word, (path, line, column) => editor.reveal(path, line, column, { focus: true }))}
          reveal={forwardSearch.target}
          slideGap={theme === null || editor === null ? undefined : {
            marked: slideGap.marked,
            onChoose: (page) => slideGap.choose(page, (path, line) => editor.reveal(path, line)),
            onClear: slideGap.clear,
          }}
        />
      )}
      status={<>
        <LatexBuildStatus build={build} path={props.path} folder={project.folder} />
        {project.root !== null && <DownloadPdfButton pdfUrl={build.pdfUrl} fileName={pdfFileNameOf(project.root)} />}
        {project.root !== null && <PresentButton pdfUrl={build.pdfUrl} />}
      </>}
      center={theme === null ? undefined : (editor) => <AddSlide theme={theme} primitives={primitives} onInsert={(text) => {
        if (!editor.send({ type: "insert_frame", text }, "Neovim is not ready, the slide was not added")) return false
        editor.focus()
        slideGap.clear()
        return true
      }} />}
      onForwardSearch={forwardSearch.reveal}
      onSaved={(_, savedPath) => {
        // A saved magic comment may change the root
        if (savedPath === props.path) reload()
      }}
      quoteLanguage="latex"
    />
  )
}
