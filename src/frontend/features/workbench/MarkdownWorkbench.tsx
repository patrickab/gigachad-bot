"use client"

import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"
import { LoadingPlaceholder } from "@/components/ui/DocumentLoadState"
import { useDisplayName } from "@/contexts/ProjectContext"
import { ARTIFACT_FILL, followFileLink, isToolArtifact } from "@/features/editor/useEditorView"
import { useModeLabel } from "@/hooks/useModeLabel"
import { baseName } from "@/lib/paths"
import { cn } from "@/lib/utils"
import { Workbench, type WorkbenchPane, type WorkbenchProps } from "./Workbench"

interface MarkdownWorkbenchProps extends Omit<WorkbenchProps, "renderLabel" | "renderPane" | "status" | "onSaved" | "onPulled" | "onRunEnd" | "quoteLanguage"> {
  /** The project the file belongs to, which may give it a display name. */
  projectSlug: string | null
  /** A clicked `#file:<path>` link in the preview. */
  onNavigate: (path: string) => void
  onModeLabel: (label: string) => void
}

const CENTERED_NOTE = "flex h-full items-center justify-center text-xs text-ink-faint"

/** The rendered Markdown on a page like a PDF page, lined up with the editor. Pages longer than the pane scroll, artifacts fill the page. */
function MarkdownPage({ path, layout, text, loadError, onNavigate }: WorkbenchPane & { path: string; onNavigate: (path: string) => void }) {
  if (loadError) return <div className={CENTERED_NOTE}>{loadError}</div>
  if (text === null) return <LoadingPlaceholder size="fill" />
  const artifact = isToolArtifact(path, text)
  return (
    <div className="h-full overflow-y-auto" onClick={(event) => followFileLink(event, onNavigate)}>
      <div style={{ paddingTop: layout.top, paddingBottom: layout.bottom }}>
        <div
          className={cn("mx-auto rounded-2xl bg-page p-6 shadow-[var(--shadow-md)]", artifact && ARTIFACT_FILL)}
          style={{ width: layout.width, minHeight: layout.height, height: artifact ? layout.height : undefined }}
        >
          <LaTeXMarkdown content={text} />
        </div>
      </div>
    </div>
  )
}

/**
 * One `.md` file in the workbench: Neovim, its rendered preview and the omp chat drawer.
 *
 * - The preview renders the synced text, so it follows every save and every agent edit
 * - Mermaid and markmap documents fill the page height
 * - `#file:` links in the preview go to `onNavigate`, the file's shown name to `onModeLabel`
 */
export function MarkdownWorkbench({ projectSlug, onNavigate, onModeLabel, ...props }: MarkdownWorkbenchProps) {
  const shownName = useDisplayName()(projectSlug, props.path) ?? baseName(props.path)
  useModeLabel(shownName, true, onModeLabel)
  return (
    <Workbench
      {...props}
      renderLabel="preview"
      renderPane={(pane) => <MarkdownPage path={props.path} onNavigate={onNavigate} {...pane} />}
      quoteLanguage="markdown"
    />
  )
}
