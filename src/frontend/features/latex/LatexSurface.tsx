"use client"

import { useCallback, useEffect, useRef, useState, type ComponentProps } from "react"
import { ErrorText } from "@/components/ui/ErrorText"
import { useNeovimSetting } from "@/contexts/SettingsContext"
import { NeovimEditor, useNeovimTerminal, type NeovimSelection } from "@/features/editor/NeovimEditor"
import { FilePanel, LOAD_MESSAGES } from "@/features/editor/FilePanel"
import { TEXT_CODEC, useRootFileDocument } from "@/features/editor/useRootFileDocument"
import { TopBarPill, TopBarTransparent } from "@/features/navigation/TopBar"
import { InTopBar } from "@/features/navigation/TopBarSlots"
import { OmpModelSelector, OmpReasoningSelector, useOmpStreaming, type ComposerInsertion, type ShellCommands } from "@/features/omp/OmpComposer"
import { OmpSurface } from "@/features/omp/OmpSurface"
import { writeRootFile } from "@/lib/backend/api"
import type { OmpView } from "@/lib/domain/types"
import { baseName } from "@/lib/paths"
import { cn } from "@/lib/utils"
import { LatexBuildStatus } from "./LatexBuildStatus"
import { LatexSplit, ShowPaneButton, useHiddenPane, usePaneLayout } from "./LatexSplit"
import { LatexPdfPane } from "./LatexPdfPane"
import { quoteSelection } from "./latexBuild"
import { useAgentSync } from "./useAgentSync"
import { useLatexBuild } from "./useLatexBuild"
import { useLatexView, useLatexWorkspace } from "./useLatexSession"

/** Width of the chat panel (the tab's file panel card, like the PDF assistant), editor and PDF share what is left. */
const DRAWER_WIDTH_PX = 360
/** Neovim sits on a page like the PDF's (`.pdf-page`), its size set by the pane layout. */
const EDITOR_PAGE = "rounded-2xl px-3 pt-3 shadow-[var(--shadow-md)]"
/** The view that shows no session, so no agent runs in it. */
const NO_VIEW: OmpView = { kind: "settings" }

const CENTERED_NOTE = "flex min-h-0 flex-1 items-center justify-center text-xs text-ink-faint"

/**
 * One `.tex` file as an Overleaf-like workspace: Neovim on the left, its PDF in the middle, the omp chat in a drawer on the right.
 *
 * - The PDF builds once on open, after every `:w` while omp is idle, and once when an omp run ends
 * - While omp runs, its edits on disk flow into Neovim and the PDF blurs until the run's build arrives
 * - Visual-mode `<leader>a` in Neovim quotes the selected lines into the chat draft, opening the drawer if it is closed
 * - The chat runs in the file's directory, its session is remembered for the file
 * - The top bar's agent controls and build status render into its `latexAgent` and `latexBuild` slots
 */
export function LatexSurface({ path, ompView, onViewChange, drawerOpen, onDrawerOpenChange, onClose, shellCommands, modelProps }: {
  path: string
  /** The chat session this tab's item shows, undefined before the first one. */
  ompView: OmpView | undefined
  onViewChange: (view: OmpView) => void
  drawerOpen: boolean
  onDrawerOpenChange: (open: boolean) => void
  /** `:q` in Neovim. */
  onClose: () => void
  shellCommands: ShellCommands
  modelProps: Omit<ComponentProps<typeof OmpModelSelector>, "view">
}) {
  const neovim = useNeovimSetting()
  const build = useLatexBuild(path)
  const { cwd, error: prepareError } = useLatexWorkspace(path)
  const { view, adopt } = useLatexView(path, ompView, cwd, onViewChange)
  const streaming = useOmpStreaming(view ?? NO_VIEW)

  const write = useCallback((text: string) => writeRootFile(path, text), [path])
  const { value: content, loadError, reload, saveFromNeovim } = useRootFileDocument({ path, codec: TEXT_CODEC, write, blankWhenMissing: false })
  useEffect(() => { reload() }, [reload])
  // The text Neovim and the disk last agreed on: what was loaded, then what each save or push stored
  const syncedRef = useRef<string | null>(null)
  useEffect(() => {
    if (content !== null && syncedRef.current === null) syncedRef.current = content
  }, [content])

  const { trigger } = build
  useEffect(() => { void trigger() }, [trigger])

  const saveAndBuild = async (text: string) => {
    const stored = await saveFromNeovim(text)
    syncedRef.current = stored
    if (!streaming) void trigger()
    return stored
  }

  const [insertion, setInsertion] = useState<ComposerInsertion | null>(null)
  const insertionCount = useRef(0)
  const quoteIntoChat = (selection: NeovimSelection) => {
    onDrawerOpenChange(true)
    setInsertion({ id: ++insertionCount.current, text: quoteSelection(path, selection) })
  }

  const editor = useNeovimTerminal(
    neovim.enabled && content !== null ? { key: path, path, filename: baseName(path), content } : null,
    saveAndBuild,
    onClose,
    { askKeymap: true, onSelection: quoteIntoChat },
  )
  const { editing } = useAgentSync({ path, streaming, setContent: editor.setContent, syncedRef, onRunEnd: trigger })

  const pdfFailure = build.failure?.message ?? (build.result && !build.result.ok ? "Build failed" : null)
  const [hiddenPane, setHiddenPane] = useHiddenPane()
  const splitRef = useRef<HTMLDivElement>(null)
  const layout = usePaneLayout(splitRef, hiddenPane)

  // Editor and PDF share the width and line up at the top, the PDF pages scroll on below.
  // A hidden pane stays mounted, so Neovim keeps its session
  return (
    <div className="flex h-full">
      {/* The padding sits outside the measured area: `useElementSize` reads the content box */}
      <div className="flex min-w-0 flex-1 px-3">
        <div ref={splitRef} className="flex min-w-0 flex-1">
          <div className={cn("flex min-w-0 flex-1 flex-col items-center", hiddenPane === "editor" && "hidden")}>
            <div className="flex flex-col" style={{ marginTop: layout.top, width: layout.width, height: layout.height }}>
              {loadError ? <div className={CENTERED_NOTE}>{LOAD_MESSAGES[loadError]}</div>
                : content === null ? <div className={CENTERED_NOTE}>Loading…</div>
                : !neovim.enabled ? <div className={CENTERED_NOTE}>Neovim is not enabled.</div>
                : <NeovimEditor terminal={editor.terminal} status={editor.status} onRestart={editor.restart} className={EDITOR_PAGE} />}
            </div>
          </div>
          <LatexSplit hidden={hiddenPane} onChange={setHiddenPane} />
          <div className={cn("min-w-0 flex-1", hiddenPane === "pdf" && "hidden")}>
            <LatexPdfPane
              url={build.pdfUrl}
              documentKey={build.pdfPath}
              building={build.building}
              editing={editing}
              emptyMessage={pdfFailure}
              pageBox={layout}
            />
          </div>
        </div>
      </div>
      {drawerOpen && (
        <FilePanel label="Chat" bare onClose={() => onDrawerOpenChange(false)} width={DRAWER_WIDTH_PX}>
          <div className="relative min-h-0 flex-1">
            {view ? (
              <OmpSurface
                view={view}
                onViewChange={adopt}
                shellCommands={shellCommands}
                topInset={0}
                adoptTitle={false}
                backdrop="surface"
                insertion={insertion}
                onInserted={() => setInsertion(null)}
              />
            ) : (
              <div className={`${CENTERED_NOTE} h-full`}>{prepareError ? <ErrorText>{prepareError}</ErrorText> : "Preparing…"}</div>
            )}
          </div>
        </FilePanel>
      )}
      <InTopBar name="latexAgent">
        {view && <>
          <TopBarPill><OmpReasoningSelector view={view} /></TopBarPill>
          <TopBarTransparent><OmpModelSelector view={view} {...modelProps} /></TopBarTransparent>
        </>}
      </InTopBar>
      <InTopBar name="latexBuild">
        <TopBarPill>
          <ShowPaneButton hidden={hiddenPane} onChange={setHiddenPane} />
          <LatexBuildStatus build={build} />
        </TopBarPill>
      </InTopBar>
    </div>
  )
}
