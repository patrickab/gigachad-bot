"use client"

import { useState, type ComponentProps, type ReactNode } from "react"
import { X } from "lucide-react"
import { PanelToggleButton } from "@/components/ui/PanelToggleButton"
import { LoadFailure, LoadingPlaceholder } from "@/components/ui/DocumentLoadState"
import { ModelDropdown } from "@/features/settings/ModelDropdown"
import { ReasoningSelector } from "@/features/settings/ReasoningSelector"
import { ThemeToggle } from "@/features/settings/ThemeToggle"
import { DocumentEditor } from "@/features/editor/DocumentEditor"
import { DiffViewer } from "@/features/editor/diff/DiffViewer"
import { FilePanelToggle } from "@/features/editor/FilePanel"
import { PdfAssistant } from "@/features/editor/pdf/PdfAssistant"
import { LazyAnnotatedPdfViewer } from "@/features/editor/pdf/LazyAnnotatedPdfViewer"
import { CanvasWorkspace } from "@/features/editor/canvas/CanvasWorkspace"
import type { useCanvasView } from "@/features/editor/canvas/useCanvasView"
import { ArchitectureWindow } from "@/features/editor/architecture/ArchitectureWindow"
import { FactorySurface } from "@/features/factory/FactorySurface"
import { LearnModelSelector, LearnReasoningSelector } from "@/features/learn/LearnModel"
import { LearnPlanButton } from "@/features/learn/LearnPlanButton"
import { LearnSurface } from "@/features/learn/LearnSurface"
import { LatexWorkbench } from "@/features/latex/LatexWorkbench"
import { MarkdownWorkbench } from "@/features/workbench/MarkdownWorkbench"
import { usePreparedWorkbench } from "@/features/workbench/usePreparedWorkbench"
import { OmpModelSelector, OmpReasoningSelector, OmpTokenCounter, type ShellCommands } from "@/features/omp/OmpComposer"
import { OmpFileChat } from "@/features/omp/OmpFileChat"
import { OmpSurface } from "@/features/omp/OmpSurface"
import { OmpSystemPrompt } from "@/features/omp/OmpSystemPrompt"
import { TokenCounter } from "@/features/chat/TokenCounter"
import { TopBar, TopBarPill, TopBarTransparent } from "@/features/navigation/TopBar"
import { TopBarSlot } from "@/features/navigation/TopBarSlots"
import { itemOf, useTab, type AppSurface, type WorkspaceItem } from "@/contexts/TabContext"
import { useHostAccess } from "@/contexts/SettingsContext"
import type { OmpView, Usage } from "@/lib/domain/types"
import { opensInWorkbench, workbenchKindOf } from "@/lib/domain/workbench"
import { writeRootFile } from "@/lib/backend/api"

/** Model picker props shared by the chat, document and omp top bars. */
export type ModelProps = Omit<ComponentProps<typeof ModelDropdown>, "selectedModel" | "onSelect" | "chatColumn">

/** The parts of the tab's canvas view state that the canvas surface reads. */
type CanvasViewControls = Pick<ReturnType<typeof useCanvasView>,
  "view" | "navigateArchitecture" | "closeCanvas" | "openCreatedCanvas" | "architectureEdgeStyle" | "setArchitectureEdgeStyle">

interface WorkspaceSurfaceProps {
  /** Leads the top bar in every mode while the sidebar is collapsed, the open sidebar holds its own toggle. */
  sidebarToggle: ReactNode
  canvas: CanvasViewControls
  modelProps: ModelProps
  reasoningLevels: string[] | undefined
  reasoningEffort: string
  totalUsage: Usage
  chatColumn: HTMLDivElement | null
  /** The chat, shown in chat mode and whenever another mode has no item to show. */
  chatBody: ReactNode
  refreshDocuments: () => void
  /** The tab's slash commands an omp tab's composer offers (`/memorize`). */
  ompShellCommands: ShellCommands
}

/** What one mode puts in the top bar beside the sidebar toggle, and the surface below it. */
interface Surface {
  left: ReactNode
  right: ReactNode
  /** Scrolling surfaces run on under the floating TopBar. The rest start below it. */
  runsUnderTopBar: boolean
  body: ReactNode
}

/**
 * The tab's workbench item once prepared: LaTeX or Markdown by the file's kind, mounted on its disk path.
 *
 * - A stored document moves to disk first, the item takes its new path and `onDocumentsChanged` reloads the lists that showed its key
 * - Preparing shows a quiet loading note, a failure its reason and a retry
 * - A Markdown preview's `#file:` link opens its target in this tab, in the workbench or the document editor
 * - A rename gives the item its new path and keeps the editor mounted under the path the item was opened with (`sessionKey`)
 * - A LaTeX item's path may change under the same `sessionKey` too: another file of its project is a buffer switch in the same editor, the item keeps its `texFolder`
 * - A LaTeX item learns its project folder (`texFolder`) from the file's project, so the sidebar can route clicks on the project's files to this tab
 */
function WorkbenchItemBody({ item, onItemChange, onDocumentsChanged, onModeLabel, drawerOpen, onDrawerOpenChange, onClose, shellCommands, modelProps }: {
  item: WorkspaceItem
  onItemChange: (item: WorkspaceItem) => void
  onDocumentsChanged: () => void
  onModeLabel: (label: string) => void
  drawerOpen: boolean
  onDrawerOpenChange: (open: boolean) => void
  onClose: () => void
  shellCommands: ShellCommands
  modelProps: ModelProps
}) {
  const { prepared, error, retry } = usePreparedWorkbench(item.path, (path) => {
    onItemChange({ ...item, path })
    onDocumentsChanged()
  }, item.sessionKey)
  if (error) return <div className="pt-11"><LoadFailure message={error} onRetry={retry} /></div>
  if (!prepared) return <div className="pt-11"><LoadingPlaceholder /></div>
  const sessionKey = item.sessionKey ?? prepared.path
  const props = {
    path: prepared.path,
    cwd: prepared.cwd,
    ompView: item.omp,
    onViewChange: (omp: OmpView) => onItemChange({ ...item, omp }),
    onPathChange: (path: string) => onItemChange({ ...item, path, sessionKey }),
    drawerOpen,
    onDrawerOpenChange,
    onClose,
    shellCommands,
    modelProps,
  }
  const navigate = (path: string) => onItemChange({ kind: opensInWorkbench(path) ? "workbench" : "document", path, projectSlug: item.projectSlug })
  return workbenchKindOf(prepared.path) === "latex"
    ? <LatexWorkbench key={sessionKey} {...props} texFolder={item.texFolder} onTexFolder={(texFolder) => onItemChange({ ...item, texFolder })} />
    : <MarkdownWorkbench key={sessionKey} {...props} projectSlug={item.projectSlug} onNavigate={navigate} onModeLabel={onModeLabel} />
}

/** The top bar and the surface of the tab's current mode. */
export function WorkspaceSurface({ sidebarToggle, canvas, modelProps, reasoningLevels, reasoningEffort, totalUsage, chatColumn, chatBody, refreshDocuments, ompShellCommands }: WorkspaceSurfaceProps) {
  const { tab, setModeLabel, updateConfig, setAppMode, setWorkspaceItem } = useTab()
  const [pdfChatOpen, setPdfChatOpen] = useState(false)
  // omp needs host access, other logins keep the PDF's local assistant
  const hostAccess = useHostAccess()
  const [workbenchDrawerOpen, setWorkbenchDrawerOpen] = useState(true)
  const { config, projectSlug, appMode } = tab
  const filePanelToggle = <TopBarTransparent><FilePanelToggle /></TopBarTransparent>

  const surfaces: Record<AppSurface, () => Surface> = {
    chat: () => ({
      left: <>
        <TopBarPill>
          <ReasoningSelector levels={reasoningLevels} reasoningEffort={reasoningEffort} onReasoningChange={(v) => updateConfig({ reasoningEffort: v })} placement="below" />
        </TopBarPill>
        <TopBarTransparent>
          <ModelDropdown selectedModel={config.selectedModel} onSelect={(m) => updateConfig({ selectedModel: m })} {...modelProps} chatColumn={chatColumn} />
        </TopBarTransparent>
        <TopBarSlot name="wideToggle">{(ref) => <TopBarTransparent ref={ref} />}</TopBarSlot>
      </>,
      right: <>
        <TopBarPill>
          <TokenCounter tokens={totalUsage.total_tokens} />
          <ThemeToggle />
        </TopBarPill>
        {filePanelToggle}
      </>,
      runsUnderTopBar: true,
      body: chatBody,
    }),
    canvas: () => ({
      left: <TopBarSlot name="drawing">{(ref) => <TopBarPill ref={ref} />}</TopBarSlot>,
      right: <>
        <TopBarPill>
          <TopBarSlot name="view">{(ref) => <div ref={ref} className="contents" />}</TopBarSlot>
          <ThemeToggle />
        </TopBarPill>
        {filePanelToggle}
      </>,
      runsUnderTopBar: canvas.view.kind !== "architecture",
      body: canvas.view.kind === "architecture"
        ? <ArchitectureWindow path={canvas.view.path} maximized hostScale={1} edgeStyle={canvas.architectureEdgeStyle} onEdgeStyleChange={canvas.setArchitectureEdgeStyle} onNavigate={canvas.navigateArchitecture} />
        : <CanvasWorkspace
          selected={canvas.view.selection}
          slug={projectSlug}
          onCloseEditor={canvas.closeCanvas}
          onCreated={canvas.openCreatedCanvas}
          onModeLabel={setModeLabel}
          editorModel={config.selectedModel}
        />,
    }),
    document: () => {
      const diffItem = itemOf(tab, "diff")
      const documentItem = itemOf(tab, "document")
      const isPdfDocument = !diffItem && !!documentItem?.path.toLowerCase().endsWith(".pdf")
      return {
        left: <TopBarTransparent>
          <ModelDropdown selectedModel={config.selectedModel} onSelect={(model) => updateConfig({ selectedModel: model })} {...modelProps} />
        </TopBarTransparent>,
        right: <>
          {isPdfDocument && <TopBarSlot name="pdf">{(ref) => <TopBarPill ref={ref} />}</TopBarSlot>}
          <TopBarTransparent>
            <ThemeToggle />
            <button aria-label="Close file" onClick={() => setAppMode("chat")} className="rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover">
              <X className="h-4 w-4" />
            </button>
          </TopBarTransparent>
          {isPdfDocument
            ? !pdfChatOpen && <TopBarTransparent><PanelToggleButton open={false} onClick={() => setPdfChatOpen(true)} openLabel="Open chat" closeLabel="Close chat" /></TopBarTransparent>
            : filePanelToggle}
        </>,
        runsUnderTopBar: isPdfDocument,
        body: diffItem ? <DiffViewer key={diffItem.path} root={diffItem.path} />
          : !documentItem ? chatBody
          : isPdfDocument ? (
            <PdfAssistant
              key={documentItem.path}
              path={documentItem.path}
              model={config.selectedModel}
              chatOpen={pdfChatOpen}
              onChatOpenChange={setPdfChatOpen}
              renderChat={hostAccess ? ({ width, onClose }) => (
                <OmpFileChat path={documentItem.path} width={width} onClose={onClose} shellCommands={ompShellCommands} />
              ) : undefined}
            >
              <LazyAnnotatedPdfViewer path={documentItem.path} topInset={44} />
            </PdfAssistant>
          ) : (
            <DocumentEditor
              path={documentItem.path}
              slug={documentItem.projectSlug ?? ""}
              overlay
              persistOverride={documentItem.rootFile ? (content) => writeRootFile(documentItem.path, content) : undefined}
              onClose={() => setAppMode("chat")}
              onSaved={refreshDocuments}
              onModeLabel={setModeLabel}
              onNavigate={(path) => setWorkspaceItem({ ...documentItem, path })}
              model={config.selectedModel}
            />
          ),
      }
    },
    omp: () => {
      const ompItem = itemOf(tab, "omp")
      const ompView = ompItem?.omp ?? null
      return {
        left: <>
          {ompView && <TopBarPill>
            <OmpReasoningSelector view={ompView} />
          </TopBarPill>}
          {ompView && <TopBarTransparent>
            <OmpModelSelector view={ompView} {...modelProps} />
          </TopBarTransparent>}
        </>,
        right: <>
          <TopBarPill>
            {ompView && <OmpSystemPrompt view={ompView} />}
            {ompView && <OmpTokenCounter view={ompView} />}
            <ThemeToggle />
          </TopBarPill>
          {filePanelToggle}
        </>,
        runsUnderTopBar: !!ompView && ompView.kind !== "settings",
        body: ompItem && ompView ? <OmpSurface view={ompView} onViewChange={(omp) => setWorkspaceItem({ ...ompItem, omp })} shellCommands={ompShellCommands} /> : chatBody,
      }
    },
    workbench: () => {
      const workbenchItem = itemOf(tab, "workbench")
      return {
        left: <TopBarSlot name="workbenchAgent">{(ref) => <div ref={ref} className="contents" />}</TopBarSlot>,
        right: <>
          <TopBarSlot name="workbenchStatus">{(ref) => <div ref={ref} className="contents" />}</TopBarSlot>
          <TopBarTransparent>
            <ThemeToggle />
            <button aria-label="Close file" onClick={() => setAppMode("chat")} className="rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover">
              <X className="h-4 w-4" />
            </button>
          </TopBarTransparent>
          {!workbenchDrawerOpen && <TopBarTransparent><PanelToggleButton open={false} onClick={() => setWorkbenchDrawerOpen(true)} openLabel="Open chat" closeLabel="Close chat" /></TopBarTransparent>}
        </>,
        runsUnderTopBar: true,
        body: workbenchItem ? (
          <WorkbenchItemBody
            item={workbenchItem}
            onItemChange={setWorkspaceItem}
            onDocumentsChanged={refreshDocuments}
            onModeLabel={setModeLabel}
            drawerOpen={workbenchDrawerOpen}
            onDrawerOpenChange={setWorkbenchDrawerOpen}
            onClose={() => setAppMode("chat")}
            shellCommands={ompShellCommands}
            modelProps={modelProps}
          />
        ) : chatBody,
      }
    },
    factory: () => {
      const factoryItem = itemOf(tab, "factory")
      return {
        left: null,
        right: <><TopBarPill><ThemeToggle /></TopBarPill>{filePanelToggle}</>,
        runsUnderTopBar: false,
        body: factoryItem ? <FactorySurface item={factoryItem} /> : chatBody,
      }
    },
    learn: () => {
      const learnItem = itemOf(tab, "learn")
      const learnView = learnItem?.learn ?? null
      return {
        left: learnView && <>
          <TopBarPill>
            <LearnReasoningSelector view={learnView} />
          </TopBarPill>
          <TopBarTransparent>
            <LearnModelSelector view={learnView} {...modelProps} />
          </TopBarTransparent>
        </>,
        right: <TopBarPill>
          {learnView && <LearnPlanButton view={learnView} />}
          <ThemeToggle />
        </TopBarPill>,
        runsUnderTopBar: false,
        body: learnItem && learnView ? <LearnSurface view={learnView} onViewChange={(learn) => setWorkspaceItem({ ...learnItem, learn })} /> : chatBody,
      }
    },
  }
  const surface = surfaces[appMode]()

  return <>
    <TopBar
      left={<><TopBarTransparent>{sidebarToggle}</TopBarTransparent>{surface.left}</>}
      right={surface.right}
    />
    <div data-workspace-surface className={`absolute inset-x-0 bottom-0 ${surface.runsUnderTopBar ? "top-0" : "top-11"} ${appMode === "canvas" ? "overflow-hidden" : ""}`}>
      {surface.body}
    </div>
  </>
}
