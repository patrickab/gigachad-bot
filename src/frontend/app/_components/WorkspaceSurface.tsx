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
import { LazyCanvasWorkspace } from "@/features/editor/canvas/LazyCanvas"
import type { useCanvasView } from "@/features/editor/canvas/useCanvasView"
import { LazyArchitectureWindow } from "@/features/editor/architecture/LazyArchitectureWindow"
import { FactorySurface } from "@/features/factory/FactorySurface"
import { LearnModelSelector, LearnReasoningSelector } from "@/features/learn/LearnModel"
import { LearnPlanButton } from "@/features/learn/LearnPlanButton"
import { LearnSurface } from "@/features/learn/LearnSurface"
import { LatexWorkbench } from "@/features/latex/LatexWorkbench"
import { MarkdownWorkbench } from "@/features/workbench/MarkdownWorkbench"
import { NotebookWorkbench } from "@/features/workbench/NotebookWorkbench"
import { usePreparedWorkbench } from "@/features/workbench/usePreparedWorkbench"
import { OmpModelSelector, OmpReasoningSelector, OmpTokenCounter, type ComposerLaunch, type ShellCommands } from "@/features/omp/OmpComposer"
import { OmpFileChat } from "@/features/omp/OmpFileChat"
import { OmpSurface } from "@/features/omp/OmpSurface"
import { OmpSystemPrompt } from "@/features/omp/OmpSystemPrompt"
import { TokenCounter } from "@/features/chat/TokenCounter"
import { TopBar, TopBarPill, TopBarTransparent } from "@/features/navigation/TopBar"
import { SessionPicker } from "@/features/omp/SessionPicker"
import { TopBarSlot } from "@/features/navigation/TopBarSlots"
import { itemOf, useTab, type AppSurface, type WorkspaceItem } from "@/contexts/TabContext"
import { useHostAccess } from "@/contexts/SettingsContext"
import type { OmpView, Usage } from "@/lib/domain/types"
import { opensInWorkbench, workbenchKindOf } from "@/lib/domain/workbench"
import { writeRootFile } from "@/lib/backend/api"
import { isNotebookPath } from "@/lib/paths"

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
  /** The tab's slash commands an omp tab's composer offers (`/memorize`). */
  ompShellCommands: ShellCommands
  /** First message of an omp session started from the chat, sent by the new-session composer of the omp tab. */
  ompLaunch: ComposerLaunch | null
  onOmpLaunched: () => void
  /** Opens a session picked in the top bar's "Agents" picker. */
  onOpenItem: (item: WorkspaceItem, inNewTab?: boolean) => void
}

/** What one mode puts in the top bar beside the sidebar toggle, and the surface below it. */
interface Surface {
  left: ReactNode
  /** Centered in the bar, independent of the left and right items. */
  center?: ReactNode
  right: ReactNode
  /** Scrolling surfaces run on under the floating TopBar. The rest start below it. */
  runsUnderTopBar: boolean
  /** Pills one level above the page, for busy content (canvas, PDF pages) */
  raised?: boolean
  body: ReactNode
}

/**
 * The tab's workbench item once prepared: LaTeX or Markdown by the file's kind, mounted on its disk path.
 *
 * - A stored document moves to disk first and the item takes its new path, the move's change to the project's catalog entry updates the lists
 * - Preparing shows a quiet loading note, a failure its reason and a retry
 * - A Markdown preview's `#file:` link opens its target in this tab, in the workbench or the document editor
 * - A rename gives the item its new path and keeps the editor mounted under the path the item was opened with (`sessionKey`)
 * - A LaTeX item's path may change under the same `sessionKey` too: another file of its project is a buffer switch in the same editor, the item keeps its `texFolder`
 * - A LaTeX item learns its project folder (`texFolder`) from the file's project, so the sidebar can route clicks on the project's files to this tab
 */
function WorkbenchItemBody({ item, onItemChange, onModeLabel, drawerOpen, onDrawerOpenChange, onClose, shellCommands, modelProps }: {
  item: WorkspaceItem
  onItemChange: (item: WorkspaceItem) => void
  onModeLabel: (label: string) => void
  drawerOpen: boolean
  onDrawerOpenChange: (open: boolean) => void
  onClose: () => void
  shellCommands: ShellCommands
  modelProps: ModelProps
}) {
  const { prepared, error, retry } = usePreparedWorkbench(item.path, (path) => onItemChange({ ...item, path }), item.sessionKey)
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
export function WorkspaceSurface({ sidebarToggle, canvas, modelProps, reasoningLevels, reasoningEffort, totalUsage, chatColumn, chatBody, ompShellCommands, ompLaunch, onOmpLaunched, onOpenItem }: WorkspaceSurfaceProps) {
  const { tab, setModeLabel, updateConfig, setAppMode, setWorkspaceItem } = useTab()
  const [pdfChatOpen, setPdfChatOpen] = useState(false)
  // omp needs host access, other logins keep the PDF's local assistant
  const hostAccess = useHostAccess()
  const [workbenchDrawerOpen, setWorkbenchDrawerOpen] = useState(true)
  const { config, projectSlug, appMode } = tab
  const filePanelToggle = <TopBarTransparent><FilePanelToggle /></TopBarTransparent>
  // The workbench's bar, a notebook's too: the surface renders its controls into the agent and status slots, the chat drawer opens at the right end
  const workbenchBar = {
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
  }
  // Only LaTeX workbenches and notebooks render into the center slot, the others keep the plain two-sided bar
  const centerSlot = <TopBarSlot name="center">{(ref) => <div ref={ref} className="contents" />}</TopBarSlot>
  const workbenchChatProps = { drawerOpen: workbenchDrawerOpen, onDrawerOpenChange: setWorkbenchDrawerOpen, shellCommands: ompShellCommands, modelProps }

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
      raised: true,
      body: canvas.view.kind === "architecture"
        ? <LazyArchitectureWindow path={canvas.view.path} maximized hostScale={1} edgeStyle={canvas.architectureEdgeStyle} onEdgeStyleChange={canvas.setArchitectureEdgeStyle} onNavigate={canvas.navigateArchitecture} />
        : <LazyCanvasWorkspace
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
      if (!diffItem && documentItem && isNotebookPath(documentItem.path)) return {
        ...workbenchBar,
        center: centerSlot,
        runsUnderTopBar: true,
        body: <NotebookWorkbench
          path={documentItem.path}
          tabId={tab.id}
          ompView={documentItem.omp}
          onViewChange={(omp) => setWorkspaceItem({ ...documentItem, omp })}
          {...workbenchChatProps}
        />,
      }
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
        raised: isPdfDocument,
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
        body: ompItem && ompView ? <OmpSurface view={ompView} onViewChange={(omp) => setWorkspaceItem({ ...ompItem, omp })} shellCommands={ompShellCommands} launch={ompLaunch} onLaunched={onOmpLaunched} /> : chatBody,
      }
    },
    workbench: () => {
      const workbenchItem = itemOf(tab, "workbench")
      return {
        ...workbenchBar,
        center: workbenchItem && workbenchKindOf(workbenchItem.path) === "latex" ? centerSlot : undefined,
        runsUnderTopBar: true,
        raised: !!workbenchItem && workbenchKindOf(workbenchItem.path) === "latex",
        body: workbenchItem ? (
          <WorkbenchItemBody
            item={workbenchItem}
            onItemChange={setWorkspaceItem}
            onModeLabel={setModeLabel}
            onClose={() => setAppMode("chat")}
            {...workbenchChatProps}
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
      left={<><TopBarTransparent><div className="flex items-center gap-3">{sidebarToggle}<SessionPicker onOpenItem={onOpenItem} /></div></TopBarTransparent>{surface.left}</>}
      center={surface.center}
      right={surface.right}
      raised={surface.raised}
    />
    <div data-workspace-surface className={`absolute inset-x-0 bottom-0 ${surface.runsUnderTopBar ? "top-0" : "top-11"} ${appMode === "canvas" ? "overflow-hidden" : ""}`}>
      {surface.body}
    </div>
  </>
}
