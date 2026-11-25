"use client"

import { useCallback, useRef, useState, type ComponentProps, type ReactNode } from "react"
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
import { CanvasCaptureContext, LazyCanvasWorkspace } from "@/features/editor/canvas/LazyCanvas"
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
import { canvasLead } from "@/features/workbench/fileLead"
import { FileChat } from "@/features/workbench/FileChat"
import { PiTab, type PiLaunch } from "@/features/pi/PiTab"
import type { PiInsertion } from "@/features/pi/PiComposer"
import { ChatContextUsage } from "@/features/chat/ContextUsage"
import { TopBar, TopBarPill, TopBarTransparent } from "@/features/navigation/TopBar"
import { PiSessionPicker } from "@/features/pi/PiSessionPicker"
import { piSessionItem } from "@/features/pi/usePiSidebar"
import { TopBarSlot, useTopBarSlot } from "@/features/navigation/TopBarSlots"
import { itemOf, useTab, type AppSurface, type WorkspaceItem } from "@/contexts/TabContext"
import { useHostAccess } from "@/contexts/SettingsContext"
import type { Usage } from "@/lib/domain/types"
import { opensInWorkbench, workbenchKindOf } from "@/lib/domain/workbench"
import { writeRootFile } from "@/lib/backend/api"
import { isNotebookPath } from "@/lib/paths"

/** Model picker props shared by the chat, document, learn and Pi top bars. */
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
  /** First message of a Pi session started from the chat, sent by the Pi tab once its session is ready. */
  launch: PiLaunch | null
  onLaunched: () => void
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
  /** Endless surfaces (the canvas) run on under the floating sidebar cards and their gutters, `--under-left`/`--under-right` from `TabContent`. */
  runsUnderSidebars?: boolean
  /** Pills one level above the page, for busy content (canvas, PDF pages) */
  raised?: boolean
  body: ReactNode
}

/** Width of a canvas's chat panel, the PDF chat's. */
const CANVAS_CHAT_WIDTH_PX = 320

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
function WorkbenchItemBody({ item, onItemChange, onModeLabel, drawerOpen, onDrawerOpenChange, onClose, modelProps }: {
  item: WorkspaceItem
  onItemChange: (item: WorkspaceItem) => void
  onModeLabel: (label: string) => void
  drawerOpen: boolean
  onDrawerOpenChange: (open: boolean) => void
  onClose: () => void
  modelProps: ModelProps
}) {
  const { prepared, error, retry } = usePreparedWorkbench(item.path, (path) => onItemChange({ ...item, path }), item.sessionKey)
  if (error) return <div className="pt-11"><LoadFailure message={error} onRetry={retry} /></div>
  if (!prepared) return <div className="pt-11"><LoadingPlaceholder /></div>
  const sessionKey = item.sessionKey ?? prepared.path
  const props = {
    path: prepared.path,
    chatSessionId: item.chat?.sessionId,
    onChatSessionChange: (sessionId: string) => onItemChange({ ...item, chat: { sessionId } }),
    onPathChange: (path: string) => onItemChange({ ...item, path, sessionKey }),
    drawerOpen,
    onDrawerOpenChange,
    onClose,
    modelProps,
  }
  const navigate = (path: string) => onItemChange({ kind: opensInWorkbench(path) ? "workbench" : "document", path, projectSlug: item.projectSlug })
  return workbenchKindOf(prepared.path) === "latex"
    ? <LatexWorkbench key={sessionKey} {...props} texFolder={item.texFolder} onTexFolder={(texFolder) => onItemChange({ ...item, texFolder })} />
    : <MarkdownWorkbench key={sessionKey} {...props} projectSlug={item.projectSlug} onNavigate={navigate} onModeLabel={onModeLabel} />
}

/** The top bar and the surface of the tab's current mode. */
export function WorkspaceSurface({ sidebarToggle, canvas, modelProps, reasoningLevels, reasoningEffort, totalUsage, chatColumn, chatBody, launch, onLaunched, onOpenItem }: WorkspaceSurfaceProps) {
  const { tab, setModeLabel, updateConfig, setAppMode, setWorkspaceItem } = useTab()
  const [pdfChatOpen, setPdfChatOpen] = useState(false)
  // The workbench's Pi chats need host access, other logins keep the PDF's local assistant
  const hostAccess = useHostAccess()
  const [workbenchDrawerOpen, setWorkbenchDrawerOpen] = useState(true)
  const { config, projectSlug, appMode } = tab
  const piSlots = { thinking: useTopBarSlot("reasoning"), model: useTopBarSlot("model"), usage: useTopBarSlot("usage") }
  const filePanelToggle = <TopBarTransparent><FilePanelToggle /></TopBarTransparent>
  // A conversation's bar, a chat's and a Pi tab's: reasoning, model and wide mode at the left, the context readout and theme toggle at the right
  const conversationBar = ({ reasoning, model, usage }: { reasoning: ReactNode; model: ReactNode; usage: ReactNode }) => ({
    left: <>
      {reasoning}
      {model}
      <TopBarSlot name="wideToggle">{(ref) => <TopBarTransparent ref={ref} />}</TopBarSlot>
    </>,
    right: <>
      <TopBarPill>
        {usage}
        <ThemeToggle />
      </TopBarPill>
      {filePanelToggle}
    </>,
    runsUnderTopBar: true,
  })
  // The workbench's bar, a notebook's too: the surface renders its controls into the status and center slots, the chat drawer opens at the right end
  const workbenchBar = {
    left: null,
    center: <>
      <TopBarSlot name="workbenchStatus">{(ref) => <div ref={ref} className="contents" />}</TopBarSlot>
      <TopBarSlot name="center">{(ref) => <div ref={ref} className="contents" />}</TopBarSlot>
    </>,
    right: <>
      <TopBarTransparent>
        <ThemeToggle />
        <button aria-label="Close file" onClick={() => setAppMode("chat")} className="rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover">
          <X className="h-4 w-4" />
        </button>
      </TopBarTransparent>
      {!workbenchDrawerOpen && <TopBarTransparent><PanelToggleButton open={false} onClick={() => setWorkbenchDrawerOpen(true)} openLabel="Open chat" closeLabel="Close chat" /></TopBarTransparent>}
    </>,
  }
  const workbenchChatProps = { drawerOpen: workbenchDrawerOpen, onDrawerOpenChange: setWorkbenchDrawerOpen, modelProps }
  // A stored or mounted canvas has a Pi chat, a screenshot opens it and lands in its draft. The scratch canvas has none
  const [canvasChatOpen, setCanvasChatOpen] = useState(false)
  const [canvasInsertion, setCanvasInsertion] = useState<PiInsertion | null>(null)
  const captureCount = useRef(0)
  const attachCapture = useCallback((png: File) => {
    setCanvasChatOpen(true)
    setCanvasInsertion({ id: ++captureCount.current, text: "", files: [png] })
  }, [])
  const canvasChatPath = hostAccess && canvas.view.kind === "canvas" ? canvas.view.selection?.path ?? null : null

  const surfaces: Record<AppSurface, () => Surface> = {
    chat: () => ({
      ...conversationBar({
        reasoning: <TopBarPill>
          <ReasoningSelector levels={reasoningLevels} reasoningEffort={reasoningEffort} onReasoningChange={(v) => updateConfig({ reasoningEffort: v })} placement="below" />
        </TopBarPill>,
        model: <TopBarTransparent>
          <ModelDropdown selectedModel={config.selectedModel} onSelect={(m) => updateConfig({ selectedModel: m })} {...modelProps} chatColumn={chatColumn} />
        </TopBarTransparent>,
        usage: <ChatContextUsage usage={totalUsage} model={config.selectedModel} />,
      }),
      body: chatBody,
    }),
    canvas: () => ({
      left: null,
      center: <TopBarSlot name="drawing">{(ref) => <TopBarPill ref={ref} />}</TopBarSlot>,
      right: <>
        <TopBarPill>
          <TopBarSlot name="view">{(ref) => <div ref={ref} className="contents" />}</TopBarSlot>
          <ThemeToggle />
        </TopBarPill>
        {canvasChatPath
          ? !canvasChatOpen && <TopBarTransparent><PanelToggleButton open={false} onClick={() => setCanvasChatOpen(true)} openLabel="Open chat" closeLabel="Close chat" /></TopBarTransparent>
          : filePanelToggle}
      </>,
      runsUnderTopBar: canvas.view.kind !== "architecture",
      runsUnderSidebars: canvas.view.kind !== "architecture",
      raised: true,
      body: canvas.view.kind === "architecture"
        ? <LazyArchitectureWindow path={canvas.view.path} maximized hostScale={1} edgeStyle={canvas.architectureEdgeStyle} onEdgeStyleChange={canvas.setArchitectureEdgeStyle} onNavigate={canvas.navigateArchitecture} />
        : <CanvasCaptureContext.Provider value={canvasChatPath ? attachCapture : null}>
          <LazyCanvasWorkspace
            selected={canvas.view.selection}
            slug={projectSlug}
            onCloseEditor={canvas.closeCanvas}
            onCreated={canvas.openCreatedCanvas}
            onModeLabel={setModeLabel}
            editorModel={config.selectedModel}
          />
          {canvasChatPath && canvasChatOpen && (
            <FileChat
              path={canvasChatPath}
              width={CANVAS_CHAT_WIDTH_PX}
              onClose={() => setCanvasChatOpen(false)}
              modelProps={modelProps}
              lead={() => canvasLead(canvasChatPath)}
              insertion={canvasInsertion}
              onInserted={() => setCanvasInsertion(null)}
            />
          )}
        </CanvasCaptureContext.Provider>,
    }),
    document: () => {
      const diffItem = itemOf(tab, "diff")
      const documentItem = itemOf(tab, "document")
      const isPdfDocument = !diffItem && !!documentItem?.path.toLowerCase().endsWith(".pdf")
      if (!diffItem && documentItem && isNotebookPath(documentItem.path)) return {
        ...workbenchBar,
        runsUnderTopBar: true,
        body: <NotebookWorkbench
          path={documentItem.path}
          tabId={tab.id}
          chatSessionId={documentItem.chat?.sessionId}
          onChatSessionChange={(sessionId) => setWorkspaceItem({ ...documentItem, chat: { sessionId } })}
          {...workbenchChatProps}
        />,
      }
      return {
        left: <TopBarTransparent>
          <ModelDropdown selectedModel={config.selectedModel} onSelect={(model) => updateConfig({ selectedModel: model })} {...modelProps} />
        </TopBarTransparent>,
        // Shown on hover or keyboard focus, pinned while an ink tool is selected (`data-ink-tool`)
        center: isPdfDocument ? <div className="flex opacity-0 transition-opacity duration-200 hover:opacity-100 has-[:focus-visible]:opacity-100 has-[[data-ink-tool]]:opacity-100">
          <TopBarSlot name="pdf">{(ref) => <TopBarPill ref={ref} />}</TopBarSlot>
        </div> : undefined,
        right: <>
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
                <FileChat path={documentItem.path} width={width} onClose={onClose} modelProps={modelProps} />
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
    pi: () => {
      const piItem = itemOf(tab, "pi")
      const piRepo = piItem?.pi?.repo ?? ""
      return {
        // The Pi tab renders its thinking, model and usage into the places a chat's bar has its own
        ...conversationBar({
          reasoning: <TopBarSlot name="reasoning">{(ref) => <TopBarPill ref={ref} />}</TopBarSlot>,
          model: <TopBarSlot name="model">{(ref) => <TopBarTransparent ref={ref} />}</TopBarSlot>,
          usage: <TopBarSlot name="usage">{(ref) => <div ref={ref} className="contents" />}</TopBarSlot>,
        }),
        body: piItem?.pi ? (
          <div className="flex h-full min-h-0 flex-col">
            <PiTab view={piItem.pi} onViewChange={(pi) => setWorkspaceItem({ ...piItem, pi })} modelProps={modelProps} slots={piSlots} launch={launch} onLaunched={onLaunched} onOpenSession={(sessionId) => setWorkspaceItem(piSessionItem({ sessionId }, piRepo, null))} />
          </div>
        ) : chatBody,
      }
    },
    workbench: () => {
      const workbenchItem = itemOf(tab, "workbench")
      return {
        ...workbenchBar,
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
        left: <TopBarSlot name="title">{(ref) => <TopBarTransparent ref={ref} />}</TopBarSlot>,
        center: <TopBarSlot name="center">{(ref) => <TopBarTransparent ref={ref} />}</TopBarSlot>,
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
      left={<><TopBarTransparent><div className="flex items-center gap-3">{sidebarToggle}<PiSessionPicker onOpenItem={onOpenItem} /></div></TopBarTransparent>{surface.left}</>}
      center={surface.center}
      right={surface.right}
      raised={surface.raised}
    />
    <div
      data-workspace-surface
      // Isolated so its own z-indices stay under the left sidebar card (z-10)
      className={`absolute inset-x-0 bottom-0 ${surface.runsUnderSidebars ? "isolate" : ""} ${surface.runsUnderTopBar ? "top-0" : "top-11"} ${appMode === "canvas" ? "overflow-hidden" : ""}`}
      style={surface.runsUnderSidebars ? { left: "calc(-1 * var(--under-left, 0px))", right: "calc(-1 * var(--under-right, 0px))" } : undefined}
    >
      {surface.body}
    </div>
  </>
}
