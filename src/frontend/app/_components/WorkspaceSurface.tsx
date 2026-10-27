"use client"

import { useState, type ComponentProps, type ReactNode } from "react"
import { X } from "lucide-react"
import { PanelToggleButton } from "@/components/ui/PanelToggleButton"
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
import { OmpModelSelector, OmpReasoningSelector, OmpTokenCounter } from "@/features/omp/OmpComposer"
import { OmpSurface } from "@/features/omp/OmpSurface"
import { OmpSystemPrompt } from "@/features/omp/OmpSystemPrompt"
import { TabSwitcher } from "@/features/navigation/TabSwitcher"
import { TokenCounter } from "@/features/chat/TokenCounter"
import { TopBar, TopBarPill, TopBarTransparent } from "@/features/navigation/TopBar"
import { TopBarSlot } from "@/features/navigation/TopBarSlots"
import { itemOf, useTab, type AppSurface } from "@/contexts/TabContext"
import type { Usage } from "@/lib/domain/types"
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
}

/** What one mode puts in the top bar beside the sidebar toggle, and the surface below it. */
interface Surface {
  left: ReactNode
  right: ReactNode
  /** Scrolling surfaces run on under the floating TopBar. The rest start below it. */
  runsUnderTopBar: boolean
  body: ReactNode
}

/** The top bar and the surface of the tab's current mode. */
export function WorkspaceSurface({ sidebarToggle, canvas, modelProps, reasoningLevels, reasoningEffort, totalUsage, chatColumn, chatBody, refreshDocuments }: WorkspaceSurfaceProps) {
  const { tab, setModeLabel, updateConfig, setAppMode, setWorkspaceItem } = useTab()
  const [pdfChatOpen, setPdfChatOpen] = useState(false)
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
            <PdfAssistant key={documentItem.path} path={documentItem.path} model={config.selectedModel} chatOpen={pdfChatOpen} onChatOpenChange={setPdfChatOpen}>
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
        body: ompItem && ompView ? <OmpSurface view={ompView} onViewChange={(omp) => setWorkspaceItem({ ...ompItem, omp })} /> : chatBody,
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
  }
  const surface = surfaces[appMode]()

  return <>
    <TopBar
      center={<TabSwitcher />}
      left={<><TopBarTransparent>{sidebarToggle}</TopBarTransparent>{surface.left}</>}
      right={surface.right}
    />
    <div data-workspace-surface className={`absolute inset-x-0 bottom-0 ${surface.runsUnderTopBar ? "top-0" : "top-11"} ${appMode === "canvas" ? "overflow-hidden" : ""}`}>
      {surface.body}
    </div>
  </>
}
