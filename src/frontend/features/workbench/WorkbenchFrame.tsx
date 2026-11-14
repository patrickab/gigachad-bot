"use client"

import { useRef, type ReactNode } from "react"
import { TopBarPill } from "@/features/navigation/TopBar"
import { InTopBar } from "@/features/navigation/TopBarSlots"
import { cn } from "@/lib/utils"
import { WorkbenchViewSwitch, usePaneLayout, type PaneLayout, type WorkbenchView } from "./WorkbenchSplit"

/**
 * The workbench layout: Neovim's page and the rendered pane share the width and line up at the top, the chat drawer on the right.
 *
 * - A hidden pane stays mounted, so Neovim keeps its session and the pane its state
 * - The view switch and `status` render into the top bar's `workbenchStatus` slot, `center` into its `center` slot, lifted off the bar with `raised`
 * - The owner holds the view (`useWorkbenchView`), so it can bring a hidden pane back itself
 */
export function WorkbenchFrame({ view: workbenchView, onViewChange, editor, renderPane, chat, status, center, raised = false }: {
  /** The chosen view, `render` whatever it is without an editor. */
  view: WorkbenchView
  onViewChange: (view: WorkbenchView) => void
  /** Neovim's page content. Null: no editor pane, no view switch, the pane takes the width. */
  editor: ReactNode | null
  /** The rendered pane, laid out against Neovim's page. */
  renderPane: (layout: PaneLayout) => ReactNode
  /** The chat drawer, `<WorkbenchChat .../>`. */
  chat: ReactNode
  /** After the view switch in the `workbenchStatus` top-bar slot. */
  status?: ReactNode
  /** Controls in the `center` top-bar slot. */
  center?: ReactNode
  /** Lift the status and center pills off the bar (`chip-raised`), like the notebook's controls. */
  raised?: boolean
}) {
  const hasEditor = editor !== null
  const view = hasEditor ? workbenchView : "render"
  const splitRef = useRef<HTMLDivElement>(null)
  const layout = usePaneLayout(splitRef, view)

  return (
    <div className="flex h-full">
      {/* The padding sits outside the measured area: `useElementSize` reads the content box */}
      <div className={cn("flex min-w-0 flex-1", hasEditor && "px-3")}>
        <div ref={splitRef} className="flex min-w-0 flex-1">
          {hasEditor && (
            <div className={cn("min-w-0 flex-1", view === "render" && "hidden")}>
              <PanePage layout={layout}>{editor}</PanePage>
            </div>
          )}
          <div className={cn("relative min-w-0 flex-1", view === "editor" && "hidden")}>
            {renderPane(layout)}
          </div>
        </div>
      </div>
      {chat}
      {(hasEditor || status != null) && (
        <InTopBar name="workbenchStatus">
          <TopBarPill raised={raised}>
            {hasEditor && <WorkbenchViewSwitch view={workbenchView} onChange={onViewChange} />}
            {status}
          </TopBarPill>
        </InTopBar>
      )}
      {center != null && (
        <InTopBar name="center">
          <TopBarPill raised={raised}>{center}</TopBarPill>
        </InTopBar>
      )}
    </div>
  )
}

/** A page of the split: `layout.width` wide and `layout.height` tall below the top bar, centered in its pane. Neovim's and the notebook's, so both sides line up. */
export function PanePage({ layout, children }: { layout: PaneLayout; children: ReactNode }) {
  return (
    <div className="flex h-full min-w-0 flex-col items-center">
      <div className="flex flex-col" style={{ marginTop: layout.top, width: layout.width, height: layout.height }}>
        {children}
      </div>
    </div>
  )
}

/** A quiet centered note in Neovim's page ("Loading…", load errors, "Neovim is not enabled."). */
export function EditorNote({ children }: { children: ReactNode }) {
  return <div className="flex min-h-0 flex-1 items-center justify-center text-xs text-ink-faint">{children}</div>
}

/** A notice in the top bar's status, e.g. why something beside the editor did not work. */
export function WorkbenchNotice({ children }: { children: ReactNode }) {
  return <span role="alert" className="truncate px-1 text-[11px] text-danger">{children}</span>
}
