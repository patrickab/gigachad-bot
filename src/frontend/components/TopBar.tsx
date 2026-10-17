import type { ReactNode, Ref } from "react"
import { cn } from "@/lib/utils"

/** App top bar with a left and a right control cluster, floating over the chat or canvas.
 *
 * Wrap every control in one of two containers:
 * - `TopBarPill` -> solid page-colored pill, for selectors and readouts (model, reasoning, tokens)
 * - `TopBarTransparent` -> no backing, for icon toggles (sidebar expanders, wide mode)
 *
 * Examples
 * --------
 * >>> <TopBar
 * ...   left={<>
 * ...     <TopBarTransparent>{sidebarToggle}</TopBarTransparent>
 * ...     <TopBarPill><ModelDropdown … /></TopBarPill>
 * ...   </>}
 * ...   right={<TopBarPill><TokenCounter … /></TopBarPill>}
 * ... />
 */
export function TopBar({ left, right, overlay = false }: { left?: ReactNode; right?: ReactNode; overlay?: boolean }) {
  // The bar ignores pointer events so clicks between controls reach the content beneath.
  // `overlay` lifts it out of the flow so the content below spans the full height.
  return <header className={cn("pointer-events-none z-40 flex h-11 items-center justify-between gap-1 px-3", overlay ? "absolute inset-x-0 top-0" : "relative shrink-0")}>
    <div className="flex min-w-0 items-center gap-3">{left}</div>
    <div className="flex items-center gap-3">{right}</div>
  </header>
}

/** Group controls on a solid pill that masks content scrolling underneath. Hidden when empty. `ref` serves as a portal slot. */
export function TopBarPill({ children, ref }: { children?: ReactNode; ref?: Ref<HTMLDivElement> }) {
  return <div ref={ref} className="pointer-events-auto relative isolate flex h-8.5 min-w-0 items-center gap-1 empty:hidden before:pointer-events-none before:absolute before:-inset-x-2 before:inset-y-0 before:-z-10 before:rounded-full before:bg-paper">
    {children}
  </div>
}

/** Place a control straight on the content without a backing. `ref` serves as a portal slot. */
export function TopBarTransparent({ children, ref }: { children?: ReactNode; ref?: Ref<HTMLDivElement> }) {
  return <div ref={ref} className="pointer-events-auto flex items-center">{children}</div>
}
