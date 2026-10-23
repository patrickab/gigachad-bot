import type { ReactNode, Ref } from "react"
import { cn } from "@/lib/utils"

/** Single-row app toolbar with controls anchored to either side.
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
export function TopBar({ left, center, right, overlay = false }: { left?: ReactNode; center?: ReactNode; right?: ReactNode; overlay?: boolean }) {
  // The bar ignores pointer events so clicks between controls reach the content beneath.
  // `overlay` lifts it out of the flow so the content below spans the full height.
  return <header className={cn("pointer-events-none z-40 flex h-11 items-center justify-between gap-3 px-3", overlay ? "absolute inset-x-0 top-0" : "relative shrink-0")}>
    <div className="relative z-10 flex min-w-0 items-center gap-3">{left}</div>
    <div className="absolute inset-x-4 top-[5px] flex justify-center">{center}</div>
    <div className="relative z-10 flex shrink-0 items-center justify-end gap-3">{right}</div>
  </header>
}

/** Group controls on a solid pill that masks content scrolling underneath. Hidden when empty. `ref` serves as a portal slot. */
export function TopBarPill({ children, ref }: { children?: ReactNode; ref?: Ref<HTMLDivElement> }) {
  return <div ref={ref} className="pointer-events-auto relative isolate flex h-8.5 min-w-0 items-center gap-1 empty:hidden before:pointer-events-none before:absolute before:-inset-x-2 before:inset-y-0 before:-z-10 before:rounded-full before:bg-paper">
    {children}
  </div>
}

/** Place a control straight on the content without a backing. Hidden when empty. `ref` serves as a portal slot. */
export function TopBarTransparent({ children, ref }: { children?: ReactNode; ref?: Ref<HTMLDivElement> }) {
  return <div ref={ref} className="pointer-events-auto flex items-center empty:hidden">{children}</div>
}
