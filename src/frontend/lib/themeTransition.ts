// Color switches share one document, even when several workspace tabs are mounted.
let activeTransition: ViewTransition | undefined
let queuedSwitches: Promise<void> = Promise.resolve()

/** Run `callback` once the running color switch shows its reveal, at once without one. Returns the cancel.
 *
 * - For cross-origin frames: Chrome paints their updates while it holds the page for the switch, so they recolor
 *   only once the old snapshot (with their old colors) covers them
 */
export function afterColorReveal(callback: () => void): () => void {
  const transition = activeTransition
  if (!transition) {
    callback()
    return () => {}
  }
  let isCancelled = false
  void transition.ready.catch(() => {}).then(() => {
    if (!isCancelled) callback()
  })
  return () => { isCancelled = true }
}

/** Apply a color change through the theme bloom: the new palette reveals from the viewport center.
 *
 * - `applyColors` must update the DOM synchronously (wrap React state in `flushSync`)
 * - A switch during a running bloom waits for it to finish, instead of skipping it to its end state
 * - Falls back to an instant change without View Transitions or with reduced motion
 */
export function switchColors(applyColors: () => void) {
  if (!document.startViewTransition || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    applyColors()
    return
  }
  queuedSwitches = queuedSwitches.then(() => bloom(applyColors))
}

function bloom(applyColors: () => void): Promise<void> {
  const root = document.documentElement
  const radius = Math.hypot(window.innerWidth / 2, window.innerHeight / 2)
  root.style.setProperty("--theme-radius", `${radius}px`)
  root.classList.add("theme-transition")
  const transition = document.startViewTransition(applyColors)
  activeTransition = transition
  void transition.ready.catch(() => {})
  return transition.finished.catch(() => {}).finally(() => {
    activeTransition = undefined
    root.classList.remove("theme-transition")
    root.style.removeProperty("--theme-radius")
  })
}
