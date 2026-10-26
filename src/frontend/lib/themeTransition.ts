// Color switches share one document, even when several workspace tabs are mounted.
let activeTransition: ViewTransition | undefined

/** Apply a color change through the theme bloom: the new palette reveals from the viewport center.
 *
 * - `applyColors` must update the DOM synchronously (wrap React state in `flushSync`)
 * - Falls back to an instant change without View Transitions or with reduced motion
 */
export function switchColors(applyColors: () => void) {
  const root = document.documentElement
  activeTransition?.skipTransition()
  if (!document.startViewTransition || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    applyColors()
    return
  }

  const radius = Math.hypot(window.innerWidth / 2, window.innerHeight / 2)
  root.style.setProperty("--theme-radius", `${radius}px`)
  root.classList.add("theme-transition")
  const transition = document.startViewTransition(applyColors)
  activeTransition = transition
  // Rapid switches skip the old animation, but still apply each requested change.
  void transition.ready.catch(() => {})
  void transition.finished.finally(() => {
    if (activeTransition !== transition) return
    activeTransition = undefined
    root.classList.remove("theme-transition")
    root.style.removeProperty("--theme-radius")
  })
}
