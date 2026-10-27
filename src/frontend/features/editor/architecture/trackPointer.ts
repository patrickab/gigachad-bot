/** Follows the pointer on `window` until it lifts or is cancelled, then runs `end`. The returned function ends the tracking early, as a cancel. */
export function trackPointer(move: ((event: PointerEvent) => void) | null, end: (cancelled: boolean) => void): () => void {
  let tracking = true
  const stop = (cancelled: boolean) => {
    if (!tracking) return
    tracking = false
    if (move) window.removeEventListener("pointermove", move)
    window.removeEventListener("pointerup", onUp)
    window.removeEventListener("pointercancel", onCancel)
    end(cancelled)
  }
  const onUp = () => stop(false)
  const onCancel = () => stop(true)
  if (move) window.addEventListener("pointermove", move)
  window.addEventListener("pointerup", onUp)
  window.addEventListener("pointercancel", onCancel)
  return () => stop(true)
}
