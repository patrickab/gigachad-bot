import { useEffect, useRef, type RefObject } from "react"

const PEN_TOUCH_SUPPRESS_MS = 700 // touch stays suppressed this long after the last pen event (hover included)

/** ratio: current/start finger distance. (cx, cy): gesture start centre, (mx, my): current centre, both relative to the element. */
export interface TwoFingerFrame { ratio: number, cx: number, cy: number, mx: number, my: number }

interface TwoFingerOptions<S> {
  enabled?: boolean
  /** Fingers landing on a target for which this returns true belong to something nested, not to this surface. */
  ignoreTarget?: (target: EventTarget | null) => boolean
  /** Snapshot the camera when the second finger lands. */
  begin: () => S
  /** Apply the camera for the current frame, derived from the snapshot. */
  update: (start: S, frame: TwoFingerFrame) => void
}

// Two-finger pinch-zoom + pan, the one touch behaviour shared by every canvas.
// Pointer events, not TouchEvents: some Linux browsers (Firefox) never deliver
// TouchEvents though touch pointer events fire fine. Observe-only, so one-finger
// interaction underneath is untouched.
export function useTwoFingerGesture<S>(ref: RefObject<HTMLElement | null>, options: TwoFingerOptions<S>) {
  const optionsRef = useRef(options)
  optionsRef.current = options
  const enabled = options.enabled ?? true
  useEffect(() => {
    const el = ref.current
    if (!el || !enabled) return
    const touches = new Map<number, { x: number, y: number }>()
    let gesture: { start: S, dist: number, cx: number, cy: number } | null = null
    const drop = () => { touches.clear(); gesture = null }

    // Palm rejection: touch is suppressed while the pen is in contact and for
    // PEN_TOUCH_SUPPRESS_MS after any pen event, hover included — the palm lands
    // just before the tip touches and lingers after it lifts. A pen hover with no
    // buttons also clears the contact flag, so a missed pointerup can never leave
    // touch permanently disabled. Capture phase so a stopPropagation elsewhere
    // can't hide pen events from us.
    let penContact = false
    let penLastSeen = 0
    const onPen = (e: PointerEvent) => {
      if (e.pointerType !== "pen") return
      penLastSeen = performance.now()
      penContact = e.type === "pointerdown" || (e.type === "pointermove" && e.buttons !== 0)
      if (penContact) drop()
    }
    const penEvents = ["pointerdown", "pointermove", "pointerup", "pointercancel"] as const
    penEvents.forEach((t) => window.addEventListener(t, onPen, true))
    const penNear = () => penContact || performance.now() - penLastSeen < PEN_TOUCH_SUPPRESS_MS

    const measure = () => {
      const [a, b] = [...touches.values()]
      const rect = el.getBoundingClientRect()
      return { dist: Math.hypot(a!.x - b!.x, a!.y - b!.y), x: (a!.x + b!.x) / 2 - rect.left, y: (a!.y + b!.y) / 2 - rect.top }
    }
    const onDown = (e: PointerEvent) => {
      if (e.pointerType !== "touch" || penNear() || optionsRef.current.ignoreTarget?.(e.target)) return
      touches.set(e.pointerId, { x: e.clientX, y: e.clientY })
      gesture = null
      if (touches.size !== 2) return
      const m = measure()
      gesture = { start: optionsRef.current.begin(), dist: m.dist, cx: m.x, cy: m.y }
    }
    const onMove = (e: PointerEvent) => {
      if (e.pointerType !== "touch" || !touches.has(e.pointerId)) return
      if (penNear()) return drop()
      touches.set(e.pointerId, { x: e.clientX, y: e.clientY })
      if (!gesture || touches.size !== 2) return
      const m = measure()
      optionsRef.current.update(gesture.start, { ratio: m.dist / gesture.dist, cx: gesture.cx, cy: gesture.cy, mx: m.x, my: m.y })
    }
    const onUp = (e: PointerEvent) => {
      if (e.pointerType !== "touch") return
      touches.delete(e.pointerId)
      gesture = null
    }

    el.addEventListener("pointerdown", onDown, true)
    window.addEventListener("pointermove", onMove)
    window.addEventListener("pointerup", onUp)
    window.addEventListener("pointercancel", onUp)
    return () => {
      penEvents.forEach((t) => window.removeEventListener(t, onPen, true))
      el.removeEventListener("pointerdown", onDown, true)
      window.removeEventListener("pointermove", onMove)
      window.removeEventListener("pointerup", onUp)
      window.removeEventListener("pointercancel", onUp)
    }
  }, [ref, enabled])
}
