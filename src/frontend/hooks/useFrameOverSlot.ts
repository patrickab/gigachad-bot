"use client"

import { useLayoutEffect } from "react"

/** A tab the tab picker shows scaled down in its preview card (`TabManager`). */
const PEEKED_TAB = ".workspace-tab[data-phase='peek']"

/** z-index of a frame over a peeked tab: above the preview window (z-55), below the peeked tab (z-57) so its chrome stays on top. */
const PEEK_FRAME_Z = "56"

function placeOver(frame: HTMLIFrameElement, slot: HTMLElement | null) {
  const { top, left, width, height } = slot?.getBoundingClientRect() ?? { top: 0, left: 0, width: 0, height: 0 }
  Object.assign(frame.style, { top: `${top}px`, left: `${left}px`, width: `${width}px`, height: `${height}px` })
}

/** Keep `frame` over `slot` as it resizes or moves, until the returned stop runs. */
function followSlot(frame: HTMLIFrameElement, slot: HTMLElement): () => void {
  const place = () => placeOver(frame, slot)
  place()
  const observer = new ResizeObserver(place)
  observer.observe(slot)
  observer.observe(document.body)
  window.addEventListener("resize", place)
  document.addEventListener("transitionend", place, true)
  return () => {
    observer.disconnect()
    window.removeEventListener("resize", place)
    document.removeEventListener("transitionend", place, true)
  }
}

/** Scale `frame` from the slot's layout size onto its scaled box every animation frame while its tab is peeked, `onEnd` once it no longer is. */
function followPeek(frame: HTMLIFrameElement, slot: HTMLElement, onEnd: () => void): () => void {
  // React owns the frame's pointer-events, `data-peek` turns them off in CSS instead
  frame.dataset.peek = ""
  let request = 0
  const stop = () => {
    cancelAnimationFrame(request)
    delete frame.dataset.peek
    Object.assign(frame.style, { transform: "", transformOrigin: "", zIndex: "" })
  }
  const follow = () => {
    if (!slot.closest(PEEKED_TAB)) {
      stop()
      onEnd()
      return
    }
    const { top, left, width } = slot.getBoundingClientRect()
    const scale = slot.offsetWidth > 0 ? width / slot.offsetWidth : 1
    Object.assign(frame.style, {
      top: `${top}px`, left: `${left}px`, width: `${slot.offsetWidth}px`, height: `${slot.offsetHeight}px`,
      transform: `scale(${scale})`, transformOrigin: "0 0", zIndex: PEEK_FRAME_Z,
    })
    request = requestAnimationFrame(follow)
  }
  follow()
  return stop
}

/**
 * Lay the fixed `frame` over `slot` and keep it there. The frame never moves in the DOM, moving an iframe reloads it.
 *
 * - Writes the frame's style directly, so it follows a resizing slot in the same paint (a render would trail it)
 * - Without a slot it parks as a zero box at the viewport origin: Chrome throttles an off-screen
 *   cross-origin frame, then the frame app never hydrates and never reports ready
 * - A slot of a peeked tab is scaled down: the frame keeps the slot's layout size, scales onto it and follows it every
 *   animation frame, back to plain following once the tab is picked
 */
export function useFrameOverSlot(frame: HTMLIFrameElement | null, slot: HTMLElement | null) {
  useLayoutEffect(() => {
    if (!frame) return
    if (!slot) {
      placeOver(frame, null)
      return
    }
    let stop = slot.closest(PEEKED_TAB)
      ? followPeek(frame, slot, () => { stop = followSlot(frame, slot) })
      : followSlot(frame, slot)
    return () => stop()
  }, [frame, slot])
}
