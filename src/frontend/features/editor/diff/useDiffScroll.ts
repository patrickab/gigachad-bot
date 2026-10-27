import { useCallback, useEffect, useRef, useState, type RefObject } from "react"
import type { RootDiffFiles } from "@/lib/domain/files"

/** Leave room above the destination for the sticky header and three code lines. */
const NOTE_JUMP_OFFSET = 96

/** Scroll `container` to the next or previous review note below its sticky file header. */
export function jumpToNote(container: HTMLElement, direction: 1 | -1) {
  const anchor = container.getBoundingClientRect().top + NOTE_JUMP_OFFSET
  const offsets = [...container.querySelectorAll("[data-review-note]")].map((note) => note.getBoundingClientRect().top - anchor)
  const target = direction > 0 ? offsets.find((offset) => offset > 1) : offsets.reverse().find((offset) => offset < -1)
  if (target !== undefined) container.scrollTop += target
}

/** Scroll `container` so `section` starts at its top edge. */
function scrollToSection(container: HTMLElement, section: HTMLElement) {
  container.scrollTop += section.getBoundingClientRect().top - container.getBoundingClientRect().top
}

/** Scroll-tracked file selection over the per-file sections inside `container`; `blocked` pauses tracking. */
export function useDiffScroll(container: RefObject<HTMLElement | null>, files: RootDiffFiles["files"], blocked: boolean) {
  const [selected, setSelected] = useState<string | null>(null)
  const sections = useRef(new Map<string, HTMLElement>())
  const scrollFrame = useRef(0)

  const registerSection = useCallback((path: string, element: HTMLElement | null) => {
    if (element) sections.current.set(path, element)
    else sections.current.delete(path)
  }, [])
  const jumpToFile = (path: string) => {
    setSelected(path)
    const element = container.current
    const section = sections.current.get(path)
    if (element && section) scrollToSection(element, section)
  }
  /** Find the file at the top of the viewport at most once per frame, scroll events fire far more often. */
  const trackScroll = () => {
    if (blocked || scrollFrame.current) return
    scrollFrame.current = requestAnimationFrame(() => {
      scrollFrame.current = 0
      const element = container.current
      if (!element) return
      const top = element.getBoundingClientRect().top
      let visible = files[0]?.path ?? null
      for (const file of files) {
        const section = sections.current.get(file.path)
        if (!section || section.getBoundingClientRect().top > top + 8) break
        visible = file.path
      }
      setSelected(visible)
    })
  }
  useEffect(() => () => cancelAnimationFrame(scrollFrame.current), [])

  return { selected, registerSection, jumpToFile, trackScroll }
}
