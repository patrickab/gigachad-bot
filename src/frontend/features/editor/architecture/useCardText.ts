import { useEffectEvent, useLayoutEffect, useRef, useState, type FocusEvent, type KeyboardEvent, type RefObject } from "react"
import { isModKey } from "@/lib/utils"
import type { ArchitectureNodeData } from "./architectureFlow"

interface PendingFocus {
  index: number
  cursor: number
}

/**
 * A card's text editor: the title and bullet drafts, moving the caret between
 * them, and one commit of both when focus leaves the card. Typing only touches
 * the drafts, so moving from the heading into the bullets never turns a new
 * node into an element mid-edit (that swaps its id and would drop the caret).
 */
export function useCardText(data: ArchitectureNodeData, cardRef: RefObject<HTMLDivElement | null>) {
  const titleRef = useRef<HTMLInputElement>(null)
  const bulletRefs = useRef<Array<HTMLTextAreaElement | null>>([])
  const [editingTitle, setEditingTitle] = useState(false)

  // Model updates refresh the drafts only while nobody is typing in this card.
  const [titleDraft, setTitleDraft] = useState(data.title)
  const refreshTitleDraft = useEffectEvent((title: string) => {
    if (cardRef.current?.contains(document.activeElement)) return
    setTitleDraft(title)
  })
  useLayoutEffect(() => refreshTitleDraft(data.title), [data.title])
  const [bulletDrafts, setBulletDrafts] = useState<string[]>(() => [...data.bullets])
  const refreshBulletDrafts = useEffectEvent((bullets: readonly string[]) => {
    if (cardRef.current?.contains(document.activeElement)) return
    setBulletDrafts([...bullets])
  })
  useLayoutEffect(() => refreshBulletDrafts(data.bullets), [data.bullets])
  // Drop refs for removed rows so the autosize pass cannot touch an unmounted textarea.
  const bulletCount = bulletDrafts.length
  useLayoutEffect(() => { bulletRefs.current.length = bulletCount }, [bulletCount])
  const [pendingFocus, setPendingFocus] = useState<PendingFocus | null>(null)
  useLayoutEffect(() => {
    if (!pendingFocus) return
    const el = bulletRefs.current[pendingFocus.index]
    el?.focus()
    el?.setSelectionRange(pendingFocus.cursor, pendingFocus.cursor)
  }, [pendingFocus])
  const focusBullet = (index: number, cursor: number) => setPendingFocus({ index, cursor })

  // The commit reads the drafts through refs: a key handler may change them and
  // blur in the same event, before React re-renders.
  const titleDraftRef = useRef(titleDraft)
  const bulletDraftsRef = useRef(bulletDrafts)
  useLayoutEffect(() => {
    titleDraftRef.current = titleDraft
    bulletDraftsRef.current = bulletDrafts
  }, [titleDraft, bulletDrafts])
  const setBullets = (next: string[]) => {
    bulletDraftsRef.current = next
    setBulletDrafts(next)
  }
  const handleCardBlur = (event: FocusEvent<HTMLDivElement>) => {
    if (event.currentTarget.contains(event.relatedTarget as globalThis.Node | null)) return
    const title = titleDraftRef.current
    // Strip a bullet glyph or dash the user typed or pasted at the line start.
    const bullets = bulletDraftsRef.current.map((line) => line.replace(/^\s*[•-]\s?/, "").trim()).filter(Boolean)
    // Leaving drops empty lines locally too, even when nothing reaches the model.
    if (bullets.length !== bulletDraftsRef.current.length) setBullets(bullets)
    if (title === data.title && bullets.join("\n") === data.bullets.join("\n")) return
    data.onChange(data.id, { title, bullets })
  }

  // ArrowDown moves into the bullets; Enter does too, starting the first one if there is none.
  const handleTitleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "ArrowDown" && bulletDrafts.length > 0) {
      event.preventDefault()
      focusBullet(0, 0)
    } else if (event.key === "Enter") {
      event.preventDefault()
      if (bulletDrafts.length === 0) setBullets([""])
      focusBullet(0, 0)
    }
  }
  const handleBulletKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>, index: number) => {
    const textarea = event.currentTarget
    if (event.key === "Enter" && !event.shiftKey && !textarea.value.trim()) {
      // Enter on an empty bullet finishes the card: the line goes, focus leaves, the card commits.
      event.preventDefault()
      setBullets(bulletDrafts.filter((_, i) => i !== index))
      textarea.blur()
    } else if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault()
      const pos = textarea.selectionStart
      const next = [...bulletDrafts]
      next.splice(index, 1, textarea.value.slice(0, pos), textarea.value.slice(pos))
      setBullets(next)
      focusBullet(index + 1, 0)
    } else if (event.key === "Backspace" && index > 0 && textarea.selectionStart === 0 && textarea.selectionEnd === 0) {
      event.preventDefault()
      const mergeAt = bulletDrafts[index - 1].length
      const next = [...bulletDrafts]
      next.splice(index - 1, 2, next[index - 1] + next[index])
      setBullets(next)
      focusBullet(index - 1, mergeAt)
    } else if ((event.key === "ArrowUp" || event.key === "ArrowDown") && !event.shiftKey && !event.altKey && !isModKey(event)) {
      // Inside a wrapped bullet the arrow moves the caret between visual lines;
      // only when the browser leaves the caret where it was (first/last line) do we hop rows.
      const before = textarea.selectionStart
      const target = index + (event.key === "ArrowUp" ? -1 : 1)
      requestAnimationFrame(() => {
        if (!textarea.isConnected || textarea.selectionStart !== before || textarea.selectionEnd !== before) return
        if (target < 0) { setEditingTitle(true); return }
        if (target >= bulletDrafts.length) return
        focusBullet(target, Math.min(before, bulletDrafts[target].length))
      })
    }
  }
  const editBullet = (index: number, text: string) => setBullets(bulletDrafts.map((line, i) => i === index ? text : line))
  // Starts a new bullet below the others, or reuses a trailing empty one.
  const startBullet = () => {
    const last = bulletDrafts.length - 1
    if (last >= 0 && !bulletDrafts[last].trim()) { focusBullet(last, 0); return }
    setBullets([...bulletDrafts, ""])
    focusBullet(bulletDrafts.length, 0)
  }

  const claimTitleEdit = useEffectEvent(() => {
    if (data.claimAutoEdit?.(data.id)) setEditingTitle(true)
  })
  // Mount only: a freshly created node claims its one-time edit, nothing else does.
  // eslint-disable-next-line react-hooks/set-state-in-effect -- claiming consumes the surface's one-shot token, which must follow the commit rather than a render that may be discarded
  useLayoutEffect(() => claimTitleEdit(), [])
  useLayoutEffect(() => {
    if (!editingTitle) return
    // React Flow keeps a new node visibility:hidden until it has measured it,
    // and a hidden input silently refuses focus, so retry for a few frames.
    let frame = 0
    let attempts = 0
    const focusTitle = () => {
      const input = titleRef.current
      if (!input || document.activeElement === input) return
      input.focus()
      if (document.activeElement !== input && attempts++ < 10) frame = requestAnimationFrame(focusTitle)
    }
    focusTitle()
    return () => cancelAnimationFrame(frame)
  }, [editingTitle])

  return { titleRef, bulletRefs, editingTitle, setEditingTitle, titleDraft, setTitleDraft, bulletDrafts, editBullet, startBullet, handleCardBlur, handleTitleKeyDown, handleBulletKeyDown }
}
