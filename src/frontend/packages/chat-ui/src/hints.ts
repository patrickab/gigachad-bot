/**
 * Keyboard click hints, like Vimium's `f`: label every visible clickable, typing a label activates it.
 *
 * - `f` outside text entry opens, Esc, a pointer press, the wheel, resize or leaving the window closes
 * - Programmatic scrolls (a streaming chat following its answer) keep the labels open
 * - Backspace drops the last typed letter, keys matching no label are swallowed
 * - Text fields, selects and editables get focus, everything else a click
 * - Across a frame boundary: the document that got `f` hosts the session, its `peer` (the frame, or the window
 *   around it) joins as a guest (`openHintGuest`). One label set covers both, each side draws its own labels
 * - Styles: `hints.css`, which reads the host's role tokens and falls back to system colors
 * - Standalone: `hints.ts` and `hints.css` import nothing, copy both into any web app. Needs the Popover API and
 *   `checkVisibility` (Chrome 114, Firefox 125, Safari 17.4)
 *
 * @example React, once at the app root
 * useEffect(() => installClickHints(), [])
 *
 * @example Plain JS
 * const stop = installClickHints()
 */

/** Home row first, then the easiest reaches. */
const HINT_ALPHABET = "fjdkslaghrueiwo"

const CLICKABLE_SELECTOR = [
  "a[href]",
  "button",
  "input:not([type=hidden])",
  "select",
  "textarea",
  "summary",
  "[contenteditable]:not([contenteditable=false])",
  "[tabindex]:not([tabindex^='-'])",
  ...["button", "link", "tab", "menuitem", "menuitemcheckbox", "menuitemradio", "option", "checkbox", "radio", "switch", "treeitem", "combobox"]
    .map((role) => `[role=${role}]`),
].join(",")

const NON_TEXT_INPUT_TYPES = new Set(["button", "checkbox", "color", "file", "image", "radio", "range", "reset", "submit"])
const MODIFIER_KEYS = new Set(["Shift", "Control", "Alt", "Meta", "AltGraph", "CapsLock"])

/** The other document of a session spanning a frame boundary, as seen by the host. */
export interface HintPeer {
  /** Have the peer collect its visible clickables (`openHintGuest`), resolves their count, 0 when it does not answer. */
  open: () => Promise<number>
  /** Labels for the peer's clickables in its own order, null hides one. `typed` is already entered. */
  show: (labels: (string | null)[], typed: string) => void
  activate: (index: number) => void
  close: () => void
}

/** This document's side of a session hosted by its peer. */
export interface HintGuest {
  count: number
  show: (labels: (string | null)[], typed: string) => void
  activate: (index: number) => void
  close: () => void
}

/**
 * Click hint options.
 *
 * Attributes
 * ----------
 * peer : () => HintPeer | null
 *     Asked on every open: the document whose clickables join this session, null when none.
 * isCovered : (rect: DOMRect) => boolean
 *     True for a target another document draws over, e.g. a parent window's chrome laid over this frame.
 */
export interface ClickHintOptions {
  peer?: () => HintPeer | null
  isCovered?: (rect: DOMRect) => boolean
}

interface Target {
  element: HTMLElement
  rect: DOMRect
}

interface HintSession {
  handleKey: (event: KeyboardEvent) => void
  close: () => void
}

const NOTHING_COVERED = () => false

/**
 * Prefix-free labels for `count` targets, as short as the alphabet allows.
 *
 * >>> hintLabels(3, "ab")
 * ["b", "aa", "ab"]
 */
export function hintLabels(count: number, alphabet = HINT_ALPHABET): string[] {
  const labels = [""]
  let expanded = 0
  while (expanded === 0 || labels.length - expanded < count) {
    const prefix = labels[expanded++]
    for (const letter of alphabet) labels.push(prefix + letter)
  }
  return labels.slice(expanded, expanded + count)
}

function isTextEntry(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  if (target.isContentEditable || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement) return true
  return target instanceof HTMLInputElement && !NON_TEXT_INPUT_TYPES.has(target.type)
}

function isOpenKey(event: KeyboardEvent): boolean {
  const plainKey = !event.altKey && !event.ctrlKey && !event.metaKey && !event.repeat && !event.isComposing
  return plainKey && event.key === "f" && !event.defaultPrevented && !isTextEntry(event.target)
}

function isDisabled(element: HTMLElement): boolean {
  return element.matches(":disabled, [aria-disabled=true]") || element.closest("[inert], [aria-hidden=true]") !== null
}

/** The part of `element` on screen, or null when hidden, off screen or covered. */
function visibleRect(element: HTMLElement): DOMRect | null {
  if (isDisabled(element) || !element.checkVisibility({ visibilityProperty: true })) return null
  for (const rect of element.getClientRects()) {
    const left = Math.max(rect.left, 0)
    const top = Math.max(rect.top, 0)
    const right = Math.min(rect.right, window.innerWidth)
    const bottom = Math.min(rect.bottom, window.innerHeight)
    if (right - left < 1 || bottom - top < 1) continue
    const hit = document.elementFromPoint((left + right) / 2, (top + bottom) / 2)
    if (hit && (hit === element || element.contains(hit))) return new DOMRect(left, top, right - left, bottom - top)
  }
  return null
}

function visibleTargets(isCovered: (rect: DOMRect) => boolean): Target[] {
  return [...document.querySelectorAll<HTMLElement>(CLICKABLE_SELECTOR)].flatMap((element) => {
    const rect = visibleRect(element)
    return rect && !isCovered(rect) ? [{ element, rect }] : []
  })
}

function activateElement(target: HTMLElement) {
  target.focus({ preventScroll: true })
  if (!isTextEntry(target)) target.click()
}

/** One hidden badge per target in a top-layer popover, above every dialog. */
function showBadgeLayer(targets: Target[]) {
  const container = document.createElement("div")
  container.className = "gc-hints"
  container.popover = "manual"
  const badges = targets.map(({ rect }) => {
    const badge = document.createElement("span")
    badge.className = "gc-hint"
    badge.hidden = true
    badge.style.left = `${Math.max(rect.left - 4, 2)}px`
    badge.style.top = `${Math.max(rect.top - 4, 2)}px`
    container.append(badge)
    return badge
  })
  document.body.append(container)
  container.showPopover()
  const render = (labels: (string | null)[], typed: string) => badges.forEach((badge, index) => renderBadge(badge, labels[index] ?? null, typed))
  return { render, remove: () => container.remove() }
}

function renderBadge(badge: HTMLElement, label: string | null, typed: string) {
  badge.hidden = label === null
  if (label === null) return
  const typedPart = document.createElement("span")
  typedPart.className = "gc-hint-typed"
  typedPart.textContent = typed
  badge.replaceChildren(typedPart, label.slice(typed.length))
}

/** Join a session the peer hosts: collect this document's visible clickables and draw the labels it sends. */
export function openHintGuest(isCovered: (rect: DOMRect) => boolean = NOTHING_COVERED): HintGuest {
  const targets = visibleTargets(isCovered)
  const layer = showBadgeLayer(targets)
  return {
    count: targets.length,
    show: layer.render,
    activate: (index) => {
      layer.remove()
      const target = targets[index]
      if (target) activateElement(target.element)
    },
    close: layer.remove,
  }
}

/** Label this document's visible clickables and `remote` ones of a peer, null when there are none. `onClose` runs once they are gone. */
function openHints(remote: { peer: HintPeer; count: number } | null, isCovered: (rect: DOMRect) => boolean, onClose: () => void): HintSession | null {
  const targets = visibleTargets(isCovered)
  const labels = hintLabels(targets.length + (remote?.count ?? 0))
  if (labels.length === 0) {
    remote?.peer.close()
    return null
  }
  const localLabels = labels.slice(0, targets.length)
  const remoteLabels = labels.slice(targets.length)
  const layer = showBadgeLayer(targets)
  let typed = ""

  const render = () => {
    const shown = (label: string) => (label.startsWith(typed) ? label : null)
    layer.render(localLabels.map(shown), typed)
    remote?.peer.show(remoteLabels.map(shown), typed)
  }
  const close = () => {
    layer.remove()
    remote?.peer.close()
    window.removeEventListener("pointerdown", close, true)
    window.removeEventListener("wheel", close, true)
    window.removeEventListener("resize", close)
    window.removeEventListener("blur", close)
    onClose()
  }
  const type = (next: string) => {
    if (!labels.some((label) => label.startsWith(next))) return
    const localIndex = localLabels.indexOf(next)
    const remoteIndex = remoteLabels.indexOf(next)
    if (localIndex >= 0) {
      close()
      activateElement(targets[localIndex].element)
    } else if (remoteIndex >= 0) {
      remote?.peer.activate(remoteIndex)
      close()
    } else {
      typed = next
      render()
    }
  }

  render()
  window.addEventListener("pointerdown", close, true)
  window.addEventListener("wheel", close, { capture: true, passive: true })
  window.addEventListener("resize", close)
  window.addEventListener("blur", close)

  const handleKey = (event: KeyboardEvent) => {
    if (event.isComposing || MODIFIER_KEYS.has(event.key)) return
    if (event.altKey || event.ctrlKey || event.metaKey) {
      close()
      return
    }
    event.preventDefault()
    event.stopImmediatePropagation()
    if (event.key === "Escape") close()
    else if (event.key === "Backspace") type(typed.slice(0, -1))
    else if (event.key.length === 1) type(typed + event.key.toLowerCase())
  }
  return { handleKey, close }
}

/** Listen for `f` on this window until the returned cleanup runs. Keys typed while the peer answers are swallowed. */
export function installClickHints({ peer = () => null, isCovered = NOTHING_COVERED }: ClickHintOptions = {}): () => void {
  let session: HintSession | null = null
  let opening = false
  let installed = true

  const start = (remote: { peer: HintPeer; count: number } | null) => {
    session = openHints(remote, isCovered, () => { session = null })
  }
  const onKeyDown = (event: KeyboardEvent) => {
    if (session) {
      session.handleKey(event)
      return
    }
    if (!opening && !isOpenKey(event)) return
    event.preventDefault()
    event.stopImmediatePropagation()
    if (opening) return
    const currentPeer = peer()
    if (!currentPeer) {
      start(null)
      return
    }
    opening = true
    void currentPeer.open().then((count) => {
      opening = false
      if (!installed || count === 0) currentPeer.close()
      if (installed) start(count > 0 ? { peer: currentPeer, count } : null)
    })
  }
  window.addEventListener("keydown", onKeyDown, true)
  return () => {
    installed = false
    session?.close()
    window.removeEventListener("keydown", onKeyDown, true)
  }
}
