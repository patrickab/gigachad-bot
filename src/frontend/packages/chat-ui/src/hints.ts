/**
 * Keyboard click hints, like Vimium's `f`: label every visible clickable, typing a label activates it.
 *
 * - `f` outside text entry opens, Esc, a pointer press, the wheel, resize or leaving the window closes
 * - Programmatic scrolls (a streaming chat following its answer) keep the labels open
 * - Backspace drops the last typed letter, keys matching no label are swallowed
 * - Text fields, selects and editables get focus, everything else a click
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

interface Target {
  element: HTMLElement
  rect: DOMRect
}

interface HintSession {
  handleKey: (event: KeyboardEvent) => void
  close: () => void
}

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

function visibleTargets(): Target[] {
  return [...document.querySelectorAll<HTMLElement>(CLICKABLE_SELECTOR)].flatMap((element) => {
    const rect = visibleRect(element)
    return rect ? [{ element, rect }] : []
  })
}

/** Focus a text entry, click anything else: focusing a button from a keystroke would leave a lasting focus-visible ring. */
function activateElement(target: HTMLElement) {
  if (isTextEntry(target)) target.focus({ preventScroll: true })
  else target.click()
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

/** Label this document's visible clickables, null when there are none. `onClose` runs once they are gone. */
function openHints(onClose: () => void): HintSession | null {
  const targets = visibleTargets()
  const labels = hintLabels(targets.length)
  if (labels.length === 0) return null
  const layer = showBadgeLayer(targets)
  let typed = ""

  const render = () => {
    const shown = (label: string) => (label.startsWith(typed) ? label : null)
    layer.render(labels.map(shown), typed)
  }
  const close = () => {
    layer.remove()
    window.removeEventListener("pointerdown", close, true)
    window.removeEventListener("wheel", close, true)
    window.removeEventListener("resize", close)
    window.removeEventListener("blur", close)
    onClose()
  }
  const type = (next: string) => {
    if (!labels.some((label) => label.startsWith(next))) return
    const localIndex = labels.indexOf(next)
    if (localIndex >= 0) {
      close()
      activateElement(targets[localIndex].element)
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

/** Listen for `f` on this window until the returned cleanup runs. */
export function installClickHints(): () => void {
  let session: HintSession | null = null
  const onKeyDown = (event: KeyboardEvent) => {
    if (session) {
      session.handleKey(event)
      return
    }
    if (!isOpenKey(event)) return
    event.preventDefault()
    event.stopImmediatePropagation()
    session = openHints(() => { session = null })
  }
  window.addEventListener("keydown", onKeyDown, true)
  return () => {
    session?.close()
    window.removeEventListener("keydown", onKeyDown, true)
  }
}
