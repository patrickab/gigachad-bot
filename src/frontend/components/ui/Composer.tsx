"use client"

import { useLayoutEffect, useRef, type ClipboardEvent, type KeyboardEvent, type ReactNode, type RefObject } from "react"
import { ArrowUp, Loader2, Mic, Square } from "lucide-react"
import { useSpeechInput } from "@/hooks/useSpeechInput"
import { cn } from "@/lib/utils"

/** Round icon button of the composer's action row, e.g. its "+" and tools menus. */
export const COMPOSER_ICON_BUTTON = "rounded-full p-1.5 text-ink-faint transition-colors hover:bg-surface-elevated hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted disabled:opacity-30"
/** Send and stop share one circle: raised surface when actionable, outlined when not. */
const SEND_BUTTON = "ml-1.5 flex h-8 w-8 items-center justify-center rounded-full border transition-all duration-200 focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-2 focus-visible:outline-ink-muted"
const SEND_READY = "border-divider bg-surface-elevated text-ink hover:border-divider-strong active:scale-[0.96]"
const SEND_IDLE = "border-divider/50 text-ink-faint"
const MAX_TEXTAREA_HEIGHT_PX = 200

/**
 * Message card: growing textarea over an action row with dictation and send/stop.
 *
 * - Enter submits, Shift+Enter adds a line. An `onKeyDown` that calls `preventDefault` takes the key instead
 * - `leading` / `trailing` add action-row controls left / before the mic
 * - `busy` shows a spinner instead of send, e.g. while an attachment is extracted
 * - `canSend` defaults to a non-blank `value`
 * - `children` render at the card's top, for overlays like a completion menu
 */
export function Composer({
  value, onValueChange, onSubmit, onStop, label, placeholder, disabled = false, isStreaming = false, busy = false,
  canSend = value.trim().length > 0, textareaRef, onKeyDown, onPaste, onBlur, hint, leading, trailing, children,
}: {
  value: string
  onValueChange: (value: string) => void
  onSubmit: () => void
  onStop?: () => void
  label?: string
  placeholder?: string
  disabled?: boolean
  isStreaming?: boolean
  busy?: boolean
  canSend?: boolean
  textareaRef?: RefObject<HTMLTextAreaElement | null>
  onKeyDown?: (event: KeyboardEvent<HTMLTextAreaElement>) => void
  onPaste?: (event: ClipboardEvent<HTMLTextAreaElement>) => void
  onBlur?: () => void
  hint?: string | null
  leading?: ReactNode
  trailing?: ReactNode
  children?: ReactNode
}) {
  const ownTextareaRef = useRef<HTMLTextAreaElement>(null)
  const textarea = textareaRef ?? ownTextareaRef
  const { isListening, toggleListening } = useSpeechInput(value, onValueChange)
  const sendable = canSend && !disabled

  // Fit the textarea to its text, reset first so scrollHeight can shrink
  useLayoutEffect(() => {
    const element = textarea.current
    if (!element) return
    element.style.height = "0"
    element.style.height = `${Math.min(element.scrollHeight, MAX_TEXTAREA_HEIGHT_PX)}px`
  }, [value, textarea])

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    onKeyDown?.(event)
    if (event.defaultPrevented || event.key !== "Enter" || event.shiftKey || isStreaming || busy) return
    event.preventDefault()
    if (sendable) onSubmit()
  }

  return (
    <div className="relative z-20 rounded-2xl border border-divider/50 px-4 pb-2.5 pt-3.5 transition-all duration-300 bg-surface shadow-[var(--shadow-xl),var(--inner-highlight)] focus-within:border-divider focus-within:shadow-[var(--shadow-xl),var(--inner-highlight-strong)]">
      {children}
      <textarea
        ref={textarea}
        aria-label={label}
        rows={1}
        value={value}
        onChange={(event) => onValueChange(event.target.value)}
        onKeyDown={handleKeyDown}
        onBlur={onBlur}
        onPaste={onPaste}
        disabled={disabled && !isStreaming}
        placeholder={placeholder}
        spellCheck={false}
        autoCorrect="off"
        autoCapitalize="off"
        data-gramm="false"
        data-gramm_editor="false"
        data-enable-grammarly="false"
        className="w-full resize-none bg-transparent text-sm leading-6 text-ink placeholder:text-ink-faint outline-none"
      />
      {hint && <p className="mt-1 text-xs text-ink-muted">{hint}</p>}
      <div className="mt-2 flex items-center justify-between">
        <div className="-ml-1.5 flex min-w-0 items-center gap-0.5">{leading}</div>
        <div className="flex items-center gap-0.5">
          {trailing}
          <button aria-label={isListening ? "Stop listening" : "Start listening"}
            onClick={toggleListening}
            disabled={disabled}
            className={cn(COMPOSER_ICON_BUTTON, isListening && "bg-danger-soft text-danger hover:bg-danger-soft hover:text-danger")}
          >
            <Mic className={cn("h-4 w-4", isListening && "animate-pulse")} />
          </button>
          {isStreaming ? (
            <button aria-label="Stop generating" onClick={onStop} className={cn(SEND_BUTTON, SEND_READY)}>
              <Square className="h-3 w-3 fill-current" />
            </button>
          ) : busy ? (
            <button aria-label="Extracting attachment" disabled className={cn(SEND_BUTTON, SEND_IDLE)}>
              <Loader2 className="h-4 w-4 animate-spin" />
            </button>
          ) : (
            <button aria-label="Send message" onClick={onSubmit} disabled={!sendable} className={cn(SEND_BUTTON, sendable ? SEND_READY : SEND_IDLE)}>
              <ArrowUp className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
