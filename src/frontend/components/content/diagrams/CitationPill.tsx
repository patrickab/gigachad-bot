"use client"

import { useCallback, useRef, useState, type ReactNode } from "react"
import { createPortal } from "react-dom"
import { Globe } from "lucide-react"

export interface CitationSource { title: string; url: string; content: string }

// Hover pill for numeric citation links. The popup is portaled and hides after a short delay so the pointer can reach it.
export function CitationPill({ source, children }: { source: CitationSource; children: ReactNode }) {
  const pillRef = useRef<HTMLAnchorElement>(null)
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null)
  const hideTimer = useRef<ReturnType<typeof setTimeout>>(null)

  const show = useCallback(() => {
    if (hideTimer.current) { clearTimeout(hideTimer.current); hideTimer.current = null }
    const el = pillRef.current
    if (!el) return
    const r = el.getBoundingClientRect()
    setPos({ top: r.top - 8, left: r.left + r.width / 2 })
  }, [])

  const hide = useCallback(() => {
    hideTimer.current = setTimeout(() => setPos(null), 150)
  }, [])

  return (
    <>
      <a
        ref={pillRef}
        href={source.url}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-flex items-center align-middle rounded-full bg-surface-elevated/40 border border-divider-strong/40 px-1.5 py-0 text-[10px] text-ink-muted hover:bg-hover hover:border-divider-strong transition-colors no-underline"
        onMouseEnter={show}
        onMouseLeave={hide}
      >
        {children}
      </a>
      {pos && createPortal(
        <div
          style={{ position: "fixed", top: pos.top, left: pos.left, transform: "translate(-50%, -100%)", zIndex: 9999 }}
          className="w-64 rounded-md border border-divider bg-surface p-2.5 shadow-[var(--shadow-xl)]"
          onMouseEnter={show}
          onMouseLeave={hide}
        >
          <span className="flex items-center gap-1.5 mb-0.5">
            <Globe className="h-2.5 w-2.5 text-ink shrink-0" />
            <span className="text-[10px] font-medium text-ink truncate">{children}</span>
          </span>
          {source.title && (
            <span className="text-[10px] font-medium text-ink mb-0.5 line-clamp-2 block">{source.title}</span>
          )}
          {source.content && (
            <span className="text-[9px] text-ink-subtle line-clamp-3 block">{source.content}</span>
          )}
        </div>,
        document.body
      )}
    </>
  )
}
