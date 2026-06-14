"use client"

import { createContext, useCallback, useContext, useState, type ReactNode } from "react"

export type MemoryViewerScope = "global" | "project"

export interface MemoryViewerTarget {
  scope: MemoryViewerScope
  projectSlug?: string | null
}

export interface MemoryViewerContextState {
  target: MemoryViewerTarget | null
  openMemoryViewer: (target: MemoryViewerTarget) => void
  closeMemoryViewer: () => void
}

const MemoryViewerContext = createContext<MemoryViewerContextState | null>(null)

export function useMemoryViewer(): MemoryViewerContextState {
  const ctx = useContext(MemoryViewerContext)
  if (!ctx) throw new Error("useMemoryViewer must be used within MemoryViewerProvider")
  return ctx
}

export function MemoryViewerProvider({ children }: { children: ReactNode }) {
  const [target, setTarget] = useState<MemoryViewerTarget | null>(null)

  const openMemoryViewer = useCallback((next: MemoryViewerTarget) => {
    setTarget(next)
  }, [])

  const closeMemoryViewer = useCallback(() => {
    setTarget(null)
  }, [])

  return (
    <MemoryViewerContext.Provider value={{ target, openMemoryViewer, closeMemoryViewer }}>
      {children}
    </MemoryViewerContext.Provider>
  )
}
