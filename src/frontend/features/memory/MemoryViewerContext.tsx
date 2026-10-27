"use client"

import { createContext, useCallback, useMemo, useState, type ReactNode } from "react"
import { useRequiredContext } from "@/hooks/useRequiredContext"

interface MemoryViewerContextState {
  isOpen: boolean
  openMemoryViewer: () => void
  closeMemoryViewer: () => void
}

const MemoryViewerContext = createContext<MemoryViewerContextState | null>(null)

export function useMemoryViewer(): MemoryViewerContextState {
  return useRequiredContext(MemoryViewerContext, "useMemoryViewer must be used within MemoryViewerProvider")
}

/** Owns whether the central memory window is open. */
export function MemoryViewerProvider({ children }: { children: ReactNode }) {
  const [isOpen, setIsOpen] = useState(false)
  const openMemoryViewer = useCallback(() => setIsOpen(true), [])
  const closeMemoryViewer = useCallback(() => setIsOpen(false), [])
  const value = useMemo(() => ({ isOpen, openMemoryViewer, closeMemoryViewer }), [isOpen, openMemoryViewer, closeMemoryViewer])

  return <MemoryViewerContext.Provider value={value}>{children}</MemoryViewerContext.Provider>
}
