"use client"

import { useCallback, useState } from "react"
import { useStartupRead } from "@/contexts/StartupReadContext"
import { getTopics } from "@/lib/backend/api"
import type { MemoryTopic } from "@/lib/domain/types"

export function useMemoryTopics(projectSlug?: string | null) {
  const [globalTopics, setGlobalTopics] = useState<MemoryTopic[]>([])
  const [projectTopics, setProjectTopics] = useState<MemoryTopic[]>([])

  const loadTopics = useCallback((signal: AbortSignal) => Promise.all([
    getTopics("global", null, signal),
    projectSlug ? getTopics("project", projectSlug, signal) : Promise.resolve({ topics: [] }),
  ]), [projectSlug])

  useStartupRead("Memory topics", loadTopics, ([globalRes, projectRes]) => {
    setGlobalTopics(globalRes.topics)
    setProjectTopics(projectRes.topics)
  })

  return { globalTopics, projectTopics }
}
