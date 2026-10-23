"use client"

import { useCallback, useState } from "react"
import { useStartupRead } from "@/contexts/StartupReadContext"
import { getCategories } from "@/lib/backend/api"
import type { CategoryDef } from "@/lib/domain/types"

export function useMemoryCategories(projectSlug?: string | null) {
  const [globalCategories, setGlobalCategories] = useState<CategoryDef[]>([])
  const [projectCategories, setProjectCategories] = useState<CategoryDef[]>([])

  const loadCategories = useCallback((signal: AbortSignal) => Promise.all([
    getCategories("global", null, signal),
    projectSlug ? getCategories("project", projectSlug, signal) : Promise.resolve({ categories: [] }),
  ]), [projectSlug])

  useStartupRead("Memory categories", loadCategories, ([globalRes, projectRes]) => {
    setGlobalCategories(globalRes.categories)
    setProjectCategories(projectRes.categories)
  })

  return { globalCategories, projectCategories }
}
