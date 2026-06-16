"use client"

import { useEffect, useState } from "react"
import { getCategories } from "@/lib/api"
import type { CategoryDef } from "@/lib/types"

export function useMemoryCategories(projectSlug?: string | null) {
  const [globalCategories, setGlobalCategories] = useState<CategoryDef[]>([])
  const [projectCategories, setProjectCategories] = useState<CategoryDef[]>([])

  useEffect(() => {
    let cancelled = false
    Promise.all([
      getCategories("global"),
      projectSlug ? getCategories("project", projectSlug) : Promise.resolve({ categories: [] }),
    ])
      .then(([globalRes, projectRes]) => {
        if (cancelled) return
        setGlobalCategories(globalRes.categories)
        setProjectCategories(projectRes.categories)
      })
      .catch((e) => {
        // eslint-disable-next-line no-console
        console.error("Failed to load memory categories:", e)
      })
    return () => { cancelled = true }
  }, [projectSlug])

  return { globalCategories, projectCategories }
}
