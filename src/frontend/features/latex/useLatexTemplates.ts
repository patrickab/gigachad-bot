"use client"

import { useEffect, useState } from "react"
import { loadLatexTemplates } from "@/lib/backend/api"
import type { LatexTemplates } from "@/lib/domain/latex"

const NOT_A_PRESENTATION: LatexTemplates = { theme: null, primitives: [] }

/**
 * The template set of the LaTeX `root`: its theme and slide primitives when it is a presentation.
 *
 * - Read again whenever `root` changes, until the answer comes the root counts as no presentation
 * - No root, or a read that failed, is no presentation
 */
export function useLatexTemplates(root: string | null): LatexTemplates {
  const [loaded, setLoaded] = useState<{ root: string; templates: LatexTemplates } | null>(null)

  useEffect(() => {
    if (root === null) return
    let cancelled = false
    loadLatexTemplates(root).then(
      (templates) => { if (!cancelled) setLoaded({ root, templates }) },
      () => { if (!cancelled) setLoaded({ root, templates: NOT_A_PRESENTATION }) },
    )
    return () => { cancelled = true }
  }, [root])

  return loaded !== null && loaded.root === root ? loaded.templates : NOT_A_PRESENTATION
}
