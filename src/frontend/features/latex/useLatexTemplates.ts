"use client"

import { useLoaded } from "@/hooks/useLoaded"
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
  return useLoaded(root, loadLatexTemplates).value ?? NOT_A_PRESENTATION
}
