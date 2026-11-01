"use client"

import { ErrorText } from "@/components/ui/ErrorText"
import { useAsyncAction } from "@/hooks/useAsyncAction"
import { relativeTo } from "@/lib/paths"

/**
 * The PDF pane of a file no root was found for: the files that could be its root, named relative to its `folder`.
 * Choosing one goes to `onChoose`, a failure shows below the list.
 */
export function LatexRootPicker({ folder, candidates, onChoose }: {
  folder: string
  candidates: string[]
  onChoose: (root: string) => Promise<void>
}) {
  const { run, pending, error } = useAsyncAction(onChoose, "Could not set the root document")
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2 text-xs">
      <span className="text-ink-faint">Root document</span>
      <div className="flex flex-col items-stretch">
        {candidates.map((candidate) => (
          <button
            key={candidate}
            type="button"
            disabled={pending}
            onClick={() => void run(candidate)}
            className="rounded px-2 py-1 text-left text-ink-subtle transition-colors hover:bg-hover hover:text-ink disabled:opacity-50"
          >
            {relativeTo(folder, candidate)}
          </button>
        ))}
      </div>
      <ErrorText>{error}</ErrorText>
    </div>
  )
}
