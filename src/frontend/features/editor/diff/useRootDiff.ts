import { useEffect, useRef, useState } from "react"
import { fetchRootDiffFiles, fetchRootFileDiff } from "@/lib/backend/api"
import type { RootDiffFiles } from "@/lib/domain/files"
import { parsePatch, type PatchLine } from "@/lib/diff"
import { usePolling } from "@/hooks/usePolling"

export type FilePatch = { lines: PatchLine[] | null; failed: boolean; stamp: string }

const POLL_INTERVAL_MS = 2000
const PATCH_WORKERS = 4

/**
 * Poll the changed files of `root` and fetch each file's patch whenever its stamp changes.
 *
 * - `changes` is undefined while loading and null when the directory cannot be read
 * - `root` must stay constant for one mount: callers key the viewer by root
 * - A `revision` change restarts polling with an immediate run
 */
export function useRootDiff(root: string, revision: number): { changes: RootDiffFiles | null | undefined; patches: Record<string, FilePatch> } {
  const [changes, setChanges] = useState<RootDiffFiles | null | undefined>(undefined)
  const [patches, setPatches] = useState<Record<string, FilePatch>>({})
  const patchesRef = useRef(patches)
  useEffect(() => { patchesRef.current = patches }, [patches])

  usePolling(() => fetchRootDiffFiles(root).then(
    (result) => setChanges((previous) => previous && JSON.stringify(previous) === JSON.stringify(result) ? previous : result),
    () => setChanges((previous) => previous ?? null),
  ), { intervalMs: POLL_INTERVAL_MS, refreshOnFocus: true, restartKey: revision })

  useEffect(() => {
    if (!changes) return
    let cancelled = false
    const known = patchesRef.current
    const queue = changes.files.filter((file) => known[file.path]?.stamp !== file.stamp)
    let index = 0
    const load = async () => {
      while (!cancelled && index < queue.length) {
        const file = queue[index++]
        let patch: FilePatch
        try {
          const diff = await fetchRootFileDiff(root, file.path)
          patch = { lines: diff.supported && diff.patch ? parsePatch(diff.patch) : null, failed: false, stamp: file.stamp }
        } catch {
          patch = { lines: null, failed: true, stamp: file.stamp }
        }
        if (cancelled) return
        setPatches((previous) => ({ ...previous, [file.path]: patch }))
      }
    }
    for (let worker = 0; worker < Math.min(PATCH_WORKERS, queue.length); worker++) void load()
    return () => { cancelled = true }
  }, [changes, root])

  return { changes, patches }
}
