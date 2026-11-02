"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { fetchKernelSpecs, startKernel } from "@/lib/backend/api"
import type { KernelSpecInfo, KernelSpecs } from "@/lib/domain/jupyter"
import { errorMessage } from "@/lib/utils"

function describe(reason: unknown): string {
  return errorMessage(reason, "The kernel could not be started.")
}

interface NotebookKernel {
  /** The environments the backend offers, null until listed. */
  specs: KernelSpecs | null
  /** The running kernel, null while it starts or after it failed. */
  kernelId: string | null
  /** Spec name of the kernel that runs or is starting. */
  kernelName: string | null
  error: string | null
  /** Replace the kernel by one of spec `name`, resolves to the spec once the new kernel runs. */
  select: (name: string) => Promise<KernelSpecInfo | null>
}

/**
 * The kernel of the notebook at `path`: lists the environments and starts the default one, or reuses the kernel the backend already runs for the path.
 * The kernel outlives the editor, `useKernelShutdown` ends it with the tab. Nothing starts until `enabled`: a notebook that fails to load gets no kernel.
 */
export function useNotebookKernel(path: string, enabled: boolean): NotebookKernel {
  const pathRef = useRef(path)
  useEffect(() => { pathRef.current = path })
  const [specs, setSpecs] = useState<KernelSpecs | null>(null)
  const [kernel, setKernel] = useState<{ id: string | null; name: string | null }>({ id: null, name: null })
  const [error, setError] = useState<string | null>(null)
  const specsRef = useRef<KernelSpecs | null>(null)

  useEffect(() => {
    if (!enabled) return
    const abort = new AbortController()
    ;(async () => {
      try {
        const listed = await fetchKernelSpecs(pathRef.current, abort.signal)
        specsRef.current = listed
        setSpecs(listed)
        setKernel({ id: null, name: listed.default })
        const started = await startKernel(pathRef.current, listed.default)
        if (!abort.signal.aborted) setKernel({ id: started.kernel_id, name: started.name })
      } catch (reason) {
        if (!abort.signal.aborted) setError(describe(reason))
      }
    })()
    return () => abort.abort()
  }, [enabled])

  const select = useCallback(async (name: string) => {
    setError(null)
    setKernel({ id: null, name })
    try {
      const started = await startKernel(pathRef.current, name)
      setKernel({ id: started.kernel_id, name: started.name })
      return specsRef.current?.specs.find((spec) => spec.name === started.name) ?? null
    } catch (reason) {
      setError(describe(reason))
      return null
    }
  }, [])

  return { specs, kernelId: kernel.id, kernelName: kernel.name, error, select }
}
