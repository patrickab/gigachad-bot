// Jupyter notebook kernel routes.
import { toQuery } from "../urls"
import { post, request, requestVoid } from "./http"
import type { KernelSpecs, RunningKernel } from "@/lib/domain/jupyter"

/** Where the notebook at absolute `path` works, its kernel and its chat: the mounted root holding it, else its folder. */
export async function fetchNotebookDirectory(path: string, signal?: AbortSignal): Promise<string> {
  return (await request<{ path: string }>(`/jupyter/directory${toQuery({ path })}`, { signal })).path
}

/** The kernel environments for the notebook at absolute `path`, and the one it opens with. */
export function fetchKernelSpecs(path: string, signal?: AbortSignal): Promise<KernelSpecs> {
  return request<KernelSpecs>(`/jupyter/kernelspecs${toQuery({ path })}`, { signal })
}

/** Start (or reuse, same name) the kernel of the notebook at `path`. Another name replaces the running one. */
export function startKernel(path: string, name: string): Promise<RunningKernel> {
  return post<RunningKernel>("/jupyter/kernel", { path, name })
}

/** Shut the kernel of the notebook at `path` down, whether or not one runs. */
export function shutdownKernel(path: string): Promise<void> {
  return requestVoid(`/jupyter/kernel${toQuery({ path })}`, { method: "DELETE" })
}
