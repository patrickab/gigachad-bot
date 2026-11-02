/** One kernel environment the backend offers for a notebook, see `GET /api/jupyter/kernelspecs`. */
export interface KernelSpecInfo {
  name: string
  display_name: string
  language: string
  /** The environment's python cannot import ipykernel yet, starting a kernel in it installs it first. */
  needs_ipykernel: boolean
}

export interface KernelSpecs {
  /** The spec a notebook opens with when it names none. */
  default: string
  specs: KernelSpecInfo[]
}

/** The kernel the backend runs for one notebook path. */
export interface RunningKernel {
  kernel_id: string
  name: string
}
