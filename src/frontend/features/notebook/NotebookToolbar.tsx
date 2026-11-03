"use client"

import { notebookStore, useNotebookStore } from "@datalayer/jupyter-react"
import { FastForward, Play, Plus, RotateCw, Square } from "lucide-react"
import { IconButton } from "@/components/ui/IconButton"
import { PopoverSelector } from "@/components/ui/PopoverSelector"
import { StatusDot, type StatusTone } from "@/components/ui/StatusDot"
import { TopBarPill } from "@/features/navigation/TopBar"
import { InTopBar } from "@/features/navigation/TopBarSlots"
import type { KernelSpecInfo, KernelSpecs } from "@/lib/domain/jupyter"

/** What the kernel is doing, in words for the trigger and the dot's tone. */
function statusOf(starting: boolean, failed: boolean, status: string | undefined): { label: string; tone: StatusTone } {
  if (failed) return { label: "failed", tone: "error" }
  if (starting || !status || status === "unknown" || status === "starting" || status === "restarting") return { label: "starting", tone: "idle" }
  if (status === "busy") return { label: "busy", tone: "attention" }
  if (status === "idle") return { label: "idle", tone: "ready" }
  return { label: status, tone: "error" }
}

function KernelPicker({ specs, name, status, failed, onSelect }: { specs: KernelSpecs | null; name: string | null; status: string | undefined; failed: boolean; onSelect: (name: string) => void }) {
  const current = specs?.specs.find((spec) => spec.name === name)
  const { label, tone } = statusOf(!current, failed, status)
  const shown = current?.display_name ?? name ?? "Kernel"
  return <PopoverSelector<KernelSpecInfo>
    triggerLabel={`Kernel: ${shown}, ${label}`}
    triggerContent={<>
      <StatusDot tone={tone} />
      <span className="truncate">{shown}</span>
    </>}
    triggerClassName="px-2.5"
    label="Kernel environment"
    categories={[{ key: "kernels", label: "Kernels", items: specs?.specs ?? [] }]}
    itemKey={(spec) => spec.name}
    isSelected={(spec) => spec.name === name}
    renderItem={(spec) => <span className="flex-1 truncate">{spec.needs_ipykernel ? `${spec.display_name} (install ipykernel)` : spec.display_name}</span>}
    onSelect={(spec) => onSelect(spec.name)}
    emptyText="No kernels found"
    placement="below"
    panelClassName="left-auto right-0 w-72"
  />
}

interface NotebookToolbarProps {
  notebookId: string
  specs: KernelSpecs | null
  kernelName: string | null
  kernelFailed: boolean
  onSelectKernel: (name: string) => void
  onRestartKernel: () => void
}

/** Run, interrupt, restart and add-cell buttons plus the kernel picker, in the top bar's `center` slot. */
export function NotebookToolbar({ notebookId, specs, kernelName, kernelFailed, onSelectKernel, onRestartKernel }: NotebookToolbarProps) {
  const status = useNotebookStore((state) => state.selectKernelStatus(notebookId))
  const idle = status === "idle"
  return <InTopBar name="center">
    <TopBarPill>
      <div role="toolbar" aria-label="Notebook" className="flex items-center gap-0.5">
        <IconButton icon={Play} label="Run cell" disabled={!idle} onClick={() => notebookStore.getState().run(notebookId)} />
        <IconButton icon={FastForward} label="Run all cells" disabled={!idle} onClick={() => notebookStore.getState().runAll(notebookId)} />
        <IconButton icon={Square} label="Interrupt kernel" disabled={status !== "busy"} onClick={() => notebookStore.getState().interrupt(notebookId)} />
        <IconButton icon={RotateCw} label="Restart kernel" disabled={!status || status === "starting" || status === "restarting"} onClick={onRestartKernel} />
        <IconButton icon={Plus} label="Add cell below" onClick={() => notebookStore.getState().insertBelow(notebookId, "code")} />
        <KernelPicker specs={specs} name={kernelName} status={status} failed={kernelFailed} onSelect={onSelectKernel} />
      </div>
    </TopBarPill>
  </InTopBar>
}
