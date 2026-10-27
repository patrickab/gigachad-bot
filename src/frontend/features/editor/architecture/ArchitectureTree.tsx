"use client"

import { useLayoutEffect, useRef } from "react"
import { applyArchitectureOperations, deleteArchitectureSources } from "@/lib/backend/api"
import type { ArchitectureOperation, ArchitectureTreeEntry, ArchitectureWorkspace } from "@/lib/domain/architecture"
import { useUndoDelete } from "@/contexts/UndoDeleteContext"
import { cn } from "@/lib/utils"
import { FileTree, type FileTreeItem } from "@/components/ui/file-tree/FileTree"
import { useFileTree } from "@/components/ui/file-tree/useFileTree"
import type { Mutate } from "./useArchitectureView"

// The project's architecture as files: systems (a folder whose own file declares
// one system) with their modules, saved views, and anything else hand-written.
// Creating writes LikeC4 from templates; deleting a system takes its modules
// along, and the server drops the connections and view entries naming them.
// A deleted row leaves the tree at once, but its files (and so the diagram) stay
// until the delete commits when the undo window closes.
export function ArchitectureTree({ slug, workspace, current, onOpen, mutate, floating }: {
  slug: string
  workspace: ArchitectureWorkspace
  current: string
  onOpen: (file: string) => void
  mutate: Mutate
  /** A card like the app sidebar; otherwise a flat column inside a small window. */
  floating: boolean
}) {
  const { schedule, pending } = useUndoDelete()
  // Read when a delete commits, which may be after the user has moved to another file.
  const currentRef = useRef(current)
  useLayoutEffect(() => { currentRef.current = current }, [current])
  const { tree, elements, views } = workspace.model
  // One undo key per delete, named after its first path: a system's key hides its modules with it.
  const deleteKey = (path: string) => `architecture:${slug}:${path}`
  const allSystems = tree.filter((entry) => entry.role === "system")
  const systems = allSystems.filter((entry) => !pending(deleteKey(entry.path)))
  const systemIds = new Set(allSystems.map((entry) => entry.element))
  const systemOf = (entry: ArchitectureTreeEntry) => entry.element?.split(".")[0] ?? ""
  const modulesOf = (system: string) => tree.filter((entry) => entry.role === "module" && systemOf(entry) === system && !pending(deleteKey(entry.path)))
  const others = tree.filter((entry) => (entry.role === "file" || (entry.role === "module" && !systemIds.has(systemOf(entry)))) && !pending(deleteKey(entry.path)))
  const titleOf = (entry: ArchitectureTreeEntry) => (entry.element ? elements.find((element) => element.id === entry.element)?.title : null)
    ?? (entry.role === "view" ? views.find((view) => view.id === entry.view)?.title : null)
    ?? entry.path.split("/").pop()!
  // Every system starts open, so its modules are visible.
  const defaultExpanded = allSystems.map((entry) => entry.path)
  const controller = useFileTree({ storageKey: "expanded_architecture_tree", defaultExpanded })
  // A refused name throws, which keeps the tree's name field open with the server's reason.
  const create = async (operation: ArchitectureOperation) => {
    const next = await mutate(() => applyArchitectureOperations(slug, [operation], null))
    const [created] = next.created
    if (created) onOpen(created)
  }
  const remove = (paths: string[], label: string) => {
    schedule(deleteKey(paths[0]), label, async () => {
      await mutate(() => deleteArchitectureSources(slug, paths))
      if (paths.includes(currentRef.current)) onOpen("")
    })
  }
  const itemOf = (entry: ArchitectureTreeEntry, doomed: string[] = [entry.path]): FileTreeItem<string> => ({
    id: entry.path, label: titleOf(entry), type: "element", isActive: entry.path === current,
    onSelect: () => onOpen(entry.path),
    onDelete: () => remove(doomed, titleOf(entry)),
  })
  const systemItems = systems.map((system): FileTreeItem<string> => {
    const modules = modulesOf(system.element!)
    return {
      ...itemOf(system, [system.path, ...modules.map((module) => module.path)]),
      type: "folder",
      children: modules.map((module) => itemOf(module)),
      create: { label: `New module in ${titleOf(system)}`, placeholder: "Module name", onSubmit: (title) => create({ op: "createModule", system: system.element!, title }) },
    }
  })
  const viewItems = tree.filter((entry) => entry.role === "view" && !pending(deleteKey(entry.path))).map((entry) => itemOf(entry))
  const separator = <div role="separator" className="my-4 border-t border-divider/50" />
  return (
    <nav aria-label="Architecture files" className={cn("flex shrink-0 flex-col", floating ? "mb-3 ml-3 w-60 rounded-2xl border border-divider/50 bg-surface shadow-[var(--shadow-lg)]" : "w-44 border-r border-divider/50")}>
      <div className="min-h-0 flex-1 overflow-y-auto px-2 py-3">
        <FileTree label="Systems" items={systemItems} controller={controller}
          create={{ label: "New system", placeholder: "System name", onSubmit: (title) => create({ op: "createSystem", title }) }} />
        {separator}
        <FileTree label="Views" items={viewItems} controller={controller}
          create={{ label: "New view", placeholder: "View name", onSubmit: (title) => create({ op: "createView", title }) }} />
        {others.length > 0 && <>
          {separator}
          <FileTree label="Other files" items={others.map((entry) => itemOf(entry))} controller={controller} />
        </>}
      </div>
    </nav>
  )
}
