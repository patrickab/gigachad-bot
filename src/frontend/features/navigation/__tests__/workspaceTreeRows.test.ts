import { describe, expect, it } from "vitest"
import type { FileTreeItem } from "@/components/ui/file-tree/FileTree"
import type { FileTreeNode } from "@/lib/domain/files"
import type { BranchMeta } from "@/lib/domain/histories"
import { projectItems, workspaceItems } from "../workspaceTree"

const MOUNT = "/code/repo"
const noop = () => {}

/** 50 folders of 100 files in one mount, 30 projects of 2000 chats each. */
function fixture() {
  const tree: FileTreeNode = {
    name: "repo", path: MOUNT, type: "root",
    children: Array.from({ length: 50 }, (_, folder) => ({
      name: `d${folder}`, path: `${MOUNT}/d${folder}`, type: "folder" as const,
      children: Array.from({ length: 100 }, (_, file) => ({ name: `f${file}.md`, path: `${MOUNT}/d${folder}/f${file}.md`, type: "file" as const })),
    })),
  }
  const projects = Array.from({ length: 30 }, (_, project) => ({ slug: `p${project}`, name: `P${project}`, tabs: [] }))
  const meta: Record<string, BranchMeta> = {}
  for (const { slug } of projects) {
    for (let chat = 0; chat < 2000; chat++) meta[`${slug}/c${chat}.json`] = { chat_id: null, parent_id: null, branch_message_idx: null, children: [], qa_count: 1 }
  }
  let visited = 0
  const branchMeta = new Proxy(meta, { get: (target, key: string) => { visited++; return target[key] } })
  return { tree, projects, branchMeta, visited: () => visited }
}

/** Rows built for files and folders of the mount, at any depth. */
const mountRows = (items: FileTreeItem<string>[]): number => items.reduce((total, item) =>
  total + (item.id.startsWith(`${MOUNT}/`) ? 1 : 0) + mountRows(item.children ?? []) + mountRows(item.sections ?? []), 0)

function build(expanded: ReadonlySet<string>) {
  const { tree, projects, branchMeta, visited } = fixture()
  const rows = { renameOf: () => undefined, renameFile: async () => {}, moveToFolder: async () => {}, createLatexFile: async () => {}, isExpanded: (id: string) => expanded.has(id) }
  const roots = projectItems({
    ...rows, projects, branchMeta, pending: () => false, hasArchitecture: () => false, documents: {}, notes: [], visibleRootFiles: [],
    activeProjectSlug: null, appMode: "chat", activeFile: null, activeQaIndex: null,
  } as unknown as Parameters<typeof projectItems>[0])
  const items = workspaceItems(roots, {
    ...rows, projects, pending: () => false, rootTreeData: [tree], mounts: [MOUNT], unmounted: [], unused: [], archived: new Set(),
    folder: () => null, sessions: { directoryCode: () => null, directoryActions: () => [] }, onItemOpen: noop,
  } as unknown as Parameters<typeof workspaceItems>[1])
  return { items, visited: visited() }
}

describe("workspace tree rows", () => {
  it("builds rows for expanded folders only and scans the chats once", () => {
    const collapsed = build(new Set())
    expect(mountRows(collapsed.items)).toBe(50)
    // One pass over the 60,000 chats, plus each chat's own lookup for its badge
    expect(collapsed.visited).toBe(120_000)
    expect(mountRows(build(new Set([`${MOUNT}/d3`])).items)).toBe(150)
  })
})
