import { MessageSquare } from "lucide-react"
import { FileTree, type FileTreeItem } from "@/components/ui/file-tree/FileTree"
import { useFileTree } from "@/components/ui/file-tree/useFileTree"
import type { DiffStatus, RootDiffFiles } from "@/lib/domain/files"
import { baseName } from "@/lib/paths"
import type { ReviewNote } from "./review"

type ChangedFile = RootDiffFiles["files"][number]
type LineCounts = { added: number; removed: number }

const STATUS_BADGE: Record<DiffStatus, string> = { modified: "M", added: "A", deleted: "D", renamed: "R", untracked: "U", conflict: "C" }

/** Counts arrive with the file list, null for files without a text diff. */
function fileCounts(file: ChangedFile): LineCounts | null {
  return file.added === null || file.removed === null ? null : { added: file.added, removed: file.removed }
}

export function LineCountLabel({ counts }: { counts: LineCounts }) {
  return <>
    <span className="text-diff-added">+{counts.added}</span>
    <span className="text-diff-removed">−{counts.removed}</span>
  </>
}

/** Nest changed files under their directories, folder ids are `dir:<path>`. */
function buildTree(files: ChangedFile[], activePath: string | null, select: (path: string) => void, notesByPath: Map<string, ReviewNote[]>): FileTreeItem<string>[] {
  const roots: FileTreeItem<string>[] = []
  for (const file of files) {
    const parts = file.path.split("/")
    let siblings = roots
    parts.slice(0, -1).forEach((name, depth) => {
      const id = `dir:${parts.slice(0, depth + 1).join("/")}`
      let folder = siblings.find((item) => item.id === id)
      if (!folder) {
        folder = { id, label: name, type: "folder", children: [] }
        siblings.push(folder)
      }
      siblings = folder.children!
    })
    const counts = fileCounts(file)
    const noteCount = notesByPath.get(file.path)?.length ?? 0
    siblings.push({
      id: file.path, label: parts[parts.length - 1], type: "element", badge: STATUS_BADGE[file.status],
      meta: (counts || noteCount > 0) && <>
        {noteCount > 0 && <span aria-label={`${noteCount} notes`} className="flex items-center gap-0.5 text-ink-muted"><MessageSquare className="h-2.5 w-2.5" />{noteCount}</span>}
        {counts && <LineCountLabel counts={counts} />}
      </>,
      isActive: file.path === activePath, onSelect: () => select(file.path),
    })
  }
  return compactFolders(roots)
}

/** Merge chains of folders holding only one folder into a single `a/b/c` row, the row keeps the deepest id. */
function compactFolders(items: FileTreeItem<string>[]): FileTreeItem<string>[] {
  return items.map((item) => {
    let folder = item
    while (folder.children?.length === 1 && folder.children[0]!.type === "folder") {
      const child = folder.children[0]!
      folder = { ...child, label: `${folder.label}/${child.label}` }
    }
    return folder.children ? { ...folder, children: compactFolders(folder.children) } : folder
  })
}

function folderIds(files: ChangedFile[]): string[] {
  return files.flatMap((file) => file.path.split("/").slice(0, -1).map((_, depth, parts) => `dir:${parts.slice(0, depth + 1).join("/")}`))
}

export function ChangedFiles({ files, activePath, onSelect, root, notesByPath }: {
  files: ChangedFile[]; activePath: string | null; onSelect: (path: string) => void; root: string; notesByPath: Map<string, ReviewNote[]>
}) {
  const controller = useFileTree({ defaultExpanded: folderIds(files) })
  return <FileTree label={`Changes in ${baseName(root)}`} items={buildTree(files, activePath, onSelect, notesByPath)} controller={controller} />
}

/** Pull-request style totals: changed files, then added and removed lines. */
export function ChangeSummary({ files }: { files: ChangedFile[] }) {
  const totals = files.reduce((sum, file) => ({ added: sum.added + (file.added ?? 0), removed: sum.removed + (file.removed ?? 0) }), { added: 0, removed: 0 })
  return <p aria-label="Change summary" className="flex items-center gap-2 px-2 pb-2 text-xs tabular-nums text-ink-faint">
    <span>{files.length} {files.length === 1 ? "file" : "files"}</span>
    <LineCountLabel counts={totals} />
  </p>
}
