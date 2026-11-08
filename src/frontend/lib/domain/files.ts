import { isInside, relativeIn } from "@/lib/paths"

export interface RootFile {
  path: string
  name: string
}

export interface FileTreeNode {
  name: string
  path: string
  type: "root" | "folder" | "file"
  /** Root nodes: the registered mount, `path` differs when one of its worktrees is shown. */
  mount?: string
  children?: FileTreeNode[]
}

/** Registered path of a file-root node: its mount when set, else its own path. */
export const mountOf = (node: FileTreeNode): string => node.mount ?? node.path

/** A folder with its listed tree, a mount root or a project's folder entry. */
interface ListedFolder {
  path: string
  children?: FileTreeNode[] | null
}

/** Whether the listed `nodes` hold the node reached by `names` from them, walking one segment per level. */
function isListed(nodes: FileTreeNode[] | null | undefined, names: string[], depth = 0): boolean {
  const node = nodes?.find((candidate) => candidate.name === names[depth])
  if (!node) return false
  return depth === names.length - 1 || isListed(node.children, names, depth + 1)
}

/**
 * Whether `path` lies below one of `folders` without being in their listed trees yet: a file created since they were listed.
 * A file the tree hides on purpose (ignored, build output) counts as new every time.
 */
export function isNewFileIn(folders: ListedFolder[], path: string): boolean {
  const holders = folders.filter((folder) => folder.children && path !== folder.path && isInside(path, folder.path))
  return holders.length > 0 && !holders.some((folder) => isListed(folder.children, relativeIn(folder.path, path).split("/").filter(Boolean)))
}

/** Subdirectories of `path` on the backend host; `parent` is null at the filesystem root. */
export interface DirectoryListing {
  path: string
  parent: string | null
  directories: string[]
}

export interface Worktree {
  path: string
  branch: string | null
  is_main: boolean
  changed_files: number
  ahead: number
  behind: number
}

export interface WorktreeList {
  /** The mount holds a `.git` folder, only clones have worktrees. */
  is_clone: boolean
  default_base: string | null
  default_branch?: string | null
  branches: string[]
  worktrees: Worktree[]
}

export type DiffStatus = "modified" | "added" | "deleted" | "renamed" | "untracked" | "conflict"

export interface RootDiffFiles {
  is_git_repository: boolean
  files: { path: string; status: DiffStatus; stamp: string; added: number | null; removed: number | null }[]
}
