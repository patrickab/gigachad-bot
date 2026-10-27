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
