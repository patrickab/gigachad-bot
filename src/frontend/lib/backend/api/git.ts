// Git worktree and diff routes of mounted file roots.
import { toQuery } from "../urls"
import { del, post, request } from "./http"
import type { DiffStatus, RootDiffFiles, RootFile, WorktreeList } from "@/lib/domain/files"

export async function fetchWorktrees(root: string): Promise<WorktreeList> {
  return request(`/fileroots/worktrees${toQuery({ root })}`)
}

/** Pull `base`, then branch worktree `name` off it, returns the new worktree's path. */
export async function createWorktree(root: string, name: string, base: string): Promise<RootFile> {
  return post("/fileroots/worktrees", { root, name, base })
}

/** Open the inferred default branch without pulling or switching the main checkout. */
export async function openDefaultWorktree(root: string): Promise<RootFile> {
  return post("/fileroots/worktrees/default", { root })
}

/** Throws ApiError 409 when the worktree holds uncommitted changes and `force` is off. */
export async function removeWorktree(root: string, path: string, force = false): Promise<void> {
  await del(`/fileroots/worktrees${toQuery({ root, path, force })}`)
}

/** Throws ApiError 409 when the branch is not fully merged and `force` is off. */
export async function deleteWorktreeBranch(root: string, branch: string, force = false): Promise<void> {
  await del(`/fileroots/worktrees/branch${toQuery({ root, branch, force })}`)
}

/** Files of the git working tree at a mounted root that differ from HEAD. */
export async function fetchRootDiffFiles(root: string): Promise<RootDiffFiles> {
  return request(`/fileroots/diff/files${toQuery({ root })}`)
}

/** Unified patch of one changed file, `supported` is false for binary or oversized files. */
export async function fetchRootFileDiff(root: string, path: string): Promise<{ supported: boolean; status?: DiffStatus; patch?: string }> {
  return request(`/fileroots/diff${toQuery({ root, path })}`)
}

export async function fetchRootDiffContent(root: string, path: string): Promise<{ base_content: string; content: string }> {
  return request(`/fileroots/diff/content${toQuery({ root, path })}`)
}
