// File root routes under /fileroots.
import { fileViewerRawUrl, toQuery } from "../urls"
import { del, post, request } from "./http"
import type { DirectoryListing, FileTreeNode, RootFile } from "@/lib/domain/files"
import type { Attachment } from "@/lib/domain/types"

export async function listRootFiles(signal?: AbortSignal): Promise<{ enabled: boolean; files: RootFile[] }> {
  return request("/fileroots/files", { signal })
}

/** Whether any file root is mounted, without listing its files. */
export async function fileRootsEnabled(signal?: AbortSignal): Promise<{ enabled: boolean }> {
  return request("/fileroots/enabled", { signal })
}

type RootTree = { enabled: boolean; tree: FileTreeNode[] }

/** Every mount's tree, a mount listed with one of its `worktrees` shows that worktree instead. */
export async function fileRootTree(worktrees: string[] = [], signal?: AbortSignal): Promise<RootTree> {
  return request(`/fileroots/tree${toQuery({ worktree: worktrees })}`, { signal })
}

export async function addFileRoot(path: string): Promise<RootTree> {
  return post("/fileroots/roots", { path })
}

/** Clone an https or ssh git link into `~/git/<repository>` on the backend host and mount it. */
export async function cloneFileRoot(url: string): Promise<RootTree> {
  return post("/fileroots/clone", { url })
}

export async function removeFileRoot(path: string): Promise<RootTree> {
  return del(`/fileroots/roots${toQuery({ path })}`)
}

/** List a backend directory for picking a workspace; `~` is the backend user's home. */
export async function browseDirectories(path: string): Promise<DirectoryListing> {
  return request(`/fileroots/browse${toQuery({ path })}`)
}

export async function readFileRootRendered(path: string): Promise<string> {
  const data = await request<{ path: string; content: string }>(`/fileroots/rendered${toQuery({ path })}`)
  return data.content
}

export async function writeRootFile(path: string, content: string): Promise<void> {
  await post("/fileroots/file", { path, content })
}

/** Read a file's text content (markdown / unknown-treated-as-text). */
export async function loadFileViewerText(path: string): Promise<string> {
  const data = await request<{ path: string; content: string }>(`/fileviewer/text${toQuery({ path })}`)
  return data.content
}

/** Attach a FileRoots file as a live reference: no copy, content is read from the source. */
export async function attachRootFile(path: string): Promise<Attachment> {
  const data = await request<{ name: string; mime: string; path: string; content?: string; parsedMd?: string }>(
    `/fileroots/attach${toQuery({ path })}`,
    { method: "POST" },
  )
  return { ...data, active: true, vaultPath: data.path, url: fileViewerRawUrl(data.path) }
}
