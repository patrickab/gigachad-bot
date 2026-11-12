// The user-global backend values every tab shows: models, prompts, provider usage, reasoning levels per model, whether any
// file root is mounted, and the sidebar's lists (loose notes, each project's documents, all documents, the file mount
// tree, each mount's worktrees, the LaTeX template sets).
// Consumes the backend client and the change feed with its file events. Provides one module-level `Resource` per value, so
// a change made in one tab reaches every tab, the re-reads that this device's changes to documents and mounts call, the
// forgetting of a deleted project's list, `revalidateOnActivation` for what changes without an event, and
// `watchDocumentLists`, the one subscription that keeps the lists current through change events and file events.

import { createResource, refresh, resourceFamily, revalidateStale, type Resource } from "@/lib/resource"
import {
  ApiError, fetchModels, fetchPrompts, fetchProviderUsage, fetchReasoningLevels, fetchWorktrees, fileRootTree, fileRootsEnabled, listAllDocuments,
  listNotes, listProjectDocuments, loadTemplateSets,
} from "@/lib/backend/api"
import { onFileChange, onFileMove, onFileRemoval } from "@/lib/backend/fileEvents"
import { subscribeToChanges } from "@/lib/backend/syncStream"
import { architectureSlug, isArchitecturePath } from "@/lib/domain/architecturePaths"
import { isNewFileIn } from "@/lib/domain/files"
import type { ProjectDocument } from "@/lib/domain/projects"
import type { ProviderUsageReport } from "@/lib/domain/types"
import { isInside } from "@/lib/paths"

// Reads call the client at load time, never at import, so tests mocking only part of the client can still import this module.

export const modelsResource = createResource((signal) => fetchModels(signal))

/** Resolved system prompt per prompt slug. */
export const promptsResource = createResource((signal) => fetchPrompts(signal))

/** Reasoning levels a model accepts, `[]` when it cannot reason. */
export const reasoningLevelsResource = resourceFamily(async (model, signal) => (await fetchReasoningLevels(model, signal)).levels)

/** Whether any file root is mounted. */
export const fileRootsEnabledResource = createResource(async (signal) => (await fileRootsEnabled(signal)).enabled)

/** Subscription usage of the logins that have limits, null before the first read answered, and whether the latest read failed. */
export interface ProviderUsageState {
  reports: ProviderUsageReport[] | null
  failed: boolean
}

/** Read the usage, `skipCache` past the server's 5 minute cache. A failure keeps the reports read before. */
async function readProviderUsage(skipCache: boolean): Promise<ProviderUsageState> {
  try {
    return { reports: (await fetchProviderUsage(skipCache)).reports.filter((report) => !report.no_limits), failed: false }
  } catch {
    return { reports: providerUsageResource.get()?.reports ?? null, failed: true }
  }
}

export const providerUsageResource: Resource<ProviderUsageState> = createResource(() => readProviderUsage(false))

/** Read usage past the server's cache into `providerUsageResource`. */
export async function refreshProviderUsage(): Promise<void> {
  providerUsageResource.set(await readProviderUsage(true))
}

/** Canvases and notes outside any project. */
export const notesResource = createResource((signal) => listNotes(signal))

/** The files each project lists, by project slug, none for a project that no longer exists. */
export const projectDocumentsResource = resourceFamily(async (slug, signal) => {
  try {
    return await listProjectDocuments(slug, signal)
  } catch (cause) {
    // A deleted or renamed project answers not found, its list holds nothing any more.
    if (cause instanceof ApiError && cause.status === 404) return []
    throw cause
  }
})

/** Every document any project references, the document picker's global list. */
export const allDocumentsResource = createResource(() => listAllDocuments())

/** The file mounts' tree, keyed by the worktree paths shown instead of their clones joined with newlines. */
export const rootTreeResource = resourceFamily(async (worktrees, signal) => (await fileRootTree(worktrees.split("\n").filter(Boolean), signal)).tree)

/** Worktrees of each mount, by mount path. */
export const worktreesResource = resourceFamily((mount) => fetchWorktrees(mount))

/** The LaTeX template sets a new file can start from. */
export const templateSetsResource = createResource(() => loadTemplateSets())

/** Read project `scope`'s documents again (the loose notes for ""), with all documents, after a file was added to it. */
export async function refreshProjectDocuments(scope: string): Promise<void> {
  await Promise.all([scope ? projectDocumentsResource(scope) : notesResource, allDocumentsResource].map(refresh))
}

/** Every document list: the loose notes, each project's documents and all documents. */
const documentLists = () => [notesResource, ...projectDocumentsResource.members(), allDocumentsResource]

/** Read every document list again, after a change that can reach any project's list: a move, rename or deletion. */
export async function refreshDocumentLists(): Promise<void> {
  await Promise.all(documentLists().map(refresh))
}

/** Empty project `slug`'s documents and read all documents again, once the project was deleted or renamed away. */
export function forgetProjectDocuments(slug: string): void {
  projectDocumentsResource(slug).set([])
  allDocumentsResource.invalidate()
}

/** Read the file mount tree again, after this device mounted, unmounted or changed files on disk. */
export async function refreshRootTree(): Promise<void> {
  await Promise.all(rootTreeResource.members().map(refresh))
}

/**
 * Read again what changes without a change event, once a tab comes to the front. Tabs calling it in one task share one request per value.
 *
 * - Every loaded mount tree and every mount's loaded worktrees
 * - Every project list holding a folder entry, whose disk tree it embeds, or architecture sources, which a file mount keeps on disk
 * - Every value whose latest re-read failed
 */
export function revalidateOnActivation(): void {
  const diskBacked = projectDocumentsResource.members().filter((list) => list.get()?.some((doc) => doc.children || isArchitecturePath(doc.path)))
  for (const resource of [...rootTreeResource.members(), ...worktreesResource.members(), ...diskBacked]) resource.invalidate()
  revalidateStale()
}

/** Key of a project's directory or of its catalog entry, which lists its files; the groups are its slug and the catalog file. */
const PROJECT_KEY = /^chat\/([^/]+)(\/project\.json)?$/

/** Prefix of the loose notes' keys. */
const NOTE_PREFIX = "note/"

/** `documents` without the entries at or inside the deleted `path`, the same array when none is. */
function withoutRemoved(documents: ProjectDocument[], path: string): ProjectDocument[] {
  const kept = documents.filter((doc) => !isInside(doc.path, path))
  return kept.length === documents.length ? documents : kept
}

/**
 * Keep the document lists and the mount tree current for every tab with one subscription. Returns the unsubscribe.
 *
 * - A change to a project's file list, on any device, reads that project's documents and all documents again
 * - A deleted project lists nothing, all documents read again
 * - A note created or moved on any device reads the notes again
 * - An architecture source created or moved on any device reads its project's list again, a deleted one leaves it
 * - A file deleted here or elsewhere leaves every list at once, a list reads again when the file lay in one of its folder entries
 * - A file created inside a folder entry or a mount, or a mounted file renamed, reads the lists that show it again
 */
export function watchDocumentLists(): () => void {
  const stops = [
    // This device's own changes too, so no list depends on the code that changed it to read it again.
    subscribeToChanges((event) => {
      if (event.resource_kind !== "document") return
      const key = event.resource_key
      const [, project, catalog] = PROJECT_KEY.exec(key) ?? []
      if (project && event.operation === "delete") {
        forgetProjectDocuments(project)
      } else if (project && catalog) {
        projectDocumentsResource(project).invalidate()
        allDocumentsResource.invalidate()
      } else if (key.startsWith(NOTE_PREFIX)) {
        const listed = notesResource.get()?.some((note) => note.path === key)
        if (event.operation === "move" || (event.operation === "write" && !listed)) notesResource.invalidate()
      } else if (isArchitecturePath(key)) {
        // The catalog entry never names architecture sources, the project's list holds them all the same.
        const list = projectDocumentsResource(architectureSlug(key))
        const documents = list.get()
        if (!documents) return
        if (event.operation === "delete") {
          const kept = withoutRemoved(documents, key)
          if (kept !== documents) list.set(kept)
        } else if (!documents.some((doc) => doc.path === key)) {
          list.invalidate()
        }
      }
    }, { ownDevice: true }),
    onFileRemoval((path) => {
      for (const list of documentLists()) {
        const documents = list.get()
        if (!documents) continue
        const kept = withoutRemoved(documents, path)
        if (kept !== documents) list.set(kept)
        if (kept.some((doc) => doc.children && path !== doc.path && isInside(path, doc.path))) list.invalidate()
      }
    }),
    onFileChange(() => true, (path) => {
      for (const list of [...projectDocumentsResource.members(), ...rootTreeResource.members()]) {
        if (isNewFileIn(list.get() ?? [], path)) list.invalidate()
      }
    }),
    onFileMove(({ from }) => {
      if (from.startsWith("/")) for (const tree of rootTreeResource.members()) tree.invalidate()
    }),
  ]
  return () => stops.forEach((stop) => stop())
}
