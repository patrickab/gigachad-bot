import type { Attachment, ProjectDocument, Usage, VaultFile } from "@/lib/types"

/**
 * Centralised mock for `@/lib/api`. Every hook test overrides only the
 * endpoints it exercises; the rest are no-ops returning sensible empties.
 *
 * `vi.mock` is hoisted to the top of the file by vitest, so the factory
 * cannot reference outer-scope variables — we expose mutators on the mock
 * itself (the `__setImpl`/`__impls` pattern) so each test can install a
 * spy for the specific function it cares about.
 */

type AnyFn = (...args: any[]) => any

interface Impl {
  listFileVaultFiles: AnyFn
  attachFileVaultFile: AnyFn
  listProjectDocuments: AnyFn
  listProjectVaultDocuments: AnyFn
  listAllDocuments: AnyFn
  addDocument: AnyFn
  removeDocument: AnyFn
  writeDocument: AnyFn
  uploadDocument: AnyFn
  uploadFile: AnyFn
  attachDocument: AnyFn
  loadFileViewerText: AnyFn
  saveChatHistory: AnyFn
  saveProjectTab: AnyFn
  buildHistoryFile: AnyFn
  parseHistoryFile: AnyFn
  generateMindmap: AnyFn
}

const noop = async () => undefined

const defaults: Impl = {
  listFileVaultFiles: async () => ({ enabled: false, files: [] as VaultFile[] }),
  attachFileVaultFile: async () => ({ name: "att", mime: "text/plain", url: "u", active: true }),
  listProjectDocuments: async () => [] as ProjectDocument[],
  listProjectVaultDocuments: async () => [] as ProjectDocument[],
  listAllDocuments: async () => [] as ProjectDocument[],
  addDocument: noop,
  removeDocument: noop,
  writeDocument: noop,
  uploadDocument: noop,
  uploadFile: async () => ({ name: "u", mime: "image/jpeg", url: "u", active: true }),
  attachDocument: async () => ({ name: "d", mime: "text/plain", url: "u", active: true }),
  loadFileViewerText: async () => "",
  saveChatHistory: noop,
  saveProjectTab: noop,
  buildHistoryFile: (filename: string, slug: string | null) => (slug ? `${slug}/${filename}` : filename),
  parseHistoryFile: (historyFile: string) => {
    const parts = historyFile.split("/")
    if (parts.length > 1) return { slug: parts[0], filename: parts.slice(1).join("/") }
    return { slug: null, filename: historyFile }
  },
  generateMindmap: async () => "# mindmap\n- a\n",
}

const impls: Impl = { ...defaults }

export const __setImpl = <K extends keyof Impl>(key: K, fn: Impl[K]) => { impls[key] = fn }
export const __resetImpls = () => Object.assign(impls, defaults)

export const listFileVaultFiles = (...a: Parameters<Impl["listFileVaultFiles"]>) => impls.listFileVaultFiles(...a)
export const attachFileVaultFile = (...a: Parameters<Impl["attachFileVaultFile"]>) => impls.attachFileVaultFile(...a)
export const listProjectDocuments = (...a: Parameters<Impl["listProjectDocuments"]>) => impls.listProjectDocuments(...a)
export const listProjectVaultDocuments = (...a: Parameters<Impl["listProjectVaultDocuments"]>) => impls.listProjectVaultDocuments(...a)
export const listAllDocuments = (...a: Parameters<Impl["listAllDocuments"]>) => impls.listAllDocuments(...a)
export const addDocument = (...a: Parameters<Impl["addDocument"]>) => impls.addDocument(...a)
export const removeDocument = (...a: Parameters<Impl["removeDocument"]>) => impls.removeDocument(...a)
export const writeDocument = (...a: Parameters<Impl["writeDocument"]>) => impls.writeDocument(...a)
export const uploadDocument = (...a: Parameters<Impl["uploadDocument"]>) => impls.uploadDocument(...a)
export const uploadFile = (...a: Parameters<Impl["uploadFile"]>) => impls.uploadFile(...a)
export const attachDocument = (...a: Parameters<Impl["attachDocument"]>) => impls.attachDocument(...a)
export const loadFileViewerText = (...a: Parameters<Impl["loadFileViewerText"]>) => impls.loadFileViewerText(...a)
export const saveChatHistory = (...a: Parameters<Impl["saveChatHistory"]>) => impls.saveChatHistory(...a)
export const saveProjectTab = (...a: Parameters<Impl["saveProjectTab"]>) => impls.saveProjectTab(...a)
export const buildHistoryFile = (...a: Parameters<Impl["buildHistoryFile"]>) => impls.buildHistoryFile(...a)
export const parseHistoryFile = (...a: Parameters<Impl["parseHistoryFile"]>) => impls.parseHistoryFile(...a)
export const generateMindmap = (...a: Parameters<Impl["generateMindmap"]>) => impls.generateMindmap(...a)

export type { Attachment, Usage }
