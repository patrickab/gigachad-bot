import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { ApiError } from "@/lib/backend/api"
import { getDeviceId } from "@/lib/backend/deviceId"
import { allDocumentsResource, projectDocumentsResource, revalidateOnActivation, rootTreeResource, watchDocumentLists } from "@/lib/backend/resources"
import { subscribeToChanges, type ChangeEvent } from "@/lib/backend/syncStream"
import type { FileTreeNode } from "@/lib/domain/files"
import type { ProjectDocument } from "@/lib/domain/projects"
import type { Resource } from "@/lib/resource"

const api = vi.hoisted(() => ({
  listProjectDocuments: vi.fn<(slug: string, signal?: AbortSignal) => Promise<ProjectDocument[]>>(),
  listAllDocuments: vi.fn<() => Promise<ProjectDocument[]>>(),
  fileRootTree: vi.fn<() => Promise<{ enabled: boolean; tree: FileTreeNode[] }>>(),
}))
vi.mock("@/lib/backend/api", async (importOriginal) => ({ ...await importOriginal<typeof import("@/lib/backend/api")>(), ...api }))

/** The change feed's connection, events go out through the newest one. */
class FakeEventSource {
  static instances: FakeEventSource[] = []
  readonly handlers = new Map<string, (message: MessageEvent<string>) => void>()
  onerror: (() => void) | null = null

  constructor(readonly url: string) {
    FakeEventSource.instances.push(this)
  }

  addEventListener(name: string, handler: (message: MessageEvent<string>) => void): void {
    this.handlers.set(name, handler)
  }

  emit(name: string, data: unknown): void {
    this.handlers.get(name)?.({ data: JSON.stringify(data) } as MessageEvent<string>)
  }

  close(): void {}
}

let seq = 1
/** A change event for `key`, from another device unless `deviceId` says otherwise. */
const emit = (key: string, operation: ChangeEvent["operation"], deviceId: string | undefined = "other-device") => {
  seq += 1
  FakeEventSource.instances.at(-1)!.emit("change", { seq, resource_kind: "document", resource_key: key, version: null, operation, device_id: deviceId })
}

const doc = (path: string, children?: FileTreeNode[]): ProjectDocument => ({ path, name: path.split("/").at(-1)!, mime: "text/plain", children })

let stops: (() => void)[] = []

/** `resource` subscribed and loaded, as a mounted tab shows it. */
async function shown<T>(resource: Resource<T>): Promise<Resource<T>> {
  stops.push(resource.subscribe(() => {}))
  await resource.load()
  return resource
}

beforeEach(() => {
  // The device id lives in local storage, which this environment lacks.
  const stored = new Map<string, string>()
  vi.stubGlobal("localStorage", { getItem: (key: string) => stored.get(key) ?? null, setItem: (key: string, value: string) => { stored.set(key, value) } })
  vi.stubGlobal("EventSource", FakeEventSource)
  stops = [watchDocumentLists()]
  FakeEventSource.instances.at(-1)!.emit("cursor", { seq })
})

afterEach(() => {
  stops.forEach((stop) => stop())
  vi.unstubAllGlobals()
  vi.resetAllMocks()
})

describe("watchDocumentLists", () => {
  it("reads a project's list once on this device's own catalog change, which a plain subscriber never gets", async () => {
    api.listProjectDocuments.mockResolvedValueOnce([doc("a.md")]).mockResolvedValueOnce([doc("a.md"), doc("b.md")])
    api.listAllDocuments.mockResolvedValue([])
    const list = await shown(projectDocumentsResource("proj"))
    const plain = vi.fn()
    stops.push(subscribeToChanges(plain))

    emit("chat/proj/project.json", "write", getDeviceId())
    await vi.waitFor(() => expect(list.get()).toEqual([doc("a.md"), doc("b.md")]))
    expect(api.listProjectDocuments).toHaveBeenCalledTimes(2)
    expect(plain).not.toHaveBeenCalled()
  })

  it("empties a deleted project's list without reading it and reads all documents again", async () => {
    api.listProjectDocuments.mockResolvedValue([doc("a.md")])
    api.listAllDocuments.mockResolvedValueOnce([doc("a.md")]).mockResolvedValueOnce([])
    const list = await shown(projectDocumentsResource("proj"))
    const all = await shown(allDocumentsResource)

    emit("chat/proj", "delete")
    expect(list.get()).toEqual([])
    await vi.waitFor(() => expect(all.get()).toEqual([]))
    expect(api.listProjectDocuments).toHaveBeenCalledOnce()
  })

  it("reads a project's list again for a new architecture source and drops a deleted one", async () => {
    api.listProjectDocuments.mockResolvedValueOnce([doc("graph/proj/system.c4")]).mockResolvedValueOnce([doc("graph/proj/system.c4"), doc("graph/proj/api.c4")])
    const list = await shown(projectDocumentsResource("proj"))

    emit("graph/proj/system.c4", "write")
    emit("graph/proj/.likec4/layout.json", "write", getDeviceId())
    expect(api.listProjectDocuments).toHaveBeenCalledOnce()
    emit("graph/proj/api.c4", "write", getDeviceId())
    await vi.waitFor(() => expect(list.get()).toHaveLength(2))

    emit("graph/proj/system.c4", "delete", getDeviceId())
    expect(list.get()).toEqual([doc("graph/proj/api.c4")])
  })
})

describe("projectDocumentsResource", () => {
  it("lists nothing for a project that no longer exists", async () => {
    api.listProjectDocuments.mockRejectedValue(new ApiError("Project not found", 404))
    await expect(projectDocumentsResource("gone").load()).resolves.toEqual([])
  })
})

describe("revalidateOnActivation", () => {
  it("reads the mount tree and the project lists that show a folder entry again, once for every tab calling it", async () => {
    api.listProjectDocuments.mockImplementation(async (slug) => slug === "latex" ? [doc("/data/paper", [])] : [doc("a.md")])
    api.fileRootTree.mockResolvedValue({ enabled: true, tree: [] })
    await shown(projectDocumentsResource("plain"))
    await shown(projectDocumentsResource("latex"))
    await shown(rootTreeResource("/source"))
    vi.clearAllMocks()

    revalidateOnActivation()
    revalidateOnActivation()
    expect(api.listProjectDocuments).toHaveBeenCalledExactlyOnceWith("latex", expect.any(AbortSignal))
    expect(api.fileRootTree).toHaveBeenCalledOnce()
  })
})
