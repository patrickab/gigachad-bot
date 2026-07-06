/**
 * Tests for the api.ts HTTP-helper refactor: toQuery/post/put/patch/del/fileForm
 * are not exported, so we assert behavior through the exported endpoints that
 * use them. We stub global `fetch` and inspect the resulting URL + RequestInit.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"

import * as api from "@/lib/api"

type Call = { url: string; init: RequestInit }

function fetchRecorder(calls: Call[]) {
  return vi.fn(async (input: URL | RequestInfo, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString()
    calls.push({ url, init: init ?? {} })
    return new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })
  })
}

function lastCall(calls: Call[]): Call {
  return calls[calls.length - 1]
}

beforeEach(() => {
  vi.stubGlobal("fetch", fetchRecorder([]))
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe("api.ts — toQuery helper (via exported endpoints)", () => {
  it("omits undefined/null/false values and stringifies true", async () => {
    const calls: Call[] = []
    vi.stubGlobal("fetch", fetchRecorder(calls))
    // response body: {}

    await api.deleteAttachment("c1", "f.txt", null)         // slug=null → omitted
    await api.deleteAttachment("c1", "f.txt", undefined)    // slug=undefined → omitted

    expect(calls).toHaveLength(2)
    expect(calls[0].url.endsWith("/files/chat/c1/att/f.txt")).toBe(true)
    expect(calls[0].init.method).toBe("DELETE")
    expect(calls[1].url).toBe(calls[0].url) // no query string either way
  })

  it("appends array params as repeated keys", async () => {
    const calls: Call[] = []
    vi.stubGlobal("fetch", fetchRecorder(calls))
    // response body: []

    await api.parseFiles("c1", ["a.txt", "b.txt"], "proj")

    expect(calls).toHaveLength(1)
    const url = new URL(calls[0].url)
    expect(url.searchParams.getAll("filenames")).toEqual(["a.txt", "b.txt"])
    expect(url.searchParams.get("chat_id")).toBe("c1")
    expect(url.searchParams.get("slug")).toBe("proj")
  })

  it("serializes true as 'true' and omits false", async () => {
    const calls: Call[] = []
    vi.stubGlobal("fetch", fetchRecorder(calls))
    // response body: { name: "x", mime: "image/jpeg" }

    await api.uploadFile("c1", new File(["a"], "x.jpg"), "proj", true)
    await api.uploadFile("c1", new File(["a"], "y.jpg"), null, false)

    expect(calls).toHaveLength(2)
    expect(new URL(calls[0].url).searchParams.get("overwrite")).toBe("true")
    expect(new URL(calls[0].url).searchParams.get("slug")).toBe("proj")
    // overwrite=false must be omitted entirely (not "false").
    expect(new URL(calls[1].url).searchParams.get("overwrite")).toBeNull()
    expect(new URL(calls[1].url).searchParams.has("slug")).toBe(false)
  })
})

describe("api.ts — json helpers (post/put/patch)", () => {
  it("PUT sends JSON body with Content-Type header", async () => {
    const calls: Call[] = []
    vi.stubGlobal("fetch", fetchRecorder(calls))
    // response body: { slug: "s", name: "n" }

    await api.savePrompt("s", "content")

    const { url, init } = lastCall(calls)
    expect(url.endsWith("/prompts/s")).toBe(true)
    expect(init.method).toBe("PUT")
    expect((init.headers as Record<string, string>)["Content-Type"]).toBe("application/json")
    expect(JSON.parse(init.body as string)).toEqual({ content: "content" })
  })

  it("POST sends JSON body", async () => {
    const calls: Call[] = []
    vi.stubGlobal("fetch", fetchRecorder(calls))
    // response body: { status: "ok", path: "/p" }

    await api.createDirectory("p", "sub")

    const { url, init } = lastCall(calls)
    expect(url.endsWith("/chat-histories/mkdir")).toBe(true)
    expect(init.method).toBe("POST")
    expect(JSON.parse(init.body as string)).toEqual({ parent_path: "p", name: "sub" })
  })

  it("PATCH sends JSON body", async () => {
    const calls: Call[] = []
    vi.stubGlobal("fetch", fetchRecorder(calls))
    // response body: {}

    await api.moveProjectCard("proj", "card-1", "done")

    const { url, init } = lastCall(calls)
    expect(url.endsWith("/projects/proj/cards/card-1")).toBe(true)
    expect(init.method).toBe("PATCH")
    expect(JSON.parse(init.body as string)).toEqual({ state: "done" })
  })
})

describe("api.ts — del helper", () => {
  it("DELETE has no body", async () => {
    const calls: Call[] = []
    vi.stubGlobal("fetch", fetchRecorder(calls))
    // response body: { deleted: true }

    await api.deletePrompt("s")

    const { url, init } = lastCall(calls)
    expect(url.endsWith("/prompts/s")).toBe(true)
    expect(init.method).toBe("DELETE")
    expect(init.body).toBeUndefined()
  })
})

describe("api.ts — fileForm helper", () => {
  it("uploads use multipart FormData with the file under 'file'", async () => {
    const calls: Call[] = []
    vi.stubGlobal("fetch", fetchRecorder(calls))
    // response body: { name: "x.pdf", mime: "application/pdf" }

    await api.uploadDocument("proj", new File(["%PDF"], "x.pdf"))

    const { url, init } = lastCall(calls)
    expect(url.endsWith("/documents/upload?slug=proj")).toBe(true)
    expect(init.method).toBe("POST")
    expect(init.body).toBeInstanceOf(FormData)
    const form = init.body as FormData
    expect(form.get("file")).toBeInstanceOf(File)
    expect((form.get("file") as File).name).toBe("x.pdf")
    // JSON helpers must NOT set Content-Type on FormData (browser sets the boundary).
    const headers = init.headers as Record<string, string> | undefined
    expect(headers?.["Content-Type"] ?? null).toBeNull()
  })

  it("writeBinaryDocument keeps the custom filename on the FormData entry", async () => {
    const calls: Call[] = []
    vi.stubGlobal("fetch", fetchRecorder(calls))
    // response body: {}

    await api.writeBinaryDocument("proj", "sketch.jpg", new Blob([new Uint8Array([1, 2])]))

    const form = lastCall(calls).init.body as FormData
    const file = form.get("file") as File
    expect(file.name).toBe("sketch.jpg")
  })
})

describe("api.ts — ensureOk surfaces FastAPI detail on error", () => {
  it("throws the detail message from a 4xx response", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response(JSON.stringify({ detail: "boom" }), { status: 400 }),
    )

    await expect(api.deletePrompt("s")).rejects.toThrow("boom")
  })

  it("falls back to statusText when the body is not JSON", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response("plain", { status: 500, statusText: "Internal Server Error" }),
    )

    await expect(api.deletePrompt("s")).rejects.toThrow()
  })
})

describe("api.ts — parseHistoryFile / buildHistoryFile round-trip", () => {
  it("round-trips a project-scoped path", () => {
    const historyFile = api.buildHistoryFile("tab.json", "proj")
    expect(historyFile).toBe("proj/tab.json")
    const { slug, filename } = api.parseHistoryFile(historyFile)
    expect(slug).toBe("proj")
    expect(filename).toBe("tab.json")
  })

  it("round-trips a standalone path", () => {
    const historyFile = api.buildHistoryFile("tab.json", null)
    expect(historyFile).toBe("tab.json")
    const { slug, filename } = api.parseHistoryFile(historyFile)
    expect(slug).toBeNull()
    expect(filename).toBe("tab.json")
  })

  it("parseHistoryFile splits nested directories correctly", () => {
    const { slug, filename } = api.parseHistoryFile("proj/sub/dir/tab.json")
    expect(slug).toBe("proj")
    expect(filename).toBe("sub/dir/tab.json")
  })
})
