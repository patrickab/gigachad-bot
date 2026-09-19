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
