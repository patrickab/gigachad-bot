/**
 * Tests for the api.ts HTTP-helper refactor: toQuery/post/put/patch/del/fileForm
 * are not exported, so we assert behavior through the exported endpoints that
 * use them. We stub global `fetch` and inspect the resulting URL + RequestInit.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"

import * as api from "@/lib/backend/api"

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

    await api.uploadFile("chat-1", new File(["%PDF"], "x.pdf"))

    const { url, init } = lastCall(calls)
    expect(url).toContain("/files/upload?chat_id=chat-1")
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
})

describe("api.ts — saveChat", () => {
  it("saves a chat of the open project through its project tab", async () => {
    const calls: Call[] = []
    vi.stubGlobal("fetch", fetchRecorder(calls))

    await api.saveChat("proj/notes.json", "proj", [], { chatId: "c", tabName: "Tab", title: "notes" })

    const { url, init } = lastCall(calls)
    expect(url).toContain("/projects/proj/tabs/notes.json")
    expect(JSON.parse(init.body as string)).toMatchObject({ filename: "notes.json", tab_name: "Tab", title: "notes" })
  })

  it("saves any other chat as a plain history without a tab name", async () => {
    const calls: Call[] = []
    vi.stubGlobal("fetch", fetchRecorder(calls))

    await api.saveChat("other/notes.json", "proj", [], { chatId: "c", tabName: "Tab", title: "notes" })

    const { url, init } = lastCall(calls)
    expect(url).toContain("/chat-histories/other/notes.json")
    expect(JSON.parse(init.body as string)).not.toHaveProperty("tab_name")
  })
})

describe("api.ts — Jupyter kernels", () => {
  it("lists the kernel environments of a notebook path", async () => {
    const calls: Call[] = []
    vi.stubGlobal("fetch", fetchRecorder(calls))

    await api.fetchKernelSpecs("/work/a b.ipynb")

    expect(new URL(lastCall(calls).url).pathname).toMatch(/\/jupyter\/kernelspecs$/)
    expect(new URL(lastCall(calls).url).searchParams.get("path")).toBe("/work/a b.ipynb")
  })

  it("starts a kernel by posting the path and spec name", async () => {
    const calls: Call[] = []
    vi.stubGlobal("fetch", fetchRecorder(calls))

    await api.startKernel("/work/a.ipynb", "python3")

    expect(new URL(lastCall(calls).url).pathname).toMatch(/\/jupyter\/kernel$/)
    expect(lastCall(calls).init.method).toBe("POST")
    expect(JSON.parse(lastCall(calls).init.body as string)).toEqual({ path: "/work/a.ipynb", name: "python3" })
  })

  it("shuts a kernel down by path and fails with the backend's detail", async () => {
    const calls: Call[] = []
    vi.stubGlobal("fetch", fetchRecorder(calls))
    await api.shutdownKernel("/work/a.ipynb")
    expect(lastCall(calls).init.method).toBe("DELETE")
    expect(new URL(lastCall(calls).url).searchParams.get("path")).toBe("/work/a.ipynb")

    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ detail: "Jupyter is not available" }), { status: 503, headers: { "Content-Type": "application/json" } })))
    await expect(api.startKernel("/work/a.ipynb", "python3")).rejects.toMatchObject({ name: "ApiError", status: 503, message: "Jupyter is not available" })
  })
})

describe("api.ts — an unreachable backend (hot reload, restart)", () => {
  const ok = () => new Response(JSON.stringify({ sessions: [], session: { sessionId: "s", cwd: "/" } }), { status: 200, headers: { "Content-Type": "application/json" } })

  beforeEach(() => { vi.useFakeTimers() })
  afterEach(() => { vi.useRealTimers() })

  it("retries an idempotent request until the backend answers", async () => {
    const fetch = vi.fn().mockRejectedValueOnce(new TypeError("Failed to fetch")).mockRejectedValueOnce(new TypeError("Failed to fetch")).mockResolvedValueOnce(ok())
    vi.stubGlobal("fetch", fetch)
    const sessions = api.fetchPiSessions()
    await vi.runAllTimersAsync()
    await expect(sessions).resolves.toEqual([])
    expect(fetch).toHaveBeenCalledTimes(3)
  })

  it("gives up once the retry window is over", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")))
    const sessions = api.fetchPiSessions()
    const settled = expect(sessions).rejects.toThrow("Failed to fetch")
    await vi.runAllTimersAsync()
    await settled
  })

  it("sends a POST that is not safe to repeat only once", async () => {
    const fetch = vi.fn().mockRejectedValue(new TypeError("Failed to fetch"))
    vi.stubGlobal("fetch", fetch)
    await expect(api.openPiSession("repo", null)).rejects.toThrow("Failed to fetch")
    expect(fetch).toHaveBeenCalledTimes(1)
  })

  it("retries resuming a session file", async () => {
    const fetch = vi.fn().mockRejectedValueOnce(new TypeError("Failed to fetch")).mockResolvedValueOnce(ok())
    vi.stubGlobal("fetch", fetch)
    const session = api.openPiSession("repo", { sessionPath: "/s.jsonl" })
    await vi.runAllTimersAsync()
    await expect(session).resolves.toEqual({ sessionId: "s", cwd: "/" })
    expect(fetch).toHaveBeenCalledTimes(2)
  })

  it("stops retrying when the caller aborts", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")))
    const controller = new AbortController()
    const hits = api.searchPiSessions("x", controller.signal)
    const settled = expect(hits).rejects.toMatchObject({ name: "AbortError" })
    controller.abort()
    await vi.runAllTimersAsync()
    await settled
  })
})
