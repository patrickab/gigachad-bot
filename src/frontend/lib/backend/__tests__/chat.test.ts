import { afterEach, describe, expect, it, vi } from "vitest"

import { listAllChats, searchChats, type ChatListing, type ChatSearchHit } from "@/lib/backend/api"

const listing: ChatListing = { historyFile: "paper/notes.json", projectSlug: "paper", title: "Notes", updatedAt: "2026-01-02T03:04:05+00:00" }

function answer(body: unknown) {
  const calls: { url: string; init: RequestInit }[] = []
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: URL | RequestInfo, init?: RequestInit) => {
      calls.push({ url: input.toString(), init: init ?? {} })
      return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } })
    }),
  )
  return calls
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe("chat list and search", () => {
  it("listAllChats reads every chat from /chat-histories/all", async () => {
    const calls = answer({ chats: [listing] })
    const controller = new AbortController()

    await expect(listAllChats(controller.signal)).resolves.toEqual([listing])

    expect(new URL(calls[0].url).pathname).toMatch(/\/chat-histories\/all$/)
    expect(calls[0].init.signal).toBe(controller.signal)
  })

  it("searchChats sends the query and limit and returns the hits", async () => {
    const hit: ChatSearchHit = { ...listing, matches: [{ qaIndex: 1, role: "assistant", ordinal: 0, snippet: "a & b" }] }
    const calls = answer({ hits: [hit] })

    await expect(searchChats("a & b", 5)).resolves.toEqual([hit])

    const url = new URL(calls[0].url)
    expect(url.pathname).toMatch(/\/chat-histories\/search$/)
    expect(url.searchParams.get("q")).toBe("a & b")
    expect(url.searchParams.get("limit")).toBe("5")
  })

  it("searchChats asks for 20 chats by default", async () => {
    const calls = answer({ hits: [] })

    await searchChats("x")

    expect(new URL(calls[0].url).searchParams.get("limit")).toBe("20")
  })
})
