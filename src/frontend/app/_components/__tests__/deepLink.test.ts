import { describe, expect, it } from "vitest"
import { parseOpenParam } from "../deepLink"

function encode(payload: unknown): string {
  const bytes = new TextEncoder().encode(JSON.stringify(payload))
  return btoa(String.fromCharCode(...bytes)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "")
}

describe("parseOpenParam", () => {
  it("opens a chat by its history file", () => {
    expect(parseOpenParam(`?open=${encode({ kind: "chat", history_file: "proj/chat.json" })}`)).toEqual({ kind: "chat", historyFile: "proj/chat.json" })
  })

  it("decodes non-ASCII paths and urlsafe characters", () => {
    const path = "projekt/übersicht ~ ?>.md"
    expect(parseOpenParam(`?x=1&open=${encode({ kind: "document", path, project_slug: null })}`)).toEqual({ kind: "item", item: { kind: "document", path, projectSlug: null } })
  })

  it("accepts plain base64 whose plus signs arrived unescaped", () => {
    const path = "a>>>b???c.md"
    const plain = btoa(JSON.stringify({ kind: "document", path }))
    expect(plain).toContain("+")
    expect(parseOpenParam(`?open=${plain}`)).toEqual({ kind: "item", item: { kind: "document", path, projectSlug: null } })
  })

  it.each([
    ["not base64", "?open=%%%"],
    ["not json", `?open=${btoa("nope")}`],
    ["unknown kind", `?open=${encode({ kind: "diff", path: "/x" })}`],
  ])("ignores %s", (_name, search) => {
    expect(parseOpenParam(search)).toBeNull()
  })
})
