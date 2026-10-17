import { describe, it, expect } from "vitest"

import { buildHistoryFile, parseHistoryFile } from "@/lib/historyFile"

describe("parseHistoryFile / buildHistoryFile round-trip", () => {
  it("round-trips a project-scoped path", () => {
    const historyFile = buildHistoryFile("tab.json", "proj")
    expect(historyFile).toBe("proj/tab.json")
    const { slug, filename } = parseHistoryFile(historyFile)
    expect(slug).toBe("proj")
    expect(filename).toBe("tab.json")
  })

  it("round-trips a standalone path", () => {
    const historyFile = buildHistoryFile("tab.json", null)
    expect(historyFile).toBe("tab.json")
    const { slug, filename } = parseHistoryFile(historyFile)
    expect(slug).toBeNull()
    expect(filename).toBe("tab.json")
  })

  it("splits nested directories correctly", () => {
    const { slug, filename } = parseHistoryFile("proj/sub/dir/tab.json")
    expect(slug).toBe("proj")
    expect(filename).toBe("sub/dir/tab.json")
  })
})
