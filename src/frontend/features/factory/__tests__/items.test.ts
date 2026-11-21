import { describe, it, expect } from "vitest"
import { factoryItem, parseFactoryPath } from "@/features/factory/items"

describe("factory items", () => {
  it("names a repo's board by its path", () => {
    const item = factoryItem("/home/me/git/gigachad")
    expect(item).toEqual({ kind: "factory", path: "/home/me/git/gigachad", projectSlug: null })
    expect(parseFactoryPath(item.path)).toEqual({ repo: "/home/me/git/gigachad", taskId: null })
  })

  it("names a task by its id and its repo's path, whatever the path ends in", () => {
    const item = factoryItem("/home/me/git/a@b/abcdef12", "0123456789abcdef0123456789abcdef")
    expect(parseFactoryPath(item.path)).toEqual({ repo: "/home/me/git/a@b/abcdef12", taskId: "0123456789abcdef0123456789abcdef" })
  })
})
