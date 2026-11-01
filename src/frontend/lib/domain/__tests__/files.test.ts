import { describe, expect, it } from "vitest"
import { isNewFileIn, type FileTreeNode } from "../files"

const file = (path: string): FileTreeNode => ({ name: path.split("/").pop()!, path, type: "file" })
const folder = (path: string, children: FileTreeNode[]): FileTreeNode => ({ name: path.split("/").pop()!, path, type: "folder", children })

const project = { path: "/a/foo", children: [file("/a/foo/main.tex"), folder("/a/foo/imgs", [file("/a/foo/imgs/one.png")])] }

describe("isNewFileIn", () => {
  it("is true for a file under a listed folder that the tree does not list", () => {
    expect(isNewFileIn([project], "/a/foo/chapter.tex")).toBe(true)
    expect(isNewFileIn([project], "/a/foo/imgs/two.png")).toBe(true)
    expect(isNewFileIn([project], "/a/foo/new/deep.png")).toBe(true)
  })

  it("is false for a listed file at any depth", () => {
    expect(isNewFileIn([project], "/a/foo/main.tex")).toBe(false)
    expect(isNewFileIn([project], "/a/foo/imgs/one.png")).toBe(false)
  })

  it("is false for a path outside the folders and for the folder itself", () => {
    expect(isNewFileIn([project], "/a/foobar/x.tex")).toBe(false)
    expect(isNewFileIn([project], "/b/x.tex")).toBe(false)
    expect(isNewFileIn([project], "/a/foo")).toBe(false)
  })

  it("ignores folders without a listed tree and counts an empty one", () => {
    expect(isNewFileIn([{ path: "/a/foo" }, { path: "/a/foo", children: null }], "/a/foo/x.tex")).toBe(false)
    expect(isNewFileIn([{ path: "/a/foo", children: [] }], "/a/foo/x.tex")).toBe(true)
  })

  it("is false when another folder lists the file", () => {
    expect(isNewFileIn([{ path: "/a/foo", children: [] }, { path: "/a/foo/imgs", children: [file("/a/foo/imgs/one.png")] }], "/a/foo/imgs/one.png")).toBe(false)
  })
})
