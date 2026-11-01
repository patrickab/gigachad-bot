import { describe, expect, it } from "vitest"
import { opensInWorkbench, relocatedWorkbenchPath, workbenchKindOf } from "../workbench"

describe("workbenchKindOf", () => {
  it("maps .tex to latex and .md to markdown, ignoring case", () => {
    expect(workbenchKindOf("/home/me/thesis/main.tex")).toBe("latex")
    expect(workbenchKindOf("project/p/document/Notes.MD")).toBe("markdown")
  })

  it("has no kind for other files", () => {
    expect(workbenchKindOf("/home/me/paper.pdf")).toBeNull()
    expect(workbenchKindOf("/home/me/.md")).toBeNull()
    expect(workbenchKindOf("README")).toBeNull()
  })
})

describe("opensInWorkbench", () => {
  it("opens project documents and disk files of a workbench kind", () => {
    expect(opensInWorkbench("project/p/document/plan.md")).toBe(true)
    expect(opensInWorkbench("/home/me/thesis/main.tex")).toBe(true)
  })

  it("keeps notes and other files out", () => {
    expect(opensInWorkbench("note/ideas.md")).toBe(false)
    expect(opensInWorkbench("note/paper.tex")).toBe(false)
    expect(opensInWorkbench("project/p/document/data.csv")).toBe(false)
  })
})

describe("relocatedWorkbenchPath", () => {
  it("renames the project folder of a workbench file", () => {
    expect(relocatedWorkbenchPath("/data/workbench/latex/old/thesis/thesis.tex", "old", "new")).toBe("/data/workbench/latex/new/thesis/thesis.tex")
    expect(relocatedWorkbenchPath("/data/workbench/markdown/old/plan/plan.md", "old", "new")).toBe("/data/workbench/markdown/new/plan/plan.md")
  })

  it("leaves other projects, other kinds and other shapes alone", () => {
    expect(relocatedWorkbenchPath("/data/workbench/latex/other/thesis/thesis.tex", "old", "new")).toBe("/data/workbench/latex/other/thesis/thesis.tex")
    expect(relocatedWorkbenchPath("/data/workbench/pdf/old/thesis/thesis.tex", "old", "new")).toBe("/data/workbench/pdf/old/thesis/thesis.tex")
    expect(relocatedWorkbenchPath("/data/workbench/latex/old/thesis.tex", "old", "new")).toBe("/data/workbench/latex/old/thesis.tex")
    expect(relocatedWorkbenchPath("project/old/document/plan.md", "old", "new")).toBe("project/old/document/plan.md")
  })
})
