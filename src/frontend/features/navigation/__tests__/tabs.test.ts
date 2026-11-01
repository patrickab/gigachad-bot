import { describe, expect, it } from "vitest"
import type { WorkspaceItem } from "@/contexts/TabContext"
import { nextTab, relocatedTab } from "../tabs"

const tabWith = (item: WorkspaceItem) => ({ ...nextTab(), appMode: item.kind === "workbench" ? "workbench" as const : "document" as const, projectSlug: item.projectSlug, workspaceItem: item })

describe("relocatedTab", () => {
  it("moves an open workbench file with its project folder", () => {
    const tab = tabWith({ kind: "workbench", path: "/data/workbench/latex/old/thesis/thesis.tex", projectSlug: "old" })
    const moved = relocatedTab(tab, "old", "new")
    expect(moved.projectSlug).toBe("new")
    expect(moved.workspaceItem).toEqual({ kind: "workbench", path: "/data/workbench/latex/new/thesis/thesis.tex", projectSlug: "new" })
  })

  it("moves a workbench file opened outside the project", () => {
    const tab = tabWith({ kind: "workbench", path: "/data/workbench/markdown/old/plan/plan.md", projectSlug: null, rootFile: true })
    expect(relocatedTab(tab, "old", "new").workspaceItem?.path).toBe("/data/workbench/markdown/new/plan/plan.md")
  })

  it("still moves a stored key that has not been prepared yet", () => {
    const tab = tabWith({ kind: "workbench", path: "project/old/document/plan.md", projectSlug: "old" })
    expect(relocatedTab(tab, "old", "new").workspaceItem?.path).toBe("project/new/document/plan.md")
  })

  it("leaves other projects' files and look-alike paths alone", () => {
    const other = tabWith({ kind: "workbench", path: "/data/workbench/latex/other/thesis/thesis.tex", projectSlug: "other" })
    expect(relocatedTab(other, "old", "new")).toBe(other)
    const lookAlike = tabWith({ kind: "workbench", path: "/home/me/workbench/latex/old/thesis.tex", projectSlug: null, rootFile: true })
    expect(relocatedTab(lookAlike, "old", "new")).toBe(lookAlike)
  })
})
