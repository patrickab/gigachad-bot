import { describe, expect, it } from "vitest"
import type { WorkspaceItem } from "@/contexts/TabContext"
import { nextTab, relocatedTab, renamedTab, texBufferShown, texProjectTab } from "../tabs"

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
    const lookAlike = tabWith({ kind: "workbench", path: "/home/me/notes/latex/old/thesis.tex", projectSlug: null, rootFile: true })
    expect(relocatedTab(lookAlike, "old", "new")).toBe(lookAlike)
  })
})

describe("renamedTab", () => {
  const OLD = "/data/workbench/latex/proj/thesis/thesis.tex"
  const NEW = "/data/workbench/latex/proj/thesis/paper.tex"

  it("moves a workbench file and remembers the path its editor session opened with", () => {
    const tab = tabWith({ kind: "workbench", path: OLD, projectSlug: "proj" })
    const renamed = renamedTab(tab, OLD, NEW)
    expect(renamed.workspaceItem).toEqual({ kind: "workbench", path: NEW, projectSlug: "proj", sessionKey: OLD })
    expect(renamedTab(renamed, NEW, "/data/workbench/latex/proj/thesis/final.tex").workspaceItem?.sessionKey).toBe(OLD)
  })

  it("moves a stored document and a mounted file without a session key", () => {
    const stored = tabWith({ kind: "document", path: "project/proj/document/notes.md", projectSlug: "proj" })
    expect(renamedTab(stored, "project/proj/document/notes.md", "project/proj/document/ideas.md").workspaceItem).toEqual({ kind: "document", path: "project/proj/document/ideas.md", projectSlug: "proj" })
    const mounted = tabWith({ kind: "document", path: "/src/a.md", projectSlug: null, rootFile: true })
    expect(renamedTab(mounted, "/src/a.md", "/src/b.md").workspaceItem).toEqual({ kind: "document", path: "/src/b.md", projectSlug: null, rootFile: true })
  })

  it("leaves tabs of other files and kinds alone", () => {
    const other = tabWith({ kind: "document", path: "project/proj/document/other.md", projectSlug: "proj" })
    expect(renamedTab(other, "project/proj/document/notes.md", "project/proj/document/ideas.md")).toBe(other)
    const learn = tabWith({ kind: "learn", path: OLD, projectSlug: null, learn: { kind: "session", sessionId: "s" } })
    expect(renamedTab(learn, OLD, NEW)).toBe(learn)
  })

  it("drops the session key when the project folder moves", () => {
    const renamed = renamedTab(tabWith({ kind: "workbench", path: OLD, projectSlug: "proj" }), OLD, NEW)
    expect(relocatedTab(renamed, "proj", "next").workspaceItem?.sessionKey).toBeUndefined()
  })
})

describe("texProjectTab", () => {
  const FOLDER = "/repo/paper"
  const project = tabWith({ kind: "workbench", path: `${FOLDER}/main.tex`, projectSlug: null, texFolder: FOLDER })
  const open = (path: string): WorkspaceItem => ({ kind: "workbench", path, projectSlug: null })

  it("finds the workbench tab whose project folder holds a LaTeX file", () => {
    expect(texProjectTab([nextTab(), project], open(`${FOLDER}/refs.bib`), "")).toBe(project)
    expect(texProjectTab([project], open(`${FOLDER}/chapters/intro.tex`), "")).toBe(project)
  })

  it("prefers the active tab among several showing the project", () => {
    const second = tabWith({ kind: "workbench", path: `${FOLDER}/main.tex`, projectSlug: null, texFolder: FOLDER })
    expect(texProjectTab([project, second], open(`${FOLDER}/refs.bib`), second.id)).toBe(second)
  })

  it("ignores files outside the folder, look-alike folders, other kinds and tabs without a folder", () => {
    expect(texProjectTab([project], open("/repo/paper-2/refs.bib"), "")).toBeUndefined()
    expect(texProjectTab([project], open(`${FOLDER}/notes.md`), "")).toBeUndefined()
    expect(texProjectTab([project], { kind: "document", path: `${FOLDER}/refs.bib`, projectSlug: null }, "")).toBeUndefined()
    const unresolved = tabWith({ kind: "workbench", path: `${FOLDER}/main.tex`, projectSlug: null })
    expect(texProjectTab([unresolved], open(`${FOLDER}/refs.bib`), "")).toBeUndefined()
  })
})

describe("texBufferShown", () => {
  it("moves the item to the file and keeps its session, chat and folder", () => {
    const tab = { ...tabWith({ kind: "workbench", path: "/repo/paper/main.tex", projectSlug: "proj", texFolder: "/repo/paper", omp: { kind: "new", cwd: "/data" } }), appMode: "chat" as const }
    const shown = texBufferShown(tab, "/repo/paper/refs.bib")
    expect(shown.appMode).toBe("workbench")
    expect(shown.workspaceItem).toEqual({
      kind: "workbench", path: "/repo/paper/refs.bib", projectSlug: "proj", texFolder: "/repo/paper", omp: { kind: "new", cwd: "/data" }, sessionKey: "/repo/paper/main.tex",
    })
    expect(texBufferShown(shown, "/repo/paper/other.tex").workspaceItem?.sessionKey).toBe("/repo/paper/main.tex")
  })
})
