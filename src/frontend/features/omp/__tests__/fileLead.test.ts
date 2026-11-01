import { describe, expect, it } from "vitest"
import { fileLead, projectLead, withoutFileLead } from "../fileLead"

describe("fileLead", () => {
  it("names a file inside the session directory relative to it", () => {
    expect(fileLead("/data/workbench/latex/proj/paper/paper.tex", "/data/workbench/latex")).toBe("File: proj/paper/paper.tex")
    expect(fileLead("/data/workbench/markdown/notes/a/a.md", "/data/workbench/markdown/")).toBe("File: notes/a/a.md")
  })

  it("names a file outside the session directory by its absolute path", () => {
    expect(fileLead("/home/me/thesis/main.tex", "/data/workbench/latex")).toBe("File: /home/me/thesis/main.tex")
    expect(fileLead("/data/workbench/latexfoo/x.tex", "/data/workbench/latex")).toBe("File: /data/workbench/latexfoo/x.tex")
  })
})

describe("projectLead", () => {
  it.each([
    ["inside the session directory", "/data/workbench/latex/proj/paper", "/data/workbench/latex/proj/paper/paper.tex", "Project: proj/paper, root: paper.tex"],
    ["outside the session directory", "/home/me/thesis", "/home/me/thesis/main.tex", "Project: /home/me/thesis, root: main.tex"],
    ["beside the session directory with a shared prefix", "/data/workbench/latexfoo/paper", "/data/workbench/latexfoo/paper/main.tex", "Project: /data/workbench/latexfoo/paper, root: main.tex"],
    ["without a root", "/home/me/thesis", null, "Project: /home/me/thesis"],
  ])("names the folder %s", (_name, folder, root, expected) => {
    expect(projectLead(folder, root, "/data/workbench/latex")).toBe(expected)
  })
})

describe("withoutFileLead", () => {
  it("drops the file line put before a first message, so the title is the request", () => {
    expect(withoutFileLead("File: proj/plan/plan.md\n\nTighten the intro")).toBe("Tighten the intro")
    expect(withoutFileLead("File: proj/plan/plan.md")).toBe("")
    expect(withoutFileLead("Profile: keep this")).toBe("Profile: keep this")
  })

  it("drops the project line too", () => {
    expect(withoutFileLead("Project: /home/me/thesis, root: main.tex\n\nFix the bibliography")).toBe("Fix the bibliography")
    expect(withoutFileLead("Project: /home/me/thesis")).toBe("")
    expect(withoutFileLead("Projects: keep this")).toBe("Projects: keep this")
  })
})
