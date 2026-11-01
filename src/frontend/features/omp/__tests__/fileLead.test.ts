import { describe, expect, it } from "vitest"
import { fileLead, withoutFileLead } from "../fileLead"

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

describe("withoutFileLead", () => {
  it("drops the file line put before a first message, so the title is the request", () => {
    expect(withoutFileLead("File: proj/plan/plan.md\n\nTighten the intro")).toBe("Tighten the intro")
    expect(withoutFileLead("File: proj/plan/plan.md")).toBe("")
    expect(withoutFileLead("Profile: keep this")).toBe("Profile: keep this")
  })
})
