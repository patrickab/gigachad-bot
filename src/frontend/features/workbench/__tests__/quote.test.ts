import { describe, expect, it } from "vitest"
import { fileLead, quoteSelection } from "../quote"

describe("quoteSelection", () => {
  it("quotes the lines under their file name and range, leaving a blank line to type in", () => {
    const text = quoteSelection("/home/me/thesis/main.tex", { start_line: 13, end_line: 14, text: "\\section{A}\nHello" }, "latex")
    expect(text).toBe("main.tex:13-14\n```latex\n\\section{A}\nHello\n```\n\n")
  })

  it("fences the lines in the given language", () => {
    const text = quoteSelection("notes/todo.md", { start_line: 2, end_line: 2, text: "- item" }, "markdown")
    expect(text).toBe("todo.md:2-2\n```markdown\n- item\n```\n\n")
  })
})

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
