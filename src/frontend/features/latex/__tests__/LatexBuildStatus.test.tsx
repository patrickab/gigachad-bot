import { afterEach, describe, expect, it } from "vitest"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import type { LatexBuild } from "@/lib/domain/latex"
import { LatexBuildStatus } from "../LatexBuildStatus"
import type { LatexBuildState } from "../useLatexBuild"

const FOLDER = "/home/me/thesis"
const SHOWN = `${FOLDER}/main.tex`

const failed = (errors: LatexBuild["errors"]): LatexBuildState => ({
  building: false,
  failure: null,
  pdfUrl: null,
  pdfPath: null,
  result: { ok: false, pdf_path: null, duration_ms: 1, errors, log_tail: "" },
})

afterEach(cleanup)

describe("LatexBuildStatus", () => {
  it("names the file of an error that is not the shown one, relative to the project folder", () => {
    render(<LatexBuildStatus path={SHOWN} folder={FOLDER} build={failed([
      { file: SHOWN, line: 3, message: "Undefined control sequence." },
      { file: `${FOLDER}/chapters/intro.tex`, line: 12, message: "Missing $ inserted." },
      { file: "/usr/share/texmf/article.cls", line: null, message: "Class error." },
      { file: null, line: null, message: "Emergency stop." },
    ])} />)
    fireEvent.click(screen.getByRole("button", { name: /Build failed/ }))

    const locationOf = (message: string) => screen.getByText(message).querySelector("span")?.textContent ?? null
    expect(locationOf("Undefined control sequence.")).toBe("line 3")
    expect(locationOf("Missing $ inserted.")).toBe("chapters/intro.tex line 12")
    expect(locationOf("Class error.")).toBe("/usr/share/texmf/article.cls")
    expect(locationOf("Emergency stop.")).toBeNull()
  })
})
