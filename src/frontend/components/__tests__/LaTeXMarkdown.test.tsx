import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { LaTeXMarkdown } from "@/components/LaTeXMarkdown"

describe("LaTeXMarkdown", () => {
  it("renders a matrix when a model omits only its opening display delimiter", () => {
    const content = String.raw`For any two vectors, a \odot b = \operatorname{diag}(b)a.

=\begin{pmatrix}\tilde g_1 \\ \tilde g_2\end{pmatrix}$$

This diagonal matrix is the filter.`

    const { container } = render(<LaTeXMarkdown content={content} />)

    expect(container.querySelector(".katex-display")).toBeInTheDocument()
    expect(screen.getByText("This diagonal matrix is the filter.")).toBeInTheDocument()
  })
})
