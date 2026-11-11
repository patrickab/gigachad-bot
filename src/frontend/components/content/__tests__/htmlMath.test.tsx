import { render } from "@testing-library/react"
import { expect, it } from "vitest"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"

it("renders math inside raw HTML tables, leaving prices and verbatim tags as text", () => {
  const html = String.raw`<table><tr><td> $\chi _ { \mathrm { P Q M } } ^ { 2 } \downarrow$ </td><td>$5 and $10</td><td><code>$a$</code></td></tr></table>`
  const { container } = render(<LaTeXMarkdown content={html} />)
  const cells = container.querySelectorAll("td")

  expect(cells[0].querySelector(".katex annotation")?.textContent).toBe(String.raw`\chi _ { \mathrm { P Q M } } ^ { 2 } \downarrow`)
  expect(cells[1].textContent).toBe("$5 and $10")
  expect(cells[2].querySelector(".katex")).toBeNull()
  expect(cells[2].textContent).toBe("$a$")
})
