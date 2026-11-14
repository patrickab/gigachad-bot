import { render } from "@testing-library/react"
import { expect, it } from "vitest"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"
import { fileViewerRawUrl } from "@/lib/backend/urls"

const CONTENT = `<img src="images/sdf.png" alt="sdf"/> ![mesh](./mesh.png) ![remote](https://example.com/a.png)`

it("loads relative images from the document's folder through the file API, other images as they are", () => {
  const { container } = render(<LaTeXMarkdown content={CONTENT} imageDirectory="/home/me/nb" />)
  const sources = [...container.querySelectorAll("img")].map((image) => image.getAttribute("src"))

  expect(sources).toEqual([fileViewerRawUrl("/home/me/nb/images/sdf.png"), fileViewerRawUrl("/home/me/nb/./mesh.png"), "https://example.com/a.png"])
})
