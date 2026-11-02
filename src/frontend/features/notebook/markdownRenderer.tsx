"use client"

import type { IRenderMime } from "@jupyterlab/rendermime-interfaces"
import { Widget } from "@lumino/widgets"
import { createRoot, type Root } from "react-dom/client"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"

/** A Lumino widget whose body is the app's own markdown pipeline (KaTeX, code blocks, diagrams), so notebook text reads like every other document. */
class AppMarkdown extends Widget implements IRenderMime.IRenderer {
  private root: Root | null = null

  constructor() {
    super()
    this.addClass("gc-nb-markdown")
  }

  async renderModel(model: IRenderMime.IMimeModel): Promise<void> {
    const source = String(model.data["text/markdown"] ?? "")
    this.root ??= createRoot(this.node)
    this.root.render(<LaTeXMarkdown content={source} />)
  }

  dispose(): void {
    // Unmount after the current render pass, React forbids unmounting a root while it renders.
    const root = this.root
    this.root = null
    if (root) queueMicrotask(() => root.unmount())
    super.dispose()
  }
}

/** Ranks below the library's own markdown renderer (60), so it takes every `text/markdown` cell and output. */
export const appMarkdownFactory: IRenderMime.IRendererFactory = {
  safe: true,
  mimeTypes: ["text/markdown"],
  defaultRank: 10,
  createRenderer: () => new AppMarkdown(),
}
