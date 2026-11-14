"use client"

import type { IRenderMime } from "@jupyterlab/rendermime-interfaces"
import { Widget } from "@lumino/widgets"
import { createRoot, type Root } from "react-dom/client"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"

/** A Lumino widget whose body is the app's own markdown pipeline (KaTeX, code blocks, diagrams), so notebook text reads like every other document. */
class AppMarkdown extends Widget implements IRenderMime.IRenderer {
  private root: Root | null = null

  constructor(private readonly imageDirectory: string) {
    super()
    this.addClass("gc-nb-markdown")
  }

  async renderModel(model: IRenderMime.IMimeModel): Promise<void> {
    const source = String(model.data["text/markdown"] ?? "")
    this.root ??= createRoot(this.node)
    this.root.render(<LaTeXMarkdown content={source} imageDirectory={this.imageDirectory} />)
  }

  dispose(): void {
    // Unmount after the current render pass, React forbids unmounting a root while it renders.
    const root = this.root
    this.root = null
    if (root) queueMicrotask(() => root.unmount())
    super.dispose()
  }
}

/**
 * Markdown of a notebook in `notebookDirectory`, whose relative image paths (`images/a.png`) resolve against that folder.
 * Ranks below the library's own markdown renderer (60), so it takes every `text/markdown` cell and output.
 */
export function appMarkdownFactory(notebookDirectory: string): IRenderMime.IRendererFactory {
  return {
    safe: true,
    mimeTypes: ["text/markdown"],
    defaultRank: 10,
    createRenderer: () => new AppMarkdown(notebookDirectory),
  }
}
