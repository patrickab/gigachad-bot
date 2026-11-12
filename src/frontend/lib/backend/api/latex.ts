// LaTeX build, SyncTeX and template routes.
import { toQuery } from "../urls"
import { post, request } from "./http"
import type { LatexBuild, LatexTemplates, SynctexClickedSource, SynctexSource, SynctexView, TemplateSetInfo, TexProject } from "@/lib/domain/latex"

/** Compile the `.tex` file at *path*. A TeX error resolves with `ok: false`, a missing TeX install rejects with ApiError 503. */
export function buildLatex(path: string): Promise<LatexBuild> {
  return post<LatexBuild>("/latex/build", { path })
}

/** The LaTeX project of the file at absolute *path*: its folder, root `.tex` and, when no root was found, the candidates. */
export function loadTexProject(path: string): Promise<TexProject> {
  return request<TexProject>(`/latex/project${toQuery({ path })}`)
}

/** The source line and column SyncTeX finds for the spot `x`, `y` (PDF points from the top-left corner) on 1-based *page* of the PDF the root at absolute *root* built. The double-clicked *word* picks its own line of a beamer frame, whose whole body SyncTeX puts on `\end{frame}`, then the clicked character's 0-based *offset* in it and the word's 0-based *occurrence* among equal words of its rendered line give the column. Rejects with ApiError 404 when there is none, 503 without `synctex`. */
export function findSynctexSource(root: string, page: number, x: number, y: number, word = "", offset = 0, occurrence = 0): Promise<SynctexClickedSource> {
  return post<SynctexClickedSource>("/latex/synctex", { path: root, page, x, y, ...(word ? { word, offset, occurrence } : {}) })
}

/** The PDF spot SyncTeX finds for 1-based *line* of the source *file* (absolute) in the PDF the root at absolute *root* built. Rejects with ApiError 404 when there is none, 403 for a file outside the roots, 503 without `synctex`. */
export function findSynctexView(root: string, file: string, line: number): Promise<SynctexView> {
  return post<SynctexView>("/latex/synctex/view", { path: root, file, line })
}

/** The source line after the slide on 1-based *page* of the PDF the root at absolute *root* built, where Neovim's cursor makes "Add slide" put the next frame: the slide's `\end{frame}`, or the heading of a theme's chapter slide. Rejects with ApiError 404 when there is none, 403 for a file outside the roots. */
export function findSlideEnd(root: string, page: number): Promise<SynctexSource> {
  return post<SynctexSource>("/latex/synctex/slide", { path: root, page })
}

/** Make *root* the root of the `.tex` at *path* by writing its `% !TEX root` line. Resolves to the project. */
export function chooseTexRoot(path: string, root: string): Promise<TexProject> {
  return post<TexProject>("/latex/root", { path, root })
}

/** Create file *name* in LaTeX project *folder*, a `.tex` starts with its `% !TEX root` line. Resolves to its path. */
export async function createTexProjectFile(folder: string, name: string): Promise<string> {
  return (await post<{ path: string }>("/latex/file", { folder, name })).path
}

/** Turn project *slug*'s workbench `.tex` at *path* into its LaTeX project folder entry. Resolves to the folder. */
export async function promoteToTexProject(slug: string, path: string): Promise<string> {
  return (await post<{ folder: string }>("/workbench/promote", { slug, path })).folder
}

/** The template set of the LaTeX root at absolute *root*, `theme` null when it is not a presentation. */
export function loadLatexTemplates(root: string): Promise<LatexTemplates> {
  return request<LatexTemplates>(`/latex/templates${toQuery({ root })}`)
}

/** The template sets a new presentation can start from. */
export function loadTemplateSets(): Promise<TemplateSetInfo[]> {
  return request<TemplateSetInfo[]>("/latex/template-sets")
}
