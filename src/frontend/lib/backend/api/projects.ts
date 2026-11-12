// Project, kanban card and project tab routes under /projects.
import { saveChatHistory } from "./chat"
import { del, patch, post, put, request } from "./http"
import { parseHistoryFile } from "@/lib/domain/historyFile"
import type { KanbanCard, ProjectData, ProjectListItem } from "@/lib/domain/projects"
import type { Message, Usage } from "@/lib/domain/types"

export async function listProjects(signal?: AbortSignal): Promise<ProjectListItem[]> {
  const data = await request<{ projects: ProjectListItem[] }>("/projects", { signal })
  return data.projects
}

const projectPath = (slug: string) => `/projects/${encodeURIComponent(slug)}`

export async function loadProject(name: string): Promise<ProjectData> {
  return request<ProjectData>(projectPath(name))
}

export async function createProject(name: string): Promise<ProjectData> {
  return post<ProjectData>("/projects", { name })
}

export async function deleteProject(name: string): Promise<void> {
  await del(projectPath(name))
}

export async function setProjectArchived(slug: string, archived: boolean): Promise<void> {
  await patch(projectPath(slug), { archived })
}

export function renameProject(slug: string, name: string): Promise<{ name: string; slug: string; archived: boolean }> {
  return patch(projectPath(slug), { name })
}

export async function addProjectCard(name: string, title: string, description: string = "", state: string = "backlog"): Promise<KanbanCard> {
  return post<KanbanCard>(`${projectPath(name)}/cards`, { title, description, state })
}

export async function moveProjectCard(name: string, cardId: string, state: string): Promise<KanbanCard> {
  return patch<KanbanCard>(`${projectPath(name)}/cards/${cardId}`, { state })
}

export async function deleteProjectCard(name: string, cardId: string): Promise<void> {
  await del(`${projectPath(name)}/cards/${cardId}`)
}

/** Omitting a field leaves it unchanged server-side; passing null clears it. */
interface SaveProjectTabOptions {
  chatId?: string
  tabName?: string
  title?: string
  usage?: Usage
}

export async function saveProjectTab(name: string, filename: string, messages: Message[], opts: SaveProjectTabOptions = {}): Promise<{ status: string }> {
  const body: Record<string, unknown> = { filename, messages, chat_id: opts.chatId ?? null, tab_name: opts.tabName ?? null, title: opts.title ?? null, usage: opts.usage ?? null }
  return put(`${projectPath(name)}/tabs/${encodeURIComponent(filename)}`, body)
}

/** Save a chat to `historyFile`: through its project's tab when it belongs to `projectSlug`, as a plain history otherwise. */
export async function saveChat(historyFile: string, projectSlug: string | null, messages: Message[], opts: { chatId: string; title: string; tabName?: string; usage?: Usage }): Promise<void> {
  const { slug, filename } = parseHistoryFile(historyFile)
  const { tabName, ...historyOpts } = opts
  if (slug && slug === projectSlug) await saveProjectTab(slug, filename, messages, { ...historyOpts, tabName })
  else await saveChatHistory(historyFile, messages, historyOpts)
}
