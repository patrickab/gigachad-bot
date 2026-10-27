export interface ProjectDocument {
  path: string
  /** The name the project shows the file under: its display name, else its file name. */
  name: string
  mime: string
}

export type KanbanColumnId = "backlog" | "doing" | "done"

export interface KanbanCard {
  id: string
  title: string
  description: string
  state: KanbanColumnId
}

export interface ProjectData {
  name: string
  slug: string
  linked_root?: string | null
  kanban: KanbanCard[]
}

interface ProjectTab {
  filename: string
  name: string | null
  title: string | null
}

export interface ProjectListItem {
  name: string
  slug: string
  archived?: boolean
  linked_root?: string | null
  tabs?: ProjectTab[]
  /** Display names the project gives its files, by path; a file shown under its own name has none. */
  file_names?: Record<string, string>
}

/** The display name project `slug` gives its file `path`, undefined when it has none. */
export function displayName(projects: ProjectListItem[], slug: string | null | undefined, path: string): string | undefined {
  return slug ? projects.find((project) => project.slug === slug)?.file_names?.[path] : undefined
}
