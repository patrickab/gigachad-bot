export interface ProjectDocument {
  path: string
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
}
