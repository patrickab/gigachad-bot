import { Archive, FilePlus, GitCompare } from "lucide-react"
import { ArchitectureIcon } from "@/components/ui/ArchitectureIcon"
import type { FileTreeItem } from "@/components/ui/file-tree/FileTree"
import type { AppSurface } from "@/contexts/TabContext"
import { projectMount } from "@/features/omp/sessions"
import { isArchitecturePath } from "@/lib/domain/architecturePaths"
import type { FileTreeNode } from "@/lib/domain/files"
import type { BranchMeta } from "@/lib/domain/histories"
import type { ProjectListItem } from "@/lib/domain/projects"
import type { ArchiveEntry } from "./ArchiveSelector"
import type { useBranches } from "./BranchContext"
import type { useCodeMounts } from "./useCodeMounts"
import type { useProjectActions } from "./useProjectActions"
import type { useWorkspaceDocuments } from "./useWorkspaceDocuments"

type CodeMounts = ReturnType<typeof useCodeMounts>

/** Everything the Projects tree shows and does. */
type ProjectTreeContext =
  & Pick<ReturnType<typeof useWorkspaceDocuments>, "documents" | "notes" | "move" | "createAndOpen" | "remove" | "importFiles" | "hasArchitecture" | "openArchitecture" | "setCreateTarget">
  & Pick<ReturnType<typeof useProjectActions>, "startRename" | "removeProject" | "changeArchive">
  & Pick<ReturnType<typeof useBranches>, "branchMeta" | "visibleRootFiles" | "activeFile" | "activeQaIndex" | "onFileClick">
  & {
    projects: ProjectListItem[]
    pending: (key: string) => boolean
    appMode: AppSurface
    activeCanvasPath?: string | null
    activeDocumentPath?: string | null
    activeProjectSlug: string | null
    onNewChat?: (slug: string | null) => void
    deleteChat: (path: string) => void
    onCanvasSelect?: (path: string, scope: string) => void
    onDocumentSelect?: (path: string, scope: string) => void
    onProjectSelect?: (slug: string) => void
  }

/** Everything the Code tree shows and does: the omp-web parts are session folders and extra actions of directory rows. */
type CodeTreeContext =
  & Pick<CodeMounts, "rootTreeData" | "mounts" | "unmounted" | "folder" | "archived" | "setArchived" | "removeMount">
  & {
    omp: Pick<CodeMounts["omp"], "directoryCode" | "directoryActions" | "archivedSessions">
    projects: ProjectListItem[]
    pending: (key: string) => boolean
    activeDocumentPath?: string | null
    onFileSelect?: (path: string) => void
    onDiffSelect?: (path: string) => void
  }

/** Registered path of a file-root node: its mount when set, else its own path. */
export const mountOf = (node: FileTreeNode): string => node.mount ?? node.path

/** Project roots followed by the shared "general" root. */
export function projectItems(context: ProjectTreeContext): FileTreeItem<string>[] {
  const { projects, pending, hasArchitecture, move, importFiles } = context
  const [generalChats, ...generalCategories] = categoryItems(context, null)
  return [
    ...projects.filter((project) => !project.archived && !pending(`project:${project.slug}`)).map((project): FileTreeItem<string> => ({
      id: project.slug, label: project.name, type: "root", isActive: context.activeProjectSlug === project.slug,
      onSelect: () => context.onProjectSelect?.(project.slug),
      onDoubleClick: () => context.startRename(project),
      onDelete: () => context.removeProject(project),
      actions: [
        { label: `Archive ${project.name}`, icon: Archive, onClick: () => context.changeArchive(project.slug, true) },
        { label: `New document in ${project.name}`, icon: FilePlus, onClick: () => context.setCreateTarget(project.slug) },
        { label: `${hasArchitecture(project.slug) ? "Open" : "Create"} architecture for ${project.name}`, icon: ArchitectureIcon,
          onClick: () => context.openArchitecture(project.slug) },
      ],
      onMoveHere: (path) => move(path, project.slug),
      onDropFiles: (paths) => importFiles(project.slug, paths),
      children: categoryItems(context, project),
    })),
    {
      id: "shared", label: "general", type: "root",
      create: generalChats.create,
      onMoveHere: (path) => move(path, ""),
      children: [...generalCategories, ...(generalChats.children ?? [])],
    },
  ]
}

/** The chats, canvases and (for a project) files folders of `project`, or of the general scope when null. */
function categoryItems(context: ProjectTreeContext, project: ProjectListItem | null): FileTreeItem<string>[] {
  const { branchMeta, pending, appMode, move, createAndOpen, remove, onCanvasSelect, onDocumentSelect } = context
  const scope = project?.slug ?? ""
  const owner = project?.name ?? "general"
  const prefix = project ? `project:${scope}` : "shared"
  const docs = (project ? context.documents[scope] ?? [] : context.notes)
    .filter((doc) => !pending(`document:${scope}:${doc.path}`))
  const chats = project ? projectChats(project, branchMeta) : context.visibleRootFiles.map((path) => ({ path, label: path.replace(/\.json$/, "") }))
  return [
    {
      id: `${prefix}:chats`, label: "chats", type: "folder",
      create: { label: `New chat in ${owner}`, onClick: () => context.onNewChat?.(scope || null) },
      onMoveHere: (path) => move(path, scope),
      children: chats.filter((chat) => !pending(`history:${chat.path}`)).map((chat) => {
        const count = branchMeta[chat.path]?.qa_count ?? 0
        return {
          id: chat.path, label: chat.label, type: "element", data: chat.path, draggable: true,
          badge: count > 0 ? `@${count - 1}` : undefined,
          isActive: appMode === "chat" && context.activeFile === chat.path && context.activeQaIndex == null,
          onSelect: () => context.onFileClick(chat.path),
          onDelete: () => context.deleteChat(chat.path),
        }
      }),
    },
    {
      id: `${prefix}:canvases`, label: "canvases", type: "folder",
      onMoveHere: (path) => move(path, scope),
      create: {
        label: `New canvas in ${owner}`, placeholder: "Canvas name",
        onSubmit: (name) => createAndOpen(scope, name.endsWith(".canvas") ? name : `${name}.canvas`, (path) => onCanvasSelect?.(path, scope)),
      },
      children: docs.filter((doc) => doc.path.endsWith(".canvas")).map((doc) => ({
        id: doc.path, label: doc.name.replace(/\.canvas$/, ""), type: "element", draggable: true,
        isActive: appMode === "canvas" && context.activeCanvasPath === doc.path,
        onSelect: () => onCanvasSelect?.(doc.path, scope),
        onDelete: () => remove(doc, scope),
      })),
    },
    ...(project ? [{
      id: `${prefix}:files`, label: "files", type: "folder" as const,
      create: {
        label: `New file in ${owner}`, placeholder: "File name",
        onSubmit: (name: string) => createAndOpen(scope, name, (path) => onDocumentSelect?.(path, scope)),
      },
      onDropFiles: (paths: string[]) => context.importFiles(scope, paths),
      children: docs.filter((doc) => !doc.path.endsWith(".canvas") && !isArchitecturePath(doc.path)).map((doc): FileTreeItem<string> => ({
        id: doc.path, label: doc.name, type: "element", isActive: context.activeDocumentPath === doc.path,
        onSelect: () => onDocumentSelect?.(doc.path, scope),
        onDelete: () => remove(doc, scope),
      })),
    }] : []),
  ]
}

/** Root chats of `project` from the branch metadata, labelled by their project tab. */
function projectChats(project: ProjectListItem, branchMeta: Record<string, BranchMeta>): { path: string; label: string }[] {
  const prefix = `${project.slug}/`
  const tabs = new Map((project.tabs ?? []).map((tab) => [tab.filename, tab]))
  return Object.entries(branchMeta).flatMap(([path, metadata]) => {
    if (!path.startsWith(prefix) || metadata.parent_id) return []
    const filename = path.slice(prefix.length)
    if (filename === "project.json" || filename.startsWith("memory/")) return []
    const tab = tabs.get(filename)
    return [{ path, label: tab?.name ?? tab?.title ?? filename.replace(/\.json$/, "") }]
  })
}

/** Mount rows, omp workspaces outside every mount, then the archived sessions folder. */
export function codeItems(context: CodeTreeContext): FileTreeItem<string>[] {
  const archivedSessions = context.omp.archivedSessions()
  return [...mountItems(context), ...unmountedItems(context), ...(archivedSessions ? [archivedSessions] : [])]
}

/** Mounts keep their registered path as id, so switching worktrees keeps them expanded. */
function mountItems(context: CodeTreeContext): FileTreeItem<string>[] {
  const { omp, projects, mounts, archived } = context
  return context.rootTreeData
    .filter((node) => !context.pending(`root:${mountOf(node)}`) && !archived.has(mountOf(node)))
    .map((node) => {
      const mount = mountOf(node)
      const code = omp.directoryCode(mount, node.name, projects.find((project) => (project.linked_root ?? projectMount(project.slug, mounts)) === mount)?.slug ?? null)
      return {
        id: mount,
        label: node.name,
        type: "folder",
        // Dragging a mount adds its path, only unambiguous while it shows the clone itself.
        multiSelect: node.path === mount,
        actions: [
          { label: `Archive ${node.name}`, icon: Archive, onClick: () => context.setArchived(mount, true) },
          { label: `Open diff for ${node.name}`, icon: GitCompare, onClick: () => context.onDiffSelect?.(node.path) },
          ...omp.directoryActions(mount, node.name),
        ],
        onDelete: () => context.removeMount(mount, node.name),
        children: [...generatedRows([context.folder(mount), code]), ...rootFileItems(context, node.children ?? [])],
      }
    })
}

function rootFileItems(context: CodeTreeContext, nodes: FileTreeNode[]): FileTreeItem<string>[] {
  return nodes.map((node) => ({
    id: node.path,
    label: node.name,
    type: node.type === "file" ? "element" : "folder",
    multiSelect: true,
    isActive: node.path === context.activeDocumentPath,
    onSelect: node.type === "file" ? () => context.onFileSelect?.(node.path) : undefined,
    children: node.children ? rootFileItems(context, node.children) : undefined,
  }))
}

/** Omp workspaces outside every mount: a directory row holding only its sessions. */
function unmountedItems({ unmounted, archived, omp, setArchived }: CodeTreeContext): FileTreeItem<string>[] {
  return unmounted
    .filter((workspace) => !archived.has(workspace.path))
    .map((workspace) => {
      const code = omp.directoryCode(workspace.path, workspace.label, null, workspace.sessions)
      return {
        id: workspace.path,
        label: workspace.label,
        type: "folder",
        actions: [
          { label: `Archive ${workspace.label}`, icon: Archive, onClick: () => setArchived(workspace.path, true) },
          ...omp.directoryActions(workspace.path, workspace.label),
        ],
        children: code ? [code] : [],
      }
    })
}

/** Every mount and unmounted workspace of the Code section, with its archive state. */
export function directoryEntries({ rootTreeData, unmounted, pending, archived }: CodeTreeContext): ArchiveEntry[] {
  return [
    ...rootTreeData.filter((node) => !pending(`root:${mountOf(node)}`)).map((node) => ({ key: mountOf(node), name: node.name })),
    ...unmounted.map((workspace) => ({ key: workspace.path, name: workspace.label })),
  ].map((entry) => ({ ...entry, archived: archived.has(entry.key) }))
}

/** Worktrees and Sessions rows that exist, the last one spaced from the files below it. */
function generatedRows(rows: (FileTreeItem<string> | null)[]): FileTreeItem<string>[] {
  const present = rows.filter((row) => row !== null)
  return present.map((row, index) => index === present.length - 1 ? { ...row, gapAfter: true } : row)
}
