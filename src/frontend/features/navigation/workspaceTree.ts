import { Archive, FileText, FolderPlus, GitCompare, MessageSquare, Shapes, Upload } from "lucide-react"
import { ArchitectureIcon } from "@/components/ui/ArchitectureIcon"
import { DEFAULT_DRAG_KIND, type FileTreeItem } from "@/components/ui/file-tree/FileTree"
import type { useInlineRename } from "@/components/ui/file-tree/useInlineRename"
import type { AppSurface, WorkspaceItem } from "@/contexts/TabContext"
import { isArchitecturePath } from "@/lib/domain/architecturePaths"
import { mountOf, type FileTreeNode } from "@/lib/domain/files"
import type { BranchMeta } from "@/lib/domain/histories"
import type { ProjectDocument, ProjectListItem } from "@/lib/domain/projects"
import { baseName, isInside, trimSlash } from "@/lib/paths"
import type { ArchiveEntry } from "./ArchiveSelector"
import type { useBranches } from "./BranchContext"
import type { useCodeMounts } from "./useCodeMounts"
import type { useProjectActions } from "./useProjectActions"
import type { useFileRename } from "./useFileRename"
import { isStoredProjectFile, type useWorkspaceDocuments } from "./useWorkspaceDocuments"

type CodeMounts = ReturnType<typeof useCodeMounts>
type Renames = Pick<ReturnType<typeof useInlineRename>, "renameOf"> & Pick<ReturnType<typeof useFileRename>, "renameFile">

/** What the rows of a folder on disk do, shared by mounted folders and a project's promoted LaTeX folders. */
type FolderRowContext =
  & Renames
  & {
    activeDocumentPath?: string | null
    onFileSelect?: (path: string) => void
    /** Move the project file `path` dropped on a directory into `folder`. */
    moveToFolder: (path: string, folder: string) => Promise<void>
    /** Create `name` in the LaTeX project folder `folder` and open it. */
    createLatexFile: (folder: string, name: string) => Promise<void>
    /** Whether the folder `id` is open: the rows of a closed folder are not built. */
    isExpanded: (id: string) => boolean
  }

/** Everything the Projects tree shows and does. */
type ProjectTreeContext =
  & Pick<ReturnType<typeof useWorkspaceDocuments>, "documents" | "notes" | "move" | "createAndOpen" | "selectDocument" | "createFile" | "newFileChoice" | "remove" | "promote" | "importFiles" | "uploadFiles" | "hasArchitecture" | "openArchitecture">
  & Pick<ReturnType<typeof useProjectActions>, "renameProjectTo" | "removeProject" | "changeArchive">
  & FolderRowContext
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
    onItemOpen?: (item: WorkspaceItem) => void
    onProjectSelect?: (slug: string) => void
  }

/** Everything the Code tree shows and does: the Pi parts are session folders and extra actions of directory rows. */
type CodeTreeContext =
  & Pick<CodeMounts, "rootTreeData" | "mounts" | "unmounted" | "unused" | "mountRepository" | "folder" | "archived" | "setArchived" | "removeMount">
  & FolderRowContext
  & {
    sessions: Pick<CodeMounts["sessions"], "directoryCode" | "directoryActions">
    projects: ProjectListItem[]
    pending: (key: string) => boolean
    onItemOpen?: (item: WorkspaceItem) => void
  }

/** Drag kind of a stored project file: it moves into another project's files or a Code folder, never among chats or canvases. */
const PROJECT_FILE_DRAG_KIND = "project-file"

/** Whether the sidebar can rename the project file `path`: a stored file or one on disk, not a library PDF. */
const isRenamableFile = (path: string) => isStoredProjectFile(path) || path.startsWith("/")

/** Whether `path` is a `.tex` directly inside a project's workbench folder `<root>/workbench/latex/<slug>/<folder>/`, which can become a LaTeX project folder. */
function isPromotableLatex(path: string, slug: string): boolean {
  const [workbench, kind, owner, folder, file] = path.split("/").slice(-5)
  return workbench === "workbench" && kind === "latex" && owner === slug && Boolean(folder) && file?.toLowerCase().endsWith(".tex") === true
}

/** File mount linked to a project: the mount whose folder name is the project slug (same rule as the backend's architecture store). */
function projectMount(slug: string, mounts: readonly string[]): string | null {
  return mounts.find((mount) => baseName(trimSlash(mount)) === slug) ?? null
}

/** The folder on disk of `project`: its linked root, else its mount. */
const projectFolder = (project: ProjectListItem, mounts: readonly string[]) => project.linked_root ?? projectMount(project.slug, mounts)

/** Slug of the project the file `path` belongs to: the one listing it (as an entry or inside a folder entry), else the one whose linked folder holds it, null when none does. */
export function owningProject(path: string, documents: Record<string, ProjectDocument[]>, projects: ProjectListItem[], mounts: readonly string[]): string | null {
  const lister = projects.find((project) => documents[project.slug]?.some((doc) => isInside(path, doc.path)))
  if (lister) return lister.slug
  return projects.find((project) => {
    const folder = projectFolder(project, mounts)
    return folder !== null && isInside(path, folder)
  })?.slug ?? null
}

/** Project roots followed by the shared "general" root. */
export function projectItems(context: ProjectTreeContext): FileTreeItem<string>[] {
  const { projects, pending, hasArchitecture, move, importFiles } = context
  // Chats by their project's slug, the first segment of their path
  const chats = Map.groupBy(Object.entries(context.branchMeta), ([path]) => path.split("/", 1)[0])
  return [
    ...projects.filter((project) => !project.archived && !pending(`project:${project.slug}`)).map((project): FileTreeItem<string> => ({
      id: project.slug, label: project.name, type: "root", isActive: context.activeProjectSlug === project.slug,
      onSelect: () => context.onProjectSelect?.(project.slug),
      rename: context.renameOf(`project:${project.slug}`, (name) => context.renameProjectTo(project, name)),
      onDelete: () => context.removeProject(project),
      actions: [
        { label: `Archive ${project.name}`, menuLabel: "Archive", icon: Archive, onClick: () => context.changeArchive(project.slug, true) },
        { label: `${hasArchitecture(project.slug) ? "Open" : "Create"} architecture for ${project.name}`,
          menuLabel: `${hasArchitecture(project.slug) ? "Open" : "Create"} architecture`, icon: ArchitectureIcon,
          onClick: () => context.openArchitecture(project.slug) },
      ],
      onMoveHere: (path) => move(path, project.slug),
      accepts: [DEFAULT_DRAG_KIND, PROJECT_FILE_DRAG_KIND],
      onDropFiles: (paths) => importFiles(project.slug, paths),
      sections: categoryItems(context, project, chats.get(project.slug) ?? []),
      children: [],
    })),
    {
      id: "shared", label: "general", type: "root",
      onMoveHere: (path) => move(path, ""),
      sections: categoryItems(context, null),
      children: [],
    },
  ]
}

/** The chats, canvases and (for a project) files folders of `project`, or of the general scope when null. */
function categoryItems(context: ProjectTreeContext, project: ProjectListItem | null, projectMeta: [string, BranchMeta][] = []): FileTreeItem<string>[] {
  const { branchMeta, pending, appMode, move, createAndOpen, remove } = context
  const scope = project?.slug ?? ""
  const openCanvas = (path: string) => context.onItemOpen?.({ kind: "canvas", path, projectSlug: scope || null })
  const owner = project?.name ?? "general"
  const prefix = project ? `project:${scope}` : "shared"
  const docs = (project ? context.documents[scope] ?? [] : context.notes)
    .filter((doc) => !pending(`document:${scope}:${doc.path}`))
  const files = docs.filter((doc) => !doc.path.endsWith(".canvas") && !isArchitecturePath(doc.path))
  const extractions = extractionsOf(files, scope)
  /** The Markdown of an extracted PDF is its text view in the PDF's top bar, so it has no row of its own. */
  const hidden = new Set([...extractions.values()].map((extraction) => extraction.path))
  /** `row`, deleting the PDF's extraction with it. */
  const withExtraction = (doc: ProjectDocument, row: FileTreeItem<string>, extraction?: ProjectDocument): FileTreeItem<string> => extraction
    ? { ...row, onDelete: () => remove(doc, scope, [extraction]) }
    : row
  const chats = project ? projectChats(project, projectMeta) : context.visibleRootFiles.map((path) => ({ path, label: path.replace(/\.json$/, "") }))
  return [
    {
      id: `${prefix}:chats`, label: "Chats", icon: MessageSquare, type: "folder",
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
      id: `${prefix}:canvases`, label: "Canvases", icon: Shapes, type: "folder",
      onMoveHere: (path) => move(path, scope),
      create: {
        label: `New canvas in ${owner}`, placeholder: "Canvas name",
        onSubmit: (name) => createAndOpen(scope, name.endsWith(".canvas") ? name : `${name}.canvas`, openCanvas),
      },
      children: docs.filter((doc) => doc.path.endsWith(".canvas")).map((doc) => ({
        id: doc.path, label: doc.name.replace(/\.canvas$/, ""), type: "element", draggable: true,
        isActive: appMode === "canvas" && context.activeCanvasPath === doc.path,
        onSelect: () => openCanvas(doc.path),
        onDelete: () => remove(doc, scope),
      })),
    },
    ...(project ? [{
      id: `${prefix}:files`, label: "Files", icon: FileText, type: "folder" as const,
      actions: [{ label: `Upload files to ${owner}`, icon: Upload, onClick: () => void context.uploadFiles(scope) }],
      create: {
        label: `New file in ${owner}`, placeholder: "File name",
        choice: context.newFileChoice,
        onSubmit: (name: string, kind?: string, template?: string) => context.createFile(scope, name, kind ?? "Markdown", template),
      },
      onDropFiles: (paths: string[]) => context.importFiles(scope, paths),
      onMoveHere: (path: string) => move(path, scope),
      accepts: [PROJECT_FILE_DRAG_KIND],
      children: files.filter((doc) => !hidden.has(doc.path)).map((doc): FileTreeItem<string> => doc.children ? {
        // A promoted LaTeX project: its folder on disk, shown and edited like a mounted folder.
        id: doc.path, label: doc.name, type: "folder",
        onDelete: () => remove(doc, scope),
        create: createFileAction(context, doc.path, doc.name),
        onMoveHere: (path) => context.moveToFolder(path, doc.path),
        accepts: [PROJECT_FILE_DRAG_KIND],
        children: rootFileItems(context, doc.children),
      } : withExtraction(doc, {
        id: doc.path, label: doc.name, type: "element", isActive: context.activeDocumentPath === doc.path,
        draggable: isStoredProjectFile(doc.path), dragKind: PROJECT_FILE_DRAG_KIND,
        onSelect: () => context.selectDocument(doc.path, scope),
        onDelete: () => remove(doc, scope),
        actions: isPromotableLatex(doc.path, scope)
          ? [{ label: "Make LaTeX project", icon: FolderPlus, onClick: () => context.promote(doc.path, scope) }]
          : undefined,
        rename: isRenamableFile(doc.path)
          ? context.renameOf(`file:${scope}:${doc.path}`, (name) => context.renameFile({ path: doc.path, name: doc.name, scope }, name), true)
          : undefined,
      }, extractions.get(doc.path))),
    }] : []),
  ]
}

/** Each PDF of `files` with the Markdown MinerU extracted from it, keyed by the PDF's path. That is `<stem>.md` stored in the project, or where opening it moved it on disk. */
function extractionsOf(files: ProjectDocument[], scope: string): Map<string, ProjectDocument> {
  const pairs = files.flatMap((pdf) => {
    if (!pdf.name.toLowerCase().endsWith(".pdf")) return []
    const stem = pdf.name.slice(0, -4)
    const stored = `project/${scope}/document/${stem}.md`
    const moved = `/workbench/markdown/${scope}/${stem}/${stem}.md`
    const extraction = files.find((doc) => doc.path === stored || doc.path.endsWith(moved))
    return extraction ? [[pdf.path, extraction] as const] : []
  })
  return new Map(pairs)
}

/** Root chats of `project` from its branch metadata, labelled by their project tab. */
function projectChats(project: ProjectListItem, projectMeta: [string, BranchMeta][]): { path: string; label: string }[] {
  const prefix = `${project.slug}/`
  const tabs = new Map((project.tabs ?? []).map((tab) => [tab.filename, tab]))
  return projectMeta.flatMap(([path, metadata]) => {
    if (metadata.parent_id) return []
    const filename = path.slice(prefix.length)
    if (filename === "project.json" || filename.startsWith("memory/")) return []
    const tab = tabs.get(filename)
    return [{ path, label: tab?.name ?? tab?.title ?? filename.replace(/\.json$/, "") }]
  })
}

/**
 * One tree of projects and code: `projectRoots` (projects, then "general") with the mount, unmounted and unused rows.
 * - A project with a linked mount is one root: its Chats, Canvases, Files, then the mount's Worktrees, Sessions and files
 * - Mounts without a visible project sit between the projects and "general"
 */
export function workspaceItems(projectRoots: FileTreeItem<string>[], context: CodeTreeContext): FileTreeItem<string>[] {
  const mountRoots = new Map(mountItems(context).map((root) => [root.id, root]))
  const linked = new Set<string>()
  const roots = projectRoots.map((root) => {
    const project = context.projects.find((candidate) => candidate.slug === root.id)
    const folder = project && projectFolder(project, context.mounts)
    const mount = folder ? mountRoots.get(folder) : undefined
    if (!mount) return root
    linked.add(mount.id)
    // The project's own Archive and Delete stand for the whole root
    const mountActions = (mount.actions ?? []).filter((action) => action.menuLabel !== "Archive")
    return { ...root, actions: [...mountActions, ...(root.actions ?? [])], sections: [...(root.sections ?? []), ...(mount.sections ?? [])], children: mount.children }
  })
  const general = roots.filter((root) => root.id === "shared")
  return [
    ...roots.filter((root) => root.id !== "shared"),
    ...[...mountRoots.values()].filter((root) => !linked.has(root.id)),
    ...general,
    ...unmountedItems(context), ...unusedItems(context),
  ]
}

/** Mounts keep their registered path as id, so switching worktrees keeps them expanded. */
function mountItems(context: CodeTreeContext): FileTreeItem<string>[] {
  const { sessions, projects, mounts, archived } = context
  return context.rootTreeData
    .filter((node) => !context.pending(`root:${mountOf(node)}`) && !archived.has(mountOf(node)))
    .map((node) => {
      const mount = mountOf(node)
      const code = sessions.directoryCode(mount, node.name, projects.find((project) => projectFolder(project, mounts) === mount)?.slug ?? null)
      return {
        id: mount,
        label: node.name,
        type: "folder",
        // Dragging a mount adds its path, only unambiguous while it shows the clone itself.
        multiSelect: node.path === mount,
        actions: [
          { label: `Open diff for ${node.name}`, menuLabel: "Open diff", icon: GitCompare, inline: true, onClick: () => context.onItemOpen?.({ kind: "diff", path: node.path, projectSlug: null }) },
          { label: `Archive ${node.name}`, menuLabel: "Archive", icon: Archive, onClick: () => context.setArchived(mount, true) },
          ...sessions.directoryActions(mount, node.name),
        ],
        onDelete: () => context.removeMount(mount, node.name),
        onMoveHere: (path) => context.moveToFolder(path, node.path),
        accepts: [PROJECT_FILE_DRAG_KIND],
        sections: [context.folder(mount), code].filter((section) => section !== null),
        children: rootFileItems(context, node.children ?? []),
      }
    })
}

/** Whether a folder directly holds a `.tex` file, so it takes new LaTeX project files. */
const holdsTex = (children?: FileTreeNode[] | null) => !!children?.some((child) => child.type === "file" && /\.tex$/i.test(child.name))

/** The `+` of the LaTeX project folder `folder`: a new file in it. */
function createFileAction(context: FolderRowContext, folder: string, name: string): NonNullable<FileTreeItem<string>["create"]> {
  return { label: `New file in ${name}`, placeholder: "File name", onSubmit: (file) => context.createLatexFile(folder, file) }
}

/** The children of a closed folder: FileTree renders them only once it opens, which builds them. */
const NO_ROWS: FileTreeItem<string>[] = []

/** Rows of the files and folders of a folder on disk. */
function rootFileItems(context: FolderRowContext, nodes: FileTreeNode[]): FileTreeItem<string>[] {
  return nodes.map((node) => ({
    id: node.path,
    label: node.name,
    type: node.type === "file" ? "element" : "folder",
    onMoveHere: node.type === "file" ? undefined : (path: string) => context.moveToFolder(path, node.path),
    accepts: [PROJECT_FILE_DRAG_KIND],
    multiSelect: true,
    isActive: node.path === context.activeDocumentPath,
    onSelect: node.type === "file" ? () => context.onFileSelect?.(node.path) : undefined,
    create: node.type !== "file" && holdsTex(node.children) ? createFileAction(context, node.path, node.name) : undefined,
    rename: node.type === "file"
      ? context.renameOf(`code:${node.path}`, (name) => context.renameFile({ path: node.path, name: node.name, scope: "" }, name), true)
      : undefined,
    children: node.children ? (context.isExpanded(node.path) ? rootFileItems(context, node.children) : NO_ROWS) : undefined,
  }))
}

/** Checkouts with Pi sessions outside every mount: a directory row holding only its sessions. */
function unmountedItems({ unmounted, archived, sessions, setArchived }: CodeTreeContext): FileTreeItem<string>[] {
  return unmounted
    .filter((workspace) => !archived.has(workspace.path))
    .map((workspace) => {
      const code = sessions.directoryCode(workspace.path, workspace.label, null, workspace.sessions)
      return {
        id: workspace.path,
        label: workspace.label,
        type: "folder",
        actions: [
          { label: `Archive ${workspace.label}`, menuLabel: "Archive", icon: Archive, onClick: () => setArchived(workspace.path, true) },
          ...sessions.directoryActions(workspace.path, workspace.label),
        ],
        sections: [code],
        children: [],
      }
    })
}

/** Checkouts never used: one chevron folder, each a row to mount (files, sessions) or open the factory of. */
function unusedItems({ unused, archived, sessions, mountRepository }: CodeTreeContext): FileTreeItem<string>[] {
  const shown = unused.filter((repository) => !archived.has(repository.path))
  if (shown.length === 0) return []
  return [{
    id: "code-unused",
    label: "Not yet used",
    type: "folder",
    chevronOnly: true,
    children: shown.map((repository) => ({
      id: `unused:${repository.path}`,
      label: repository.name,
      type: "element",
      onSelect: () => { void mountRepository(repository.path) },
      actions: [
        { label: `Mount ${repository.name}`, menuLabel: "Mount", icon: FolderPlus, onClick: () => { void mountRepository(repository.path) } },
        ...sessions.directoryActions(repository.path, repository.name),
      ],
    })),
  }]
}

/** Every mount and unmounted workspace of the Code section, with its archive state. */
export function directoryEntries({ rootTreeData, unmounted, pending, archived }: CodeTreeContext): ArchiveEntry[] {
  return [
    ...rootTreeData.filter((node) => !pending(`root:${mountOf(node)}`)).map((node) => ({ key: mountOf(node), name: node.name })),
    ...unmounted.map((workspace) => ({ key: workspace.path, name: workspace.label })),
  ].map((entry) => ({ ...entry, archived: archived.has(entry.key) }))
}
