/**
 * Compressed path-prefix tree over path segments, for browsing flat filepaths as directories.
 * Chains of directories with no files and a single subdirectory collapse, so the tree opens
 * at the common root and branches only where paths diverge.
 */

export interface PathTreeNode {
  /** Absolute directory path this node represents. */
  path: string
  /** Immediate child directories (already chain-collapsed). */
  dirs: PathTreeNode[]
  /** Absolute filepaths directly contained in this directory. */
  files: string[]
}

interface RawNode {
  path: string
  children: Map<string, RawNode>
  files: string[]
}

// Joins path segments. A leading empty segment (absolute path) yields "/" as the root.
function joinSeg(base: string, seg: string): string {
  if (base === "") return seg === "" ? "/" : seg
  if (base === "/") return "/" + seg
  return base + "/" + seg
}

function compress(raw: RawNode): PathTreeNode {
  const dirs = [...raw.children.values()].map(compress)
  // No files and one subdirectory: no branching information, collapse into the child
  if (raw.files.length === 0 && dirs.length === 1) return dirs[0]
  return { path: raw.path, dirs, files: raw.files }
}

/** `paths` are file paths split on "/". Each file lands in its parent directory node. Returns the collapsed root. */
export function buildPathTree(paths: string[]): PathTreeNode {
  const root: RawNode = { path: "", children: new Map(), files: [] }
  for (const p of paths) {
    const slash = p.lastIndexOf("/")
    const dir = slash === -1 ? "" : p.slice(0, slash)
    let node = root
    if (dir !== "") {
      let cur = ""
      for (const seg of dir.split("/")) {
        cur = joinSeg(cur, seg)
        let child = node.children.get(cur)
        if (!child) {
          child = { path: cur, children: new Map(), files: [] }
          node.children.set(cur, child)
        }
        node = child
      }
    }
    node.files.push(p)
  }
  return compress(root)
}

/** Display label of *childPath* relative to its parent directory *parentPath*. */
export function relativeName(parentPath: string, childPath: string): string {
  if (parentPath && childPath.startsWith(parentPath === "/" ? "/" : parentPath + "/")) {
    return childPath.slice(parentPath === "/" ? 1 : parentPath.length + 1)
  }
  const i = childPath.lastIndexOf("/")
  return i === -1 ? childPath : childPath.slice(i + 1)
}
