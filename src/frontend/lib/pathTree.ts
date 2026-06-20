/**
 * Compressed path-prefix tree (radix trie over path segments) for browsing a
 * flat list of filepaths as directories.
 *
 * Built in a single O(total segments) pass, then single-child/no-file chains are
 * collapsed so the tree opens at the *common root* and only branches where the
 * paths actually diverge. This generalises cleanly: one vault collapses to its
 * root folder; multiple vaults (or any unrelated roots) surface as sibling
 * entries under their lowest common ancestor.
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

function joinSeg(base: string, seg: string): string {
  if (base === "") return seg === "" ? "/" : seg
  if (base === "/") return "/" + seg
  return base + "/" + seg
}

function compress(raw: RawNode): PathTreeNode {
  const dirs = [...raw.children.values()].map(compress)
  // A directory with no files of its own and exactly one subdirectory carries no
  // branching information — collapse it into that child.
  if (raw.files.length === 0 && dirs.length === 1) return dirs[0]
  return { path: raw.path, dirs, files: raw.files }
}

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
