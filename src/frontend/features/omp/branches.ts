import type { BranchNode } from "@gigachad/chat-ui/protocol"

/** One in-session branch: the path from a session's first entry to `leafId`. */
export interface SessionBranch {
  leafId: string
  /** First user prompt after the branch left its siblings, else the newest text on it. */
  label: string
}

const isPrompt = (node: BranchNode) => node.role === "user" && node.preview.trim() !== ""
const holdsMessage = (node: BranchNode) => node.role !== undefined || node.preview.trim() !== ""

/**
 * Every leaf of an omp session tree as a branch, in tree order. A tree without forks has a single branch.
 *
 * - Skips a branch without a message after its fork: an Edit never sent leaves one (omp-web's history-navigation marker)
 * - Iterative depth-first walk over one shared path, sessions run thousands of entries deep
 */
export function sessionBranches(roots: readonly BranchNode[]): SessionBranch[] {
  const branches: SessionBranch[] = []
  const path: BranchNode[] = []
  // `splitAt`: path index of the first entry after the deepest fork above `node`.
  const pending = [...roots].reverse().map((node) => ({ node, depth: 0, splitAt: 0 }))
  while (pending.length > 0) {
    const { node, depth, splitAt } = pending.pop()!
    path.length = depth
    path.push(node)
    if (node.children.length === 0) {
      const sinceFork = path.slice(splitAt)
      if (!sinceFork.some(holdsMessage)) continue
      const prompt = sinceFork.find(isPrompt)
      const newest = path.findLast((entry) => entry.preview.trim() !== "")
      branches.push({ leafId: node.id, label: (prompt ?? newest)?.preview.trim() || "Empty branch" })
      continue
    }
    const childSplitAt = node.children.length > 1 ? depth + 1 : splitAt
    for (const child of [...node.children].reverse()) pending.push({ node: child, depth: depth + 1, splitAt: childSplitAt })
  }
  return branches
}
