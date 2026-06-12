"use client"

import { GitMerge, Trash2 } from "lucide-react"
import { useMemo } from "react"
import { cn } from "@/lib/utils"
import type { BranchChild, BranchMeta } from "@/lib/types"
import { useBranchContext, buildChatIdMap, isAncestorOf } from "@/contexts/BranchContext"
import { useTree, type VaultTreeItem } from "./VaultTree"

interface Node {
  file: string
  label: string
  qaCount: number
  parentQaCount: number
  children: Node[]
  branchIdx: number | null
}

function buildTree(root: string, meta: Record<string, BranchMeta>, chatIdMap: Map<string, string>): Node {
  const rootMeta = meta[root]
  const seen = new Set<string>()
  function walk(file: string, parentQaCount: number = 0): Node {
    if (seen.has(file)) return { file, label: file.split("/").pop()!.replace(".json", ""), qaCount: 0, parentQaCount: 0, children: [], branchIdx: null }
    seen.add(file)
    const m = meta[file]
    const qa = m?.qa_count ?? 0
    return {
      file,
      label: file.split("/").pop()!.replace(".json", ""),
      qaCount: qa,
      parentQaCount,
      children: (m?.children ?? []).map((c: BranchChild) => {
        const cf = chatIdMap.get(c.chat_id) ?? c.chat_id
        return { ...walk(cf, qa), branchIdx: c.branch_message_idx }
      }),
      branchIdx: null,
    }
  }
  const qa = rootMeta?.qa_count ?? 0
  return {
    file: root,
    label: root.split("/").pop()!.replace(".json", ""),
    qaCount: qa,
    parentQaCount: 0,
    children: (rootMeta?.children ?? []).map((c: BranchChild) => {
      const cf = chatIdMap.get(c.chat_id) ?? c.chat_id
      return { ...walk(cf, qa), branchIdx: c.branch_message_idx }
    }),
    branchIdx: null,
  }
}

type Point = { idx: number; nodes: Node[] }

function groupByPoint(children: Node[]): Point[] {
  const m = new Map<number, Node[]>()
  for (const c of children) {
    const k = c.branchIdx ?? 0
    if (!m.has(k)) m.set(k, [])
    m.get(k)!.push(c)
  }
  return [...m.entries()].sort(([a], [b]) => a - b).map(([idx, nodes]) => ({ idx, nodes }))
}

function BranchTree({ root }: { root: Node }) {
  const { branchMeta, onFileClick, onMerge, onDelete, activeFile, activeQaIndex } = useBranchContext()
  const points = groupByPoint(root.children)
  if (points.length === 0) return null

  return (
    <div className="relative">
      <div className="absolute left-[11px] top-0 bottom-0 w-px bg-divider pointer-events-none" />
      {points.map((pt) => (
        <div key={`${root.file}-${pt.idx}`}>
          <button
            onClick={() => onFileClick?.(root.file, pt.idx)}
            className={cn(
              "relative z-[1] ml-3 px-1 py-0.5 rounded-sm text-[10px] font-mono tabular-nums transition-colors",
              activeFile === root.file && activeQaIndex === pt.idx
                ? "bg-surface-elevated text-ink font-semibold"
                : "text-ink-muted hover:text-ink hover:bg-surface-elevated/50"
            )}
          >
            @{pt.idx}
          </button>
          <div className="relative">
            <div className="absolute left-[11px] top-0 bottom-0 w-px bg-divider pointer-events-none" />
            {pt.nodes.map((n) => (
              <BranchRow key={n.file} node={n} />
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}

function BranchRow({ node }: { node: Node }) {
  const { onFileClick, onMerge, onDelete, activeFile, activeQaIndex } = useBranchContext()
  const isActive = activeFile === node.file && activeQaIndex == null
  const hasKids = node.children.length > 0
  const mergeBlocked = node.branchIdx != null && node.parentQaCount > node.branchIdx + 1

  return (
    <div className="relative">
      <div className="absolute left-[11px] top-2 h-px w-[10px] bg-divider pointer-events-none" />
      <div className={cn(
        "group ml-6 flex items-center gap-1.5 py-0.5 pr-1 rounded-sm transition-colors relative z-[1]",
        isActive ? "bg-surface-elevated" : "hover:bg-surface-elevated/50"
      )}>
        <button
          onClick={() => onFileClick?.(node.file)}
          className={cn("flex-1 text-left text-[11px] truncate", isActive ? "text-ink font-semibold" : "text-ink-muted hover:text-ink")}
        >
          {node.label}
          {hasKids && node.qaCount > 0 && <span className="text-ink-muted ml-0.5 text-[10px]">@{node.qaCount - 1}</span>}
        </button>
        {!mergeBlocked && (
          <button onClick={() => onMerge(node.file)} className="p-0.5 rounded text-ink-faint hover:text-ink transition-colors opacity-0 group-hover:opacity-100 shrink-0" title="Merge">
            <GitMerge className="h-3 w-3" />
          </button>
        )}
        {mergeBlocked && (
          <span className="p-0.5 rounded text-ink-faint/40 shrink-0 cursor-not-allowed opacity-0 group-hover:opacity-100 transition-colors" title="Parent has diverged past this branch point">
            <GitMerge className="h-3 w-3" />
          </span>
        )}
        <button onClick={() => onDelete(node.file)} className="p-0.5 rounded text-ink-faint hover:text-danger transition-colors opacity-0 group-hover:opacity-100 shrink-0" title="Delete">
          <Trash2 className="h-3 w-3" />
        </button>
      </div>

      {hasKids && (
        <div className="relative ml-6">
          <BranchTree root={node} />
        </div>
      )}
    </div>
  )
}

export interface ChatBranchItemProps {
  file: string
  label: string
  depth: number
}

const INDENT_BASE = 8
const INDENT_STEP = 16

export function ChatBranchItem({ file, label, depth }: ChatBranchItemProps) {
  const { branchMeta, chatIdMap, onFileClick, onMerge, onDelete, activeFile, activeQaIndex } = useBranchContext()
  const { onDragStart } = useTree<string>()
  const bm = branchMeta[file]
  const hasChildren = (bm?.children?.length ?? 0) > 0
  const isActive = activeFile === file && activeQaIndex == null
  const isRelevant = isActive || activeFile === file || isAncestorOf(activeFile, file, branchMeta, chatIdMap)
  const tree = useMemo(() => hasChildren && isRelevant ? buildTree(file, branchMeta, chatIdMap) : null, [hasChildren, isRelevant, file, branchMeta, chatIdMap])
  const qaLast = bm?.qa_count != null && bm.qa_count > 0 ? bm.qa_count - 1 : null
  const paddingLeft = INDENT_BASE + depth * INDENT_STEP

  return (
    <div style={{ paddingLeft }}>
      {tree && <BranchTree root={tree} />}
      <div className="flex items-center gap-1 py-0.5 group" style={{ paddingRight: 8 }}>
        <button
          draggable={!!onDragStart}
          onDragStart={onDragStart ? (e) => onDragStart(e, file) : undefined}
          onClick={() => onFileClick?.(file)}
          className={cn(
            "flex items-center gap-1.5 flex-1 truncate text-left text-[11px] transition-colors",
            isActive ? "text-ink font-medium" : "text-ink-muted hover:text-ink",
          )}
        >
          {label}
          {qaLast !== null && <span className="text-ink-muted text-[10px] ml-0.5">@{qaLast}</span>}
        </button>
        <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
          {onDelete && (
            <button
              onClick={() => onDelete(file)}
              className="p-0.5 rounded text-ink-faint hover:text-danger transition-colors shrink-0"
              title="Delete"
            >
              <Trash2 className="h-3 w-3" />
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
