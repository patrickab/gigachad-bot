"use client"

import { Archive, Clock, Folder, Loader2, Trash2 } from "lucide-react"
import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { archiveChatHistory, deleteChatHistory, loadChatHistory } from "@/lib/api"
import { useSafeAction } from "@/hooks/useSafeAction"
import { SidebarElement, ChevronToggle } from "./SidebarElement"
import { Skeleton } from "./Skeleton"

const pathOf = (dir: string, filename: string) => dir === "root" ? filename : `${dir}/${filename}`

interface ChatHistoryManagerProps {
  histories: Record<string, string[]>
  historiesLoading?: boolean
  onLoad: (filename: string) => void
  onRefresh: () => void
  collapsed?: boolean
  open: boolean
  onOpenChange: (open: boolean) => void
  onExpand?: () => void
}

export function ChatHistoryManager({ histories, historiesLoading, onLoad, onRefresh, collapsed, open, onOpenChange, onExpand }: ChatHistoryManagerProps) {
  const [expandedDirs, setExpandedDirs] = useState<Set<string>>(new Set())
  const [loadingFile, setLoadingFile] = useState<string | null>(null)
  const { run, error } = useSafeAction()

  const dirs = Object.entries(histories)
  const totalFiles = dirs.reduce((sum, [, files]) => sum + files.length, 0)

  const handleLoad = (filename: string, dir: string) => {
    const path = pathOf(dir, filename)
    setLoadingFile(path)
    return run("load history", async () => {
      await loadChatHistory(path)
      onLoad(path)
    }).finally(() => setLoadingFile(null))
  }

  const handleDelete = (filename: string, dir: string) =>
    run("delete history", async () => { await deleteChatHistory(pathOf(dir, filename)); onRefresh() })

  const handleArchive = (filename: string, dir: string) =>
    run("archive history", async () => { await archiveChatHistory(pathOf(dir, filename)); onRefresh() })

  return (
    <div className="space-y-1 w-full">
      {collapsed ? (
          <SidebarElement
            icon={Clock}
            title="Histories"
          collapsed={true}
          onClick={() => {
            if (onExpand) onExpand()
            onOpenChange(true)
          }}
        />
      ) : (
        <button
            onClick={() => onOpenChange(!open)}
          className="flex w-full items-center justify-between p-2 rounded-md transition-colors text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-200"
        >
          <span className="flex items-center gap-3">
            <Clock className="h-4 w-4 shrink-0" />
            <span className="text-sm font-medium">Histories</span>
            {totalFiles > 0 && (
              <span className="rounded-full bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-500">{totalFiles}</span>
            )}
          </span>
          <ChevronToggle open={open} />
        </button>
      )}

      <AnimatePresence>
        {!collapsed && open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="space-y-1 pl-1">
              {historiesLoading ? (
                <div className="space-y-1.5 px-2 py-1">
                  <Skeleton className="h-3 w-24" />
                  <Skeleton className="h-3 w-32" />
                  <Skeleton className="h-3 w-20" />
                </div>
              ) : (
                <>
                  {dirs.map(([dir, files]) => (
                    <div key={dir}>
                      <button
                        onClick={() =>
                          setExpandedDirs((prev) => {
                            const next = new Set(prev)
                            next.has(dir) ? next.delete(dir) : next.add(dir)
                            return next
                          })
                        }
                        className="flex items-center gap-1.5 px-2 py-1 text-[11px] text-zinc-600 hover:text-zinc-400 transition-colors w-full"
                      >
                        <Folder className="h-3 w-3" />
                        {dir}
                      </button>
                      <AnimatePresence>
                        {expandedDirs.has(dir) &&
                          files.map((file) => {
                            const path = pathOf(dir, file)
                            return (
                              <motion.div
                                key={path}
                                initial={{ opacity: 0, height: 0 }}
                                animate={{ opacity: 1, height: "auto" }}
                                exit={{ opacity: 0, height: 0 }}
                                className="flex items-center gap-1 pl-5 pr-2 py-0.5"
                              >
                                <button
                                  onClick={() => handleLoad(file, dir)}
                                  disabled={loadingFile === path}
                                  className="flex-1 truncate text-left text-[11px] text-zinc-400 hover:text-zinc-200 transition-colors"
                                >
                                  {loadingFile === path ? (
                                    <Loader2 className="inline h-3 w-3 animate-spin mr-1" />
                                  ) : null}
                                  {file.replace(".json", "")}
                                </button>
                                <button
                                  onClick={() => handleArchive(file, dir)}
                                  className="rounded p-0.5 text-zinc-600 hover:text-amber-400 transition-colors"
                                >
                                  <Archive className="h-3 w-3" />
                                </button>
                                <button
                                  onClick={() => handleDelete(file, dir)}
                                  className="rounded p-0.5 text-zinc-600 hover:text-red-400 transition-colors"
                                >
                                  <Trash2 className="h-3 w-3" />
                                </button>
                              </motion.div>
                            )
                          })}
                      </AnimatePresence>
                    </div>
                  ))}
                  {error && (
                    <p className="px-2 py-1 text-[11px] text-red-400">{error}</p>
                  )}
                  {dirs.length === 0 && (
                    <p className="px-2 py-2 text-[11px] text-zinc-600">No saved chats.</p>
                  )}
                </>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}