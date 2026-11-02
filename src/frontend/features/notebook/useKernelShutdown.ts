"use client"

import { useEffect } from "react"
import { shutdownKernel } from "@/lib/backend/api"
import { onFileClosed } from "@/lib/backend/fileEvents"

/**
 * Shut the kernel of a notebook down when its last tab lets go of it: the tab closed, the file was deleted or the tab opened another file.
 *
 * - Mounted once for the app, so it also hears the closing of a notebook whose editor was already unmounted (its tab stayed, showing the chat)
 * - A rename moves the kernel on the backend first, so the old path then has none to shut down
 * - An editor that goes away with its tab still open (switching tabs, reloading the page) leaves the kernel to the backend's idle cull
 */
export function useKernelShutdown(): void {
  useEffect(() => onFileClosed((path) => {
    if (path.endsWith(".ipynb")) shutdownKernel(path).catch(() => {})
  }), [])
}
