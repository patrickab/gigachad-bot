"use client"

import dynamic from "next/dynamic"
import { DesktopBackendProvider } from "@/app/_components/DesktopBackendProvider"
import { ProjectProvider } from "@/contexts/ProjectContext"
import { SettingsProvider } from "@/contexts/SettingsContext"
import { StartupReadProvider } from "@/contexts/StartupReadContext"
import { UndoDeleteProvider } from "@/contexts/UndoDeleteContext"

// Client only: reads `window.location` on first render
const PiWorkbench = dynamic(() => import("@/features/pi/PiWorkbench").then((m) => ({ default: m.PiWorkbench })), { ssr: false })

/** Prototype: Pi coding agent sessions over its RPC mode, `/pi?repo=<name>`. */
export default function PiPage() {
  return (
    <DesktopBackendProvider>
      <StartupReadProvider>
        <SettingsProvider>
          <ProjectProvider>
            <UndoDeleteProvider>
              <PiWorkbench />
            </UndoDeleteProvider>
          </ProjectProvider>
        </SettingsProvider>
      </StartupReadProvider>
    </DesktopBackendProvider>
  )
}
