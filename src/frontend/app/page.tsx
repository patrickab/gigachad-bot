"use client"

import { DesktopBackendProvider } from "@/components/DesktopBackendProvider"
import { ProjectProvider } from "@/contexts/ProjectContext"
import { SettingsProvider } from "@/contexts/SettingsContext"
import { UndoDeleteProvider } from "@/contexts/UndoDeleteContext"
import { AppShell } from "./_components/AppShell"

export default function Home() {
  return (
    <DesktopBackendProvider>
      <SettingsProvider>
        <ProjectProvider>
          <UndoDeleteProvider>
            <div id="main-content" className="h-dvh">
              <AppShell />
            </div>
          </UndoDeleteProvider>
        </ProjectProvider>
      </SettingsProvider>
    </DesktopBackendProvider>
  )
}
