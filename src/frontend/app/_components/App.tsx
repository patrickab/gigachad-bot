"use client"

import { DesktopBackendProvider } from "@/app/_components/DesktopBackendProvider"
import { ProjectProvider } from "@/contexts/ProjectContext"
import { SettingsProvider } from "@/contexts/SettingsContext"
import { UndoDeleteProvider } from "@/contexts/UndoDeleteContext"
import { StartupReadProvider } from "@/contexts/StartupReadContext"
import { ErrorBoundary } from "@/components/ui/ErrorBoundary"
import { AppShell } from "@/app/_components/AppShell"

/** The whole app: the shell inside the providers every tab shares. */
export function App() {
  return (
    <ErrorBoundary fullscreen resetLabel="Reload">
    <DesktopBackendProvider>
      <StartupReadProvider>
        <SettingsProvider>
          <ProjectProvider>
            <UndoDeleteProvider>
              <div id="main-content" className="h-dvh">
                <AppShell />
              </div>
            </UndoDeleteProvider>
          </ProjectProvider>
        </SettingsProvider>
      </StartupReadProvider>
    </DesktopBackendProvider>
    </ErrorBoundary>
  )
}
