import { afterEach, describe, expect, it, vi } from "vitest"
import type { ReactNode } from "react"
import { cleanup, render } from "@testing-library/react"
import { AppShell } from "../AppShell"

const hints = vi.hoisted(() => ({ install: vi.fn(), stop: vi.fn() }))
vi.mock("@gigachad/chat-ui/hints", () => ({
  installClickHints: (...args: unknown[]) => {
    hints.install(...args)
    return hints.stop
  },
}))

const { pass } = vi.hoisted(() => ({ pass: ({ children }: { children?: ReactNode }) => children }))
vi.mock("@/contexts/SettingsContext", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/contexts/SettingsContext")>(),
  useSettings: () => ({ tabDefaults: {}, ompWebUrl: null, hostAccess: false }),
}))
vi.mock("@/contexts/ProjectContext", () => ({ useProject: () => ({ refreshProjects: async () => {} }) }))
vi.mock("@/features/navigation/BranchContext", () => ({ BranchProvider: pass, useBranches: () => ({ branchMeta: {} }) }))
vi.mock("@/features/navigation/SidebarContext", () => ({ SidebarProvider: pass }))
vi.mock("@/features/memory/MemoryViewerContext", () => ({ MemoryViewerProvider: pass }))
vi.mock("@/features/memory/MemoryViewer", () => ({ MemoryViewer: () => null }))
vi.mock("@/features/projects/ProjectDashboard", () => ({ ProjectDashboard: () => null }))
vi.mock("@/features/navigation/TabManager", () => ({ TabManager: pass }))
vi.mock("@/features/chat/EnabledToolsContext", () => ({ EnabledToolsProvider: pass }))
vi.mock("@/features/editor/FilePanel", () => ({ FilePanelProvider: pass }))
vi.mock("@/features/editor/ShellTerminal", () => ({ ShellTerminals: () => null }))
vi.mock("@/features/navigation/ProviderUsage", () => ({ ProviderUsagePoll: () => null }))
vi.mock("@/features/learn/LearnProvider", () => ({ LearnProvider: pass }))
vi.mock("@/features/pi/sessionStore", () => ({ PiSessionsProvider: pass, markPiSessionRead: () => {} }))
vi.mock("@/features/pi/notifications", () => ({ startPiNotifications: () => () => {} }))
vi.mock("@/features/notebook/useKernelShutdown", () => ({ useKernelShutdown: () => {} }))
vi.mock("@/lib/backend/fileEvents", () => ({ onFileMove: () => () => {} }))
vi.mock("@/lib/backend/resources", () => ({ watchDocumentLists: () => () => {} }))
vi.mock("../TabContent", () => ({ TabContent: () => null }))

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe("AppShell click hints", () => {
  it("installs the shell's frameless hints once, the omp provider no longer adds its own", () => {
    const { unmount } = render(<AppShell />)

    expect(hints.install).toHaveBeenCalledTimes(1)
    expect(hints.install).toHaveBeenCalledWith()

    unmount()
    expect(hints.stop).toHaveBeenCalledTimes(1)
  })
})
