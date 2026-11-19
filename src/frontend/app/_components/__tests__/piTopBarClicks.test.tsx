import { afterEach, describe, expect, it, vi } from "vitest"
import { createPortal } from "react-dom"
import { cleanup, render, screen } from "@testing-library/react"
import { TabContext, type Tab } from "@/contexts/TabContext"
import { TopBarSlotProvider } from "@/features/navigation/TopBarSlots"
import { WorkspaceSurface } from "../WorkspaceSurface"

// The Pi tab portals its controls into the top bar slot it gets, as the real one does.
vi.mock("@/features/pi/PiTab", () => ({
  PiTab: ({ topBar }: { topBar: HTMLElement | null }) => topBar ? createPortal(<button>Model: Default</button>, topBar) : null,
}))
vi.mock("@/features/pi/PiSessionPicker", () => ({ PiSessionPicker: () => null }))
vi.mock("@/features/editor/FilePanel", () => ({ FilePanelToggle: () => null }))
vi.mock("@/features/settings/ThemeToggle", () => ({ ThemeToggle: () => null }))

const TAB = {
  id: "t", chatId: "c", config: {}, projectSlug: null, appMode: "pi",
  workspaceItem: { kind: "pi", path: "repo", projectSlug: null, pi: { sessionId: null, repo: "repo" } },
} as unknown as Tab
const noop = () => {}

afterEach(cleanup)

/** The nearest `pointer-events-*` utility on `element` or an ancestor, what decides whether a real click lands. */
function pointerEvents(element: HTMLElement): string | undefined {
  for (let node: HTMLElement | null = element; node; node = node.parentElement) {
    const match = [...node.classList].find((name) => name === "pointer-events-none" || name === "pointer-events-auto")
    if (match) return match
  }
  return undefined
}

describe("WorkspaceSurface Pi top bar", () => {
  it("lets the Pi tab's controls take clicks inside the click-through top bar", () => {
    render(
      <TabContext.Provider value={{ tab: TAB, isActive: true, setModeLabel: noop, updateConfig: noop, setAppMode: noop, setWorkspaceItem: noop, setHistoryFile: noop, setTitle: noop }}>
        <TopBarSlotProvider>
          <WorkspaceSurface
            sidebarToggle={null}
            canvas={{} as never}
            modelProps={{} as never}
            reasoningLevels={undefined}
            reasoningEffort="medium"
            totalUsage={{} as never}
            chatColumn={null}
            chatBody={null}
            shellCommands={{} as never}
            launch={null}
            onLaunched={noop}
            onOpenItem={noop}
          />
        </TopBarSlotProvider>
      </TabContext.Provider>,
    )
    const trigger = screen.getByRole("button", { name: "Model: Default" })
    expect(trigger.closest("header")).toHaveClass("pointer-events-none")
    expect(pointerEvents(trigger)).toBe("pointer-events-auto")
  })
})
